"""
llm.py — Agent thinking protocol and prompt builders for Engram.

Engram does NOT call LLM APIs directly. Instead, it defines a ThinkFn protocol
that the host agent (e.g. echo-code) injects. Each function in this module:
  1. Builds a prompt
  2. Calls the injected think_fn
  3. Parses the response

When no think_fn is provided, Engram falls back to heuristic-only mode.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Callable, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .extract import FactCandidate

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ThinkFn Protocol
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class ThinkFn(Protocol):
    """Protocol for agent thinking/inference capability.

    The host agent (e.g. echo-code) provides an implementation that
    processes a prompt and returns a response.  Engram only cares
    about the interface::

        def my_think(prompt: str, system: str = "", **kwargs) -> Optional[str]:
            return agent.prompt_collect(prompt)

        stack = MemoryStack(config=cfg, think_fn=my_think)

    Keyword arguments (``**kwargs``) may include:
        temperature (float): Sampling temperature.
        max_tokens (int):    Maximum response tokens.
        timeout (int):       Timeout in seconds.
    """

    def __call__(
        self,
        prompt: str,
        system: str = "",
        **kwargs,
    ) -> Optional[str]: ...


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Query Detection & Reasoning
# ═══════════════════════════════════════════════════════════════════════════

# Fallback compiled regex for when no config is provided.
# The canonical pattern list lives in config._BUILTIN_PATTERNS["temporal"]["markers"].
_TEMPORAL_MARKERS_FALLBACK = re.compile(
    r"\b("
    r"when did|when was|when were|when is|when are|"
    r"how long ago|how many days|how many weeks|how many months|how many years|"
    r"days ago|weeks ago|months ago|years ago|"
    r"since when|last time|first time|"
    r"before|after|during|between .+ and|"
    r"timeline|chronolog|sequence of events"
    r")\b",
    re.IGNORECASE,
)

_TEMPORAL_SYSTEM = (
    "You are a temporal reasoning assistant. Given a time-related question "
    "and memory excerpts with timestamps, reason about the answer. "
    "Be precise about dates and durations. "
    "Output only the answer — no preamble."
)

_TEMPORAL_PROMPT = """\
Question: {query}

Memory excerpts (with dates):
{context}

Based on these memory excerpts, answer the question about timing/dates.
If you cannot determine the answer, say "Unable to determine from available memories."
Answer:"""


def is_temporal_query(query: str, config=None) -> bool:
    """
    Check if a query contains temporal markers.

    Args:
        query: The search query string.
        config: Optional EngramConfig — when provided, uses user-configured
                temporal patterns (which may include non-English markers).
    """
    if config is not None:
        patterns = config.temporal_markers
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    return bool(_TEMPORAL_MARKERS_FALLBACK.search(query))


def answer_temporal(
    query: str,
    hits: list,
    think_fn: ThinkFn,
    config=None,
) -> Optional[str]:
    """Reason about a time-related question.

    Args:
        query: The user's temporal question.
        hits: Search hits with content and timestamps.
        think_fn: Agent thinking function.
        config: Optional EngramConfig for pattern configuration.

    Returns:
        An answer string, or None if reasoning fails or query is not temporal.
    """
    if not is_temporal_query(query, config):
        return None

    if not hits:
        return None

    # Build context from hits
    context_lines = []
    for hit in hits:
        date = getattr(hit, "created", "") or ""
        if isinstance(date, str):
            date = date[:10]  # YYYY-MM-DD
        content_preview = hit.content.replace("\n", " ").strip()
        if len(content_preview) > 200:
            content_preview = content_preview[:200] + "..."
        context_lines.append(f"[{date}] {content_preview}")

    context = "\n".join(context_lines)
    prompt = _TEMPORAL_PROMPT.format(query=query, context=context)

    try:
        result = think_fn(prompt, system=_TEMPORAL_SYSTEM, temperature=0.0, max_tokens=256)
        if result and result.strip():
            answer = result.strip()
            if "unable to determine" in answer.lower():
                return None
            return answer
    except Exception as exc:
        logger.debug("Temporal reasoning failed: %s", exc)

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Fact Extraction via Callback
# ═══════════════════════════════════════════════════════════════════════════

_EXTRACT_SYSTEM = "You are a fact extraction engine. Output only valid JSON."

_EXTRACT_PROMPT = """\
Extract structured facts from the following text.

Project context: {project}
{known_facts_section}
Text to analyze:
---
{content}
---

Extract ALL factual claims as (subject, predicate, object) triples.
Include: decisions, assignments, technical choices, timelines, concerns, metrics, relationships.
Skip: opinions about code quality, generic statements, pleasantries, hypotheticals.

For each fact:
- subject: the entity (person, project, component)
- predicate: the relationship (uses, assigned_to, decided, status, etc.)
- object: the value
- confidence: 0.0-1.0 based on how certain this fact is from the text
- temporal: ISO date or relative time phrase if present, empty string otherwise
- conflicts_with: "subject → predicate → object" of a conflicting known fact, or empty string

Output ONLY a valid JSON array, no markdown formatting:
[
  {{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9, "temporal": "", "conflicts_with": ""}}
]"""


def extract_facts_via_callback(
    content: str,
    think_fn: ThinkFn,
    project: Optional[str] = None,
    existing_facts: Optional[list] = None,
) -> list:
    """Extract facts using agent thinking.

    Args:
        content: Text to extract facts from.
        think_fn: Agent thinking function.
        project: Project context.
        existing_facts: Known facts for conflict detection.

    Returns:
        List of FactCandidate objects (imported lazily to avoid circular imports).
    """
    from .extract import FactCandidate, _parse_llm_response

    # Build known facts section
    known_facts_section = ""
    if existing_facts:
        facts_lines = []
        for f in existing_facts[:20]:
            facts_lines.append(f"  - {f.subject} → {f.predicate} → {f.object}")
        known_facts_section = (
            "\nKnown facts about this project:\n"
            + "\n".join(facts_lines)
            + "\n\nIf any extracted fact contradicts a known fact above, "
            'note it in "conflicts_with".\n'
        )

    prompt = _EXTRACT_PROMPT.format(
        project=project or "(unknown)",
        known_facts_section=known_facts_section,
        content=content,
    )

    try:
        response = think_fn(prompt, system=_EXTRACT_SYSTEM, temperature=0.1, max_tokens=1024)
        if response:
            return _parse_llm_response(response)
    except Exception as exc:
        logger.debug("Fact extraction via callback failed: %s", exc)

    return []
