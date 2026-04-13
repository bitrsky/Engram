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


# ═══════════════════════════════════════════════════════════════════════════
# Deep Search — Prompt & Response Parsing
# ═══════════════════════════════════════════════════════════════════════════

DEEP_SEARCH_SYSTEM = """\
You are a memory search assistant. You have access to a personal knowledge base \
stored on disk. The knowledge base has three layers:

## Directory Structure

  {base_dir}/
  ├── identity.md              # User identity and preferences
  ├── memories/                # Conversation memories (markdown)
  │   ├── 2024-01-15_auth-decision.md
  │   └── ...
  ├── facts/                   # Knowledge graph (per-project)
  │   ├── project-name.md      # Facts as subject|predicate|object triples
  │   └── ...
  └── projects/                # Project registry
      ├── project-name.md      # Project metadata, status, aliases
      └── ...

## File Formats

### memories/*.md
YAML frontmatter with id, project, topics, memory_type, importance, created.
Body contains the raw conversation or note text.
Example frontmatter:
  ---
  id: session_5
  project: my-project
  topics: [auth, infrastructure]
  memory_type: decision
  importance: 4.0
  created: 2024-01-15T14:23:00
  ---
  We decided to use Clerk for authentication because...

### facts/*.md
Knowledge graph stored as structured lines:
  - subject | predicate | object | since: YYYY-MM | confidence: 0.95
    - source: mem_xxx
Sections: ## Current, ## Expired, ## Conflicts
Use this to find factual relationships (who uses what, preferences, tech stack).

### projects/*.md
Project metadata: display_name, status (active/paused/archived), aliases.
Use this to understand what projects exist and their current state.

## Search Strategy

1. Check the vector search hints first — they may already contain the answer.
2. For factual questions (what does X use, who prefers Y), check facts/ first.
3. For event/conversation questions, grep or read memories/.
4. Use grep to search across files efficiently, then read specific files.

## Response Format (MUST follow exactly)

ANSWER: {{your answer}}
EVIDENCE: {{comma-separated memory IDs (e.g. session_5, session_12) or fact file names}}

If not found:
ANSWER: NOT_FOUND
EVIDENCE: none
"""


def build_deep_search_prompt(
    query: str,
    base_dir: str,
    vector_hits: list,
) -> tuple[str, str]:
    """Build system prompt and user prompt for deep search.

    Args:
        query: The user's question.
        base_dir: Path to the engram base directory (e.g. ~/.engram).
        vector_hits: Vector search top-K results (SearchHit objects).

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    system = DEEP_SEARCH_SYSTEM.format(base_dir=base_dir)

    # Build vector search hints
    hint_lines = []
    for i, hit in enumerate(vector_hits[:10]):
        content = hit.content.strip()
        if len(content) > 300:
            content = content[:300] + "..."
        created = getattr(hit, "created", "") or ""
        hit_id = getattr(hit, "id", f"hit_{i}")
        file_path = getattr(hit, "file_path", "") or ""
        hit_project = getattr(hit, "project", "") or ""
        similarity = getattr(hit, "similarity", 0.0)
        memory_type = getattr(hit, "memory_type", "conversation") or "conversation"

        hint_lines.append(
            f"[{i+1}] type={memory_type} | id={hit_id} | similarity={similarity:.2f} | "
            f"project={hit_project} | created={created}"
        )
        if file_path:
            hint_lines.append(f"    file: {file_path}")
        if memory_type == "fact":
            hint_lines.append(f"    fact: {hit.content.strip()}")
        else:
            hint_lines.append(f"    preview: {content}")
        hint_lines.append("")

    hints_block = "\n".join(hint_lines) if hint_lines else "(no vector results)"

    user_prompt = (
        f"Question: {query}\n\n"
        f"=== Vector search hints (ranked by semantic similarity) ===\n"
        f"{hints_block}\n\n"
        f"Each hint includes a file path — you can use read to get the full content.\n"
        f"If the hints answer the question, respond with the memory IDs.\n"
        f"Otherwise, search the knowledge base at {base_dir}:\n"
        f"- grep in memories/ for keywords\n"
        f"- check facts/ for factual relationships\n"
        f"- read specific files for full context\n"
        f"EVIDENCE must list memory IDs (like session_5), not filenames."
    )

    return system, user_prompt


def parse_deep_search_response(response: str) -> dict:
    """Parse a deep search LLM response.

    Extracts ANSWER and EVIDENCE fields from the response text.

    Args:
        response: Raw LLM response containing ANSWER: and EVIDENCE: lines.

    Returns:
        dict with keys:
            - "answer": str (the answer text, or "NOT_FOUND")
            - "evidence": list[str] (memory IDs or fact file names)
            - "found": bool (True if answer is not NOT_FOUND)
            - "raw": str (original response)
    """
    if not response:
        return {"answer": "NOT_FOUND", "evidence": [], "found": False, "raw": ""}

    answer = "NOT_FOUND"
    evidence: List[str] = []

    # Extract ANSWER line
    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()

    # Extract EVIDENCE line
    evidence_match = re.search(r"EVIDENCE:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if evidence_match:
        raw_evidence = evidence_match.group(1).strip()
        if raw_evidence.lower() != "none":
            evidence = [e.strip() for e in raw_evidence.split(",") if e.strip()]

    found = answer.upper() != "NOT_FOUND" and bool(answer)

    return {
        "answer": answer,
        "evidence": evidence,
        "found": found,
        "raw": response,
    }
