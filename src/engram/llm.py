"""
llm.py — LLM callback protocol and prompt builders for Engram.

Engram does NOT call LLM APIs directly. Instead, it defines a callback protocol
that the host agent (e.g. echo-code) injects. Each function in this module:
  1. Builds a prompt
  2. Calls the injected llm_fn
  3. Parses the response

When no llm_fn is provided, the legacy HTTP-based providers in extract.py
and rerank.py are used as a fallback (for standalone CLI usage).
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
# LLM Callback Protocol
# ═══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class LLMCallback(Protocol):
    """Protocol for LLM invocation.

    The host agent provides an implementation that sends the prompt to
    whichever model it is using.  Engram only cares about the interface::

        def my_llm(prompt: str, system: str = "", **kwargs) -> Optional[str]:
            return call_my_model(prompt, system_message=system)

        stack = MemoryStack(config=cfg, llm_fn=my_llm)

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
# Query Rewrite
# ═══════════════════════════════════════════════════════════════════════════

_REWRITE_SYSTEM = (
    "You are a search query expander. Given a vague or short question, "
    "rewrite it as a more specific query with related keywords to improve "
    "semantic search recall. Output ONLY the expanded query — no explanation."
)

_REWRITE_PROMPT = """\
Expand this question into a more specific search query with related terms.

Original question: {query}

Rules:
- Keep the original intent
- Add synonyms and related terms
- Keep it under 50 words
- Output ONLY the expanded query, nothing else"""


def rewrite_query(query: str, llm_fn: LLMCallback) -> str:
    """Rewrite a vague query into a more specific search query using LLM.

    Args:
        query: The original user query.
        llm_fn: LLM callback function.

    Returns:
        Expanded query string, or the original query on failure.
    """
    prompt = _REWRITE_PROMPT.format(query=query)
    try:
        result = llm_fn(prompt, system=_REWRITE_SYSTEM, temperature=0.0, max_tokens=128)
        if result and result.strip():
            expanded = result.strip().strip('"').strip("'")
            # Sanity: if the LLM returned something wildly different or too long, skip
            if len(expanded) > 500 or len(expanded) < 3:
                return query
            return expanded
    except Exception as exc:
        logger.debug("Query rewrite failed: %s", exc)
    return query


# ═══════════════════════════════════════════════════════════════════════════
# Reranking
# ═══════════════════════════════════════════════════════════════════════════

_RERANK_SYSTEM = (
    "You are a search relevance judge. Output ONLY a JSON array of document "
    "numbers (1-based), most relevant first."
)

# Maximum characters per candidate in the reranking prompt.
_MAX_DOC_CHARS = 300


def rerank_with_llm(
    query: str,
    candidates: list,
    llm_fn: LLMCallback,
    top_k: int = 5,
) -> list:
    """Rerank search candidates using the LLM callback.

    Args:
        query: The user's search query.
        candidates: List of SearchHit objects from vector_search().
        llm_fn: LLM callback function.
        top_k: Number of results to return.

    Returns:
        Reranked list of SearchHit objects (length <= top_k).
        Falls back to candidates[:top_k] on any failure.
    """
    if not candidates:
        return []

    # Build prompt
    doc_lines = []
    for i, hit in enumerate(candidates):
        content = hit.content.replace("\n", " ").strip()
        if len(content) > _MAX_DOC_CHARS:
            content = content[:_MAX_DOC_CHARS] + "..."
        doc_lines.append(f"[{i+1}] {content}")

    documents_block = "\n".join(doc_lines)
    prompt = (
        f"Given a question and a list of documents, rank them by relevance.\n\n"
        f"Return ONLY a JSON array of document numbers (1-based), most relevant first. "
        f"Return the top {top_k} most relevant documents.\n\n"
        f"Example output: [3, 1, 7, 5, 2]\n\n"
        f"Question: {query}\n\n"
        f"Documents:\n{documents_block}\n\n"
        f"Output (top {top_k} document numbers, JSON array):"
    )

    try:
        response = llm_fn(prompt, system=_RERANK_SYSTEM, temperature=0.0, max_tokens=256)
    except Exception as exc:
        logger.debug("Rerank LLM call failed: %s", exc)
        return candidates[:top_k]

    if not response:
        return candidates[:top_k]

    # Parse response
    indices = _parse_rerank_indices(response, len(candidates), top_k)
    if indices is None:
        return candidates[:top_k]

    # Map indices back to candidates
    reranked = []
    seen: set = set()
    for idx in indices:
        if 0 <= idx < len(candidates) and idx not in seen:
            reranked.append(candidates[idx])
            seen.add(idx)
        if len(reranked) >= top_k:
            break

    # Fill remaining from original order
    if len(reranked) < top_k:
        for i, c in enumerate(candidates):
            if i not in seen:
                reranked.append(c)
                seen.add(i)
            if len(reranked) >= top_k:
                break

    return reranked


def _parse_rerank_indices(
    response: str, n_candidates: int, top_k: int
) -> Optional[List[int]]:
    """Parse LLM rerank response into 0-based indices.

    Handles JSON arrays, markdown-wrapped JSON, and plain numbers.
    """
    text = response.strip()

    # Try 1: JSON array
    json_match = re.search(r'\[[\d\s,]+\]', text)
    if json_match:
        try:
            arr = json.loads(json_match.group())
            seen: set = set()
            indices = []
            for x in arr:
                if isinstance(x, (int, float)):
                    idx = int(x) - 1  # 1-based → 0-based
                    if 0 <= idx < n_candidates and idx not in seen:
                        seen.add(idx)
                        indices.append(idx)
            if indices:
                return indices[:top_k]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try 2: Extract numbers
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        seen = set()
        indices = []
        for n_str in numbers:
            idx = int(n_str) - 1
            if 0 <= idx < n_candidates and idx not in seen:
                seen.add(idx)
                indices.append(idx)
        if indices:
            return indices[:top_k]

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Query Detection & Reasoning
# ═══════════════════════════════════════════════════════════════════════════

_TEMPORAL_MARKERS = re.compile(
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


def is_temporal_query(query: str) -> bool:
    """Check if a query contains temporal markers."""
    return bool(_TEMPORAL_MARKERS.search(query))


def answer_temporal(
    query: str,
    hits: list,
    llm_fn: LLMCallback,
) -> Optional[str]:
    """Use LLM to reason about a time-related question.

    Args:
        query: The user's temporal question.
        hits: Search hits with content and timestamps.
        llm_fn: LLM callback function.

    Returns:
        An answer string, or None if reasoning fails or query is not temporal.
    """
    if not is_temporal_query(query):
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
        result = llm_fn(prompt, system=_TEMPORAL_SYSTEM, temperature=0.0, max_tokens=256)
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
    llm_fn: LLMCallback,
    project: str = None,
    existing_facts: list = None,
) -> list:
    """Extract facts using the LLM callback.

    Args:
        content: Text to extract facts from.
        llm_fn: LLM callback function.
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
        response = llm_fn(prompt, system=_EXTRACT_SYSTEM, temperature=0.1, max_tokens=1024)
        if response:
            return _parse_llm_response(response)
    except Exception as exc:
        logger.debug("Fact extraction via callback failed: %s", exc)

    return []
