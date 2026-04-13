"""
rerank.py -- LLM-powered reranking for Engram search results.

Two-stage retrieval: vector search returns top-N candidates, then LLM
reranks them to produce top-K final results.

Supports the same providers as extract.py (ollama, openai, anthropic)
via raw HTTP -- zero additional dependencies.

Falls back gracefully to original vector ranking when:
  - LLM provider is "none"
  - LLM call fails (timeout, parse error, etc.)
  - rerank is explicitly disabled in config
"""

import json
import logging
import re
import urllib.error
import urllib.request
from typing import List, Optional

from .config import EngramConfig

logger = logging.getLogger(__name__)

# Maximum characters per candidate in the reranking prompt.
# Keeps total prompt under ~1500 tokens for 20 candidates.
_MAX_DOC_CHARS = 300


def rerank(
    query: str,
    candidates: list,
    config: EngramConfig,
    top_k: int = 5,
) -> list:
    """
    Rerank search candidates using LLM.

    Args:
        query: The user's search query.
        candidates: List of SearchHit objects from vector_search().
        config: EngramConfig (must have LLM configured).
        top_k: Number of results to return after reranking.

    Returns:
        Reranked list of SearchHit objects (length <= top_k).
        Falls back to candidates[:top_k] on any failure.
    """
    if not candidates:
        return []

    if not config.rerank_enabled:
        return candidates[:top_k]

    # Build the prompt
    prompt = _build_rerank_prompt(query, candidates, top_k)

    # Call LLM
    response = _call_llm_for_rerank(prompt, config)
    if response is None:
        logger.debug("Rerank: LLM call failed, falling back to vector order")
        return candidates[:top_k]

    # Parse response into ordered indices
    indices = _parse_rerank_response(response, len(candidates), top_k)
    if indices is None:
        logger.debug("Rerank: failed to parse LLM response, falling back")
        return candidates[:top_k]

    # Map indices back to candidates
    reranked = []
    seen = set()
    for idx in indices:
        if 0 <= idx < len(candidates) and idx not in seen:
            reranked.append(candidates[idx])
            seen.add(idx)
        if len(reranked) >= top_k:
            break

    # If LLM returned fewer than top_k, fill with remaining candidates
    if len(reranked) < top_k:
        for i, c in enumerate(candidates):
            if i not in seen:
                reranked.append(c)
                seen.add(i)
            if len(reranked) >= top_k:
                break

    return reranked


def _build_rerank_prompt(
    query: str, candidates: list, top_k: int
) -> str:
    """
    Build a listwise reranking prompt.

    The prompt asks the LLM to return a JSON array of 1-based document
    indices sorted by relevance to the query.
    """
    doc_lines = []
    for i, hit in enumerate(candidates):
        # Truncate content to keep prompt compact
        content = hit.content.replace("\n", " ").strip()
        if len(content) > _MAX_DOC_CHARS:
            content = content[:_MAX_DOC_CHARS] + "..."
        doc_lines.append(f"[{i+1}] {content}")

    documents_block = "\n".join(doc_lines)

    prompt = (
        "You are a search relevance judge. Given a question and a list of documents, "
        "rank them by relevance to answering the question.\n\n"
        "Return ONLY a JSON array of document numbers (1-based), most relevant first. "
        f"Return the top {top_k} most relevant documents.\n\n"
        "Example output: [3, 1, 7, 5, 2]\n\n"
        f"Question: {query}\n\n"
        f"Documents:\n{documents_block}\n\n"
        f"Output (top {top_k} document numbers, JSON array):"
    )
    return prompt


def _parse_rerank_response(
    response: str, n_candidates: int, top_k: int
) -> Optional[List[int]]:
    """
    Parse LLM response into a list of 0-based indices.

    Handles:
      - Clean JSON array: [3, 1, 7]
      - JSON in markdown: ```json [3, 1, 7] ```
      - Comma-separated numbers: 3, 1, 7
      - Numbers with other text: "The most relevant are 3, 1, and 7"

    Returns None if parsing fails completely.
    """
    if not response:
        return None

    text = response.strip()

    # Try 1: Extract JSON array
    json_match = re.search(r'\[[\d\s,]+\]', text)
    if json_match:
        try:
            arr = json.loads(json_match.group())
            # Convert 1-based to 0-based, deduplicate preserving order
            seen = set()
            indices = []
            for x in arr:
                if isinstance(x, (int, float)):
                    idx = int(x) - 1
                    if 0 <= idx < n_candidates and idx not in seen:
                        seen.add(idx)
                        indices.append(idx)
            if indices:
                return indices[:top_k]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Try 2: Extract all numbers from the response
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        indices = []
        for n_str in numbers:
            idx = int(n_str) - 1  # 1-based to 0-based
            if 0 <= idx < n_candidates:
                indices.append(idx)
        if indices:
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique.append(idx)
            return unique[:top_k]

    return None


# ── LLM calling (reuses patterns from extract.py) ──────────────────────


def _call_llm_for_rerank(prompt: str, config: EngramConfig) -> Optional[str]:
    """
    Call LLM for reranking. Supports ollama, openai, anthropic.
    Uses shorter timeout than extract (reranking should be fast).
    """
    provider = config.llm_provider
    if provider == "none" or provider == "":
        return None

    try:
        if provider == "ollama":
            return _call_ollama(prompt, config)
        elif provider == "openai":
            return _call_openai(prompt, config)
        elif provider == "anthropic":
            return _call_anthropic(prompt, config)
    except Exception as exc:
        logger.debug("Rerank LLM call failed: %s", exc)

    return None


def _call_ollama(prompt: str, config: EngramConfig) -> Optional[str]:
    """Call Ollama API."""
    base_url = config.llm_base_url or "http://localhost:11434"
    url = f"{base_url}/api/generate"
    payload = {
        "model": config.llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},  # Deterministic for ranking
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "")
    except (urllib.error.URLError, urllib.error.HTTPError,
            OSError, json.JSONDecodeError):
        return None


def _call_openai(prompt: str, config: EngramConfig) -> Optional[str]:
    """Call OpenAI-compatible API."""
    base_url = config.llm_base_url or "https://api.openai.com"
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": config.llm_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a search relevance judge. Output only a JSON array of numbers.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.llm_api_key}",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
    except (urllib.error.URLError, urllib.error.HTTPError,
            OSError, json.JSONDecodeError):
        pass
    return None


def _call_anthropic(prompt: str, config: EngramConfig) -> Optional[str]:
    """Call Anthropic API."""
    base_url = config.llm_base_url or "https://api.anthropic.com"
    url = f"{base_url}/v1/messages"
    payload = {
        "model": config.llm_model,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": config.llm_api_key,
        "anthropic-version": "2023-06-01",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content_blocks = body.get("content", [])
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
    except (urllib.error.URLError, urllib.error.HTTPError,
            OSError, json.JSONDecodeError):
        pass
    return None
