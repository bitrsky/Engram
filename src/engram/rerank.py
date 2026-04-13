"""
rerank.py -- LLM-powered reranking for Engram search results.

Two-stage retrieval: vector search returns top-N candidates, then LLM
reranks them to produce top-K final results.

The host agent (e.g. echo-code) injects LLM capability via the think_fn callback.
Engram never calls LLM APIs directly.

Falls back gracefully to original vector ranking when:
  - No think_fn callback is provided
  - LLM call fails (timeout, parse error, etc.)
  - rerank is explicitly disabled in config
"""

import json
import logging
import re
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
    think_fn=None,
) -> list:
    """
    Rerank search candidates using LLM.

    Priority:
        1. think_fn callback (when host agent provides one)
        2. No reranking (return original vector order)

    Args:
        query: The user's search query.
        candidates: List of SearchHit objects from vector_search().
        config: EngramConfig.
        top_k: Number of results to return after reranking.
        think_fn: Optional agent thinking function (see engram.llm.ThinkFn).

    Returns:
        Reranked list of SearchHit objects (length <= top_k).
        Falls back to candidates[:top_k] on any failure.
    """
    if not candidates:
        return []

    if not config.rerank_enabled and think_fn is None:
        return candidates[:top_k]

    # Agent thinking reranking
    if think_fn is not None:
        try:
            from .llm import rerank_with_llm
            return rerank_with_llm(query, candidates, think_fn, top_k)
        except Exception as exc:
            logger.debug("Rerank via callback failed: %s, falling back", exc)

    return candidates[:top_k]


