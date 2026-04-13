"""
dedup.py — Semantic deduplication for Engram.

Three-level dedup strategy:
  Level 1: Hash match (O(1), exact content match)
  Level 2: Vector similarity ≥ 0.92 (near-exact semantic match)
  Level 3: Merge judgment for 0.82-0.92 range (same project + same type = dup)
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

from .index import IndexManager

# Thresholds
_EXACT_SEMANTIC_THRESHOLD = 0.92
_MERGE_CANDIDATE_THRESHOLD = 0.82


@dataclass
class DedupResult:
    """Result of deduplication check."""
    is_duplicate: bool
    reason: str = ""          # "hash_match" | "exact_semantic" | "merge_candidate" | ""
    existing_id: str = ""     # ID of the existing memory that matches
    similarity: float = 0.0   # Similarity score (for Level 2/3)


def normalize_for_hash(content: str) -> str:
    """
    Normalize content for hash comparison.
    - Lowercase
    - Strip to alphanumeric words only (remove all punctuation/symbols)
    - Collapse whitespace to single spaces
    - Strip leading/trailing whitespace
    """
    text = content.lower()
    # Keep only alphanumeric characters and whitespace
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_duplicate(
    content: str,
    index_manager: IndexManager,
    project: Optional[str] = None,
    memory_type: Optional[str] = None,
) -> DedupResult:
    """
    Three-level deduplication check.

    Level 1: Hash match
      → sha256(normalize(content)) → look up in SQLite content_hash
      → If found: DedupResult(is_duplicate=True, reason="hash_match", existing_id=...)

    Level 2: Vector similarity
      → vector_search(content, n=3) — no project filter (catch cross-project dupes)
      → Highest similarity hit:
        - ≥ 0.92: DedupResult(is_duplicate=True, reason="exact_semantic", ...)
        - 0.82-0.92: Go to Level 3
        - < 0.82: DedupResult(is_duplicate=False)

    Level 3: Merge judgment (only for 0.82-0.92 range)
      → Compare metadata of the candidate:
        - Same project AND same memory_type → it IS a duplicate
          (reason="merge_candidate", keep the newer/longer one)
        - Different project → NOT a duplicate
        - Different memory_type → NOT a duplicate
      → DedupResult accordingly

    Args:
        content: Content to check
        index_manager: The IndexManager instance
        project: Project context (affects Level 3 judgment)
        memory_type: Memory type context (affects Level 3 judgment)

    Returns:
        DedupResult
    """
    # ------------------------------------------------------------------
    # Level 1: Hash match
    # ------------------------------------------------------------------
    normalized = normalize_for_hash(content)
    content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    existing_id = index_manager.get_content_hash(content_hash)

    if existing_id is not None:
        return DedupResult(
            is_duplicate=True,
            reason="hash_match",
            existing_id=existing_id,
            similarity=1.0,
        )

    # ------------------------------------------------------------------
    # Level 2: Vector similarity (unfiltered — catches cross-project dupes)
    # ------------------------------------------------------------------
    hits = index_manager.vector_search(content, n=3)

    # Empty index → nothing to deduplicate against.
    if not hits:
        return DedupResult(is_duplicate=False)

    best = hits[0]

    if best.similarity >= _EXACT_SEMANTIC_THRESHOLD:
        return DedupResult(
            is_duplicate=True,
            reason="exact_semantic",
            existing_id=best.id,
            similarity=best.similarity,
        )

    if best.similarity < _MERGE_CANDIDATE_THRESHOLD:
        return DedupResult(is_duplicate=False, similarity=best.similarity)

    # ------------------------------------------------------------------
    # Level 3: Merge judgment (0.82 ≤ similarity < 0.92)
    # ------------------------------------------------------------------
    # Same project AND same memory_type → duplicate (merge candidate).
    # Either field differs → distinct memory, keep both.
    same_project = (
        project is not None
        and best.project != ""
        and best.project == project
    )
    same_type = (
        memory_type is not None
        and best.memory_type != ""
        and best.memory_type == memory_type
    )

    if same_project and same_type:
        return DedupResult(
            is_duplicate=True,
            reason="merge_candidate",
            existing_id=best.id,
            similarity=best.similarity,
        )

    return DedupResult(is_duplicate=False, similarity=best.similarity)
