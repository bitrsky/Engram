"""
remember.py — Write pipeline for Engram.

The complete flow that ties everything together:

1. Quality gate   → reject noise (heuristic, no LLM)
2. Dedup check    → skip exact duplicates and near-duplicates
3. Write Markdown → create the memory file (source of truth)
4. Update index   → make it searchable (ChromaDB + SQLite)
5. Extract facts  → heuristic or LLM-powered
6. Add facts      → with conflict detection and auto-resolution
7. Update project → refresh last_active timestamp

Steps 4–7 are non-critical: if any of them fail, the Markdown file
is already saved and will be picked up on the next index rebuild or
fact-extraction pass.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import EngramConfig
from .dedup import DedupResult, check_duplicate
from .extract import FactCandidate, extract_facts
from .facts import add_fact, get_active_facts
from .index import IndexManager
from .projects import update_project
from .quality import quality_gate
from .store import parse_frontmatter, write_memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class RememberResult:
    """Outcome of the remember pipeline."""

    success: bool
    id: str = ""
    file_path: str = ""
    rejected_reason: str = ""  # "low_quality" | "duplicate:exact" | "duplicate:near" | ""
    facts_extracted: int = 0
    facts_added: int = 0
    conflicts_detected: int = 0
    conflict_details: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Single-item pipeline
# ---------------------------------------------------------------------------


def remember(
    content: str,
    project: Optional[str] = None,
    topics: Optional[List[str]] = None,
    memory_type: str = "note",
    source: str = "manual",
    config: Optional[EngramConfig] = None,
    index_manager: Optional[IndexManager] = None,
    skip_quality_check: bool = False,
    skip_dedup: bool = False,
    skip_facts: bool = False,
    think_fn=None,
) -> RememberResult:
    """
    Run the complete remember pipeline.

    Flow::

        content
          │
          ▼
        ┌──────────────┐  rejected
        │ 1. Quality   ├──────────► RememberResult(success=False)
        │    gate       │
        └──────┬───────┘
               │ pass
               ▼
        ┌──────────────┐  duplicate
        │ 2. Dedup     ├──────────► RememberResult(success=False)
        │    check      │
        └──────┬───────┘
               │ unique
               ▼
        ┌──────────────┐
        │ 3. Write .md │  ← source of truth, always succeeds or raises
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ 4. Index     │  ← non-critical (file exists, next rebuild catches it)
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ 5. Extract   │  ← non-critical
        │    facts      │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ 6. Add facts │  ← non-critical (includes conflict detection)
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ 7. Update    │  ← non-critical
        │    project    │
        └──────────────┘

    Args:
        content:            Text to remember.
        project:            Project tag (``None`` for cross-project memories).
        topics:             Topic tags.
        memory_type:        ``"note"`` | ``"decision"`` | ``"milestone"`` |
                            ``"problem"`` | ``"preference"`` | ``"emotional"``.
        source:             ``"manual"`` | ``"claude-code"`` | ``"chatgpt"`` |
                            ``"ingest"`` | ``"batch"``.
        config:             Override configuration (default: fresh ``EngramConfig``).
        index_manager:      Shared ``IndexManager`` (if ``None``, one is created
                            and closed automatically).
        skip_quality_check: Bypass the quality gate.
        skip_dedup:         Bypass duplicate detection.
        skip_facts:         Bypass fact extraction and addition.

    Returns:
        A ``RememberResult`` describing what happened.
    """
    config = config or EngramConfig()

    # ── Step 1: Quality gate ─────────────────────────────────────────────
    if not skip_quality_check:
        should_store, importance = quality_gate(content, source, config=config)
        if not should_store:
            return RememberResult(success=False, rejected_reason="low_quality")
    else:
        importance = 3.0

    # ── Acquire IndexManager (may be caller-supplied or locally owned) ───
    own_index = index_manager is None
    if own_index:
        index_manager = IndexManager(
            index_dir=config.index_dir,
            memories_dir=config.memories_dir,
        )

    try:
        # ── Step 2: Dedup check ──────────────────────────────────────────
        if not skip_dedup:
            dedup = check_duplicate(content, index_manager, project)
            if dedup.is_duplicate:
                return RememberResult(
                    success=False,
                    rejected_reason=f"duplicate:{dedup.reason}",
                    id=dedup.existing_id,
                )

        # ── Step 3: Write Markdown file ──────────────────────────────────
        filepath = write_memory(
            content=content,
            project=project,
            topics=topics,
            memory_type=memory_type,
            source=source,
            importance=importance,
            memories_dir=config.memories_dir,
        )

        # ── Step 4: Update vector index ──────────────────────────────────
        memory_id = ""
        try:
            memory_id = index_manager.index_memory(filepath)
        except Exception:
            logger.debug("Index update failed for %s; will be caught on rebuild", filepath)

        # Fallback: read ID from the file we just wrote.
        if not memory_id:
            try:
                fm, _ = parse_frontmatter(filepath)
                memory_id = fm.get("id", "")
            except Exception:
                memory_id = ""

        result = RememberResult(
            success=True,
            id=memory_id,
            file_path=str(filepath),
        )

        # ── Steps 5 & 6: Extract facts → add facts ───────────��──────────
        if not skip_facts and project:
            _extract_and_add_facts(
                content=content,
                project=project,
                memory_id=memory_id,
                config=config,
                result=result,
                think_fn=think_fn,
            )

        # ── Step 7: Update project last_active ───────────────────────────
        if project:
            try:
                update_project(
                    project,
                    projects_dir=config.projects_dir,
                    last_active=datetime.now().isoformat(),
                )
            except Exception:
                # Project file may not exist — that's fine.
                logger.debug("Could not update project '%s' last_active", project)

        return result

    finally:
        if own_index:
            index_manager.close()


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------


def remember_batch(
    items: List[dict],
    config: Optional[EngramConfig] = None,
    index_manager: Optional[IndexManager] = None,
) -> List[RememberResult]:
    """
    Remember multiple items, sharing a single ``IndexManager``.

    Each *item* is a ``dict`` whose keys mirror :func:`remember` parameters::

        {
            "content": "We decided to use Clerk for auth.",
            "project": "saas-app",
            "topics": ["auth"],
            "memory_type": "decision",
            "source": "batch",
            "skip_quality_check": False,
            "skip_dedup": False,
            "skip_facts": False,
        }

    Only ``"content"`` is required; everything else has sensible defaults.

    Args:
        items:          List of parameter dicts.
        config:         Override configuration.
        index_manager:  Shared ``IndexManager`` (created internally if ``None``).

    Returns:
        A list of ``RememberResult`` — one per input item, in the same order.
    """
    config = config or EngramConfig()
    own_index = index_manager is None
    if own_index:
        index_manager = IndexManager(
            index_dir=config.index_dir,
            memories_dir=config.memories_dir,
        )

    results: List[RememberResult] = []
    try:
        for item in items:
            r = remember(
                content=item.get("content", ""),
                project=item.get("project"),
                topics=item.get("topics"),
                memory_type=item.get("memory_type", "note"),
                source=item.get("source", "batch"),
                config=config,
                index_manager=index_manager,
                skip_quality_check=item.get("skip_quality_check", False),
                skip_dedup=item.get("skip_dedup", False),
                skip_facts=item.get("skip_facts", False),
            )
            results.append(r)
    finally:
        if own_index:
            index_manager.close()

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_and_add_facts(
    content: str,
    project: str,
    memory_id: str,
    config: EngramConfig,
    result: RememberResult,
    think_fn=None,
) -> None:
    """
    Steps 5–6: extract facts from *content* and add them to the project
    facts file.  Mutates *result* in-place (facts_extracted, facts_added,
    conflicts_detected, conflict_details).

    This is wrapped so the caller can treat the entire block as non-critical.
    """
    try:
        existing_facts = get_active_facts(project, facts_dir=config.facts_dir)

        candidates: List[FactCandidate] = extract_facts(
            content=content,
            project=project,
            existing_facts=existing_facts,
            config=config,
            think_fn=think_fn,
        )
        result.facts_extracted = len(candidates)

        for candidate in candidates:
            fact_result = add_fact(
                project=project,
                subject=candidate.subject,
                predicate=candidate.predicate,
                object_val=candidate.object,
                confidence=candidate.confidence,
                source_memory_id=memory_id,
                since=candidate.temporal or datetime.now().strftime("%Y-%m"),
                source_text=candidate.source_text,
                facts_dir=config.facts_dir,
                exclusive_predicates=config.exclusive_predicates,
                supersede_signals=config.conflict_supersede_signals,
            )

            if fact_result.get("added"):
                result.facts_added += 1

            conflict = fact_result.get("conflict")
            if conflict is not None:
                result.conflicts_detected += 1
                result.conflict_details.append(
                    _format_conflict_detail(conflict, fact_result)
                )

    except Exception:
        logger.debug(
            "Fact extraction/addition failed for project '%s'",
            project,
            exc_info=True,
        )


def _format_conflict_detail(conflict, fact_result: dict) -> dict:
    """Build a serialisable conflict-detail dict for ``RememberResult``."""
    old = conflict.old_fact
    new = conflict.new_fact
    resolution = fact_result.get("resolution") or {}

    return {
        "type": conflict.conflict_type,
        "old": f"{old.subject} → {old.predicate} → {old.object}",
        "new": f"{new.subject} → {new.predicate} → {new.object}",
        "resolution": resolution.get("action", "unresolved"),
    }
