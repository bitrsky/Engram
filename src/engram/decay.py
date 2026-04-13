"""
decay.py — Decay and promotion engine for Engram.

Adjusts memory importance based on access patterns and age.
Run periodically (daily or on wake-up).

Decay rules:
- 30 days without access: importance *= 0.95
- 90 days without access: importance *= 0.90
- 180 days without access: importance *= 0.85

Promotion rules:
- access_count >= 3: importance += 1.0 (one-time boost)
- access_count >= 10: importance += 0.5 (additional boost)

Importance is clamped to [0.1, 5.0].
Changes are written back to both Markdown frontmatter and SQLite index.
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from .config import EngramConfig
from .store import update_frontmatter

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

IMPORTANCE_MIN = 0.1
IMPORTANCE_MAX = 5.0

DECAY_THRESHOLDS: List[Tuple[int, float]] = [
    # (days_without_access, multiplier) — evaluated from longest to shortest
    (180, 0.85),
    (90, 0.90),
    (30, 0.95),
]

PROMOTION_RULES: List[Tuple[int, float]] = [
    # (min_access_count, boost) — order matters: applied sequentially
    (3, 1.0),   # first boost at 3 accesses
    (10, 0.5),  # additional boost at 10 accesses
]


@dataclass
class DecayResult:
    """Result of a decay run."""

    total_scanned: int = 0
    decayed: int = 0       # Number of memories whose importance decreased
    promoted: int = 0      # Number of memories whose importance increased
    unchanged: int = 0
    errors: int = 0


def _clamp(value: float, lo: float = IMPORTANCE_MIN, hi: float = IMPORTANCE_MAX) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def _parse_iso(dt_str: str, fallback: Optional[datetime] = None) -> datetime:
    """
    Parse an ISO datetime string, returning *fallback* on failure.

    Handles both ``YYYY-MM-DD`` and ``YYYY-MM-DDTHH:MM:SS`` forms.
    """
    if not dt_str:
        return fallback or datetime.min
    try:
        return datetime.fromisoformat(str(dt_str))
    except (ValueError, TypeError):
        return fallback or datetime.min


# ── Core calculation ────────────────────────────────────────────────────────


def _calculate_new_importance(
    current_importance: float,
    access_count: int,
    last_accessed: str,
    created: str,
    now: Optional[datetime] = None,
    already_promoted_3: bool = False,
    already_promoted_10: bool = False,
) -> Tuple[float, str, bool, bool]:
    """
    Calculate new importance for a single memory.

    Args:
        current_importance: Current importance value.
        access_count: Number of times accessed.
        last_accessed: ISO datetime of last access (may be empty/None).
        created: ISO datetime of creation.
        now: Current time (injectable for testing).
        already_promoted_3: True if the 3-access boost was already applied.
        already_promoted_10: True if the 10-access boost was already applied.

    Returns:
        (new_importance, reason, promoted_3, promoted_10)

        reason is one of:
            "decayed_30d" | "decayed_90d" | "decayed_180d" |
            "promoted_3" | "promoted_10" | "promoted_3_10" |
            "unchanged"

        promoted_3 / promoted_10: updated flags (True if boost was applied
        this run OR in a previous run).
    """
    if now is None:
        now = datetime.now()

    importance = current_importance
    promoted_3 = already_promoted_3
    promoted_10 = already_promoted_10

    # ── Determine reference time for "days without access" ──────────────
    # If never accessed, fall back to the creation date.
    if last_accessed:
        reference_dt = _parse_iso(last_accessed, fallback=_parse_iso(created))
    else:
        reference_dt = _parse_iso(created)

    days_idle = (now - reference_dt).total_seconds() / 86400.0

    # ── Decay ───────────────────────────────────────────────────────────
    # Apply the FIRST (longest) matching threshold only — they are mutually
    # exclusive tiers, evaluated from most severe to least.
    decay_reason = None
    for threshold_days, multiplier in DECAY_THRESHOLDS:
        if days_idle >= threshold_days:
            importance *= multiplier
            decay_reason = f"decayed_{threshold_days}d"
            break  # only the most severe tier applies

    # ── Promotion (one-time boosts) ─────────────────────────────────────
    promotion_reasons: List[str] = []

    if access_count >= 3 and not promoted_3:
        importance += 1.0
        promoted_3 = True
        promotion_reasons.append("promoted_3")

    if access_count >= 10 and not promoted_10:
        importance += 0.5
        promoted_10 = True
        promotion_reasons.append("promoted_10")

    # ── Clamp ───────────────────────────────────────────────────────────
    importance = _clamp(importance)

    # ── Build composite reason ──────────────────────────────────────────
    if promotion_reasons and decay_reason:
        # Both decay and promotion happened — promotion wins for the label
        # since the net effect is a boost (promotion deltas are larger).
        reason = "_".join(promotion_reasons)
    elif promotion_reasons:
        reason = "_".join(promotion_reasons)
    elif decay_reason:
        reason = decay_reason
    else:
        reason = "unchanged"

    # If after all adjustments + clamping the value hasn't actually changed,
    # normalise the reason to "unchanged".
    if importance == current_importance:
        reason = "unchanged"

    return importance, reason, promoted_3, promoted_10


# ── Public API ──────────────────────────────────────────────────────────────


def run_decay(
    index_dir: Optional[str | Path] = None,
    memories_dir: Optional[str | Path] = None,
    config: Optional[EngramConfig] = None,
    dry_run: bool = False,
) -> DecayResult:
    """
    Run the decay engine.

    Scans SQLite index for all memories, calculates new importance
    based on access patterns and age, then updates both:
    1. Markdown frontmatter (source of truth)
    2. SQLite index (derived)

    Args:
        index_dir: Override index directory.
        memories_dir: Override memories directory.
        config: Configuration (used for defaults when dirs not specified).
        dry_run: If True, calculate changes but don't write them.

    Returns:
        DecayResult with statistics.
    """
    if config is None:
        config = EngramConfig()

    if index_dir is None:
        index_dir = config.index_dir
    index_dir = Path(index_dir)

    if memories_dir is None:
        memories_dir = config.memories_dir
    memories_dir = Path(memories_dir)

    result = DecayResult()

    # ── Open SQLite directly (read-only scan, then targeted writes) ─────
    db_path = index_dir / "meta.sqlite3"
    if not db_path.exists():
        logger.warning("SQLite index not found at %s — nothing to decay.", db_path)
        return result

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            "SELECT id, importance, access_count, last_accessed, created, file_path "
            "FROM memory_index"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.error("Failed to query memory_index: %s", exc)
        conn.close()
        return result

    now = datetime.now()

    for row in rows:
        result.total_scanned += 1
        memory_id: str = row["id"]
        current_importance: float = float(row["importance"] or 3.0)
        access_count: int = int(row["access_count"] or 0)
        last_accessed: str = row["last_accessed"] or ""
        created: str = row["created"] or ""
        file_path_str: str = row["file_path"] or ""
        file_path = Path(file_path_str)

        # ── Read existing promotion flags from Markdown frontmatter ────
        already_promoted_3 = False
        already_promoted_10 = False

        if file_path.is_file():
            try:
                from .store import parse_frontmatter

                meta, _ = parse_frontmatter(file_path)
                decay_promoted = meta.get("decay_promoted", [])
                if isinstance(decay_promoted, list):
                    already_promoted_3 = "access_3" in decay_promoted
                    already_promoted_10 = "access_10" in decay_promoted
                elif isinstance(decay_promoted, str):
                    already_promoted_3 = "access_3" in decay_promoted
                    already_promoted_10 = "access_10" in decay_promoted
            except Exception as exc:
                logger.debug(
                    "Could not read frontmatter for %s: %s", memory_id, exc
                )

        # ── Calculate ──────────────────────────────────────────────────
        new_importance, reason, promoted_3, promoted_10 = _calculate_new_importance(
            current_importance=current_importance,
            access_count=access_count,
            last_accessed=last_accessed,
            created=created,
            now=now,
            already_promoted_3=already_promoted_3,
            already_promoted_10=already_promoted_10,
        )

        if reason == "unchanged":
            result.unchanged += 1
            continue

        # ── Classify as decay or promotion for stats ───────────────────
        if new_importance < current_importance:
            result.decayed += 1
        else:
            result.promoted += 1

        logger.info(
            "%-30s  %.2f → %.2f  (%s)",
            memory_id,
            current_importance,
            new_importance,
            reason,
        )

        if dry_run:
            continue

        # ── Write back to Markdown frontmatter ─────────────────────────
        if file_path.is_file():
            try:
                # Build the decay_promoted list for persistence.
                promoted_flags: List[str] = []
                if promoted_3:
                    promoted_flags.append("access_3")
                if promoted_10:
                    promoted_flags.append("access_10")

                fm_updates: dict = {"importance": round(new_importance, 4)}
                if promoted_flags:
                    fm_updates["decay_promoted"] = promoted_flags

                update_frontmatter(file_path, fm_updates)
            except Exception as exc:
                logger.error(
                    "Failed to update frontmatter for %s: %s", memory_id, exc
                )
                result.errors += 1
                # Still try to update SQLite below so it stays consistent
                # with whatever the frontmatter was before the error.
                continue
        else:
            logger.warning(
                "Markdown file missing for %s: %s", memory_id, file_path_str
            )
            # Update SQLite anyway so the index at least has the right value.

        # ── Write back to SQLite index ─────────────────────────────────
        try:
            conn.execute(
                "UPDATE memory_index SET importance = ? WHERE id = ?",
                (round(new_importance, 4), memory_id),
            )
        except sqlite3.Error as exc:
            logger.error(
                "Failed to update SQLite for %s: %s", memory_id, exc
            )
            result.errors += 1

    # Commit all SQLite changes in one batch.
    if not dry_run:
        try:
            conn.commit()
        except sqlite3.Error as exc:
            logger.error("Failed to commit SQLite changes: %s", exc)

    conn.close()
    return result
