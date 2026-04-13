"""
conflicts.py — Conflict classification and resolution for Engram.

Handles 4 types of fact conflicts:
1. temporal_succession — newer fact supersedes older (auto-resolve)
2. implicit_supersede — language implies replacement (auto-resolve)
3. opinion_change — preference/opinion changed over time (auto-resolve, both kept)
4. hard_contradiction — cannot determine which is correct (defer to user)
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict


# Default exclusive predicates: same subject can only have ONE object at a time
# scope="subject" means (subject, predicate, ?) should only have one active value
# scope="none" means multiple values are fine (e.g., someone can love many things)
DEFAULT_EXCLUSIVE_PREDICATES = {
    # Exclusive (only one at a time)
    "uses_database": "subject",
    "uses_auth": "subject",
    "uses_framework": "subject",
    "uses_language": "subject",
    "deployed_on": "subject",
    "assigned_to": "subject",
    "status": "subject",
    "led_by": "subject",
    "managed_by": "subject",
    "database": "subject",
    "auth": "subject",
    "framework": "subject",
    "hosting": "subject",
    "role": "subject",
    # Non-exclusive (multiple allowed)
    "loves": "none",
    "knows": "none",
    "works_with": "none",
    "interested_in": "none",
    "uses_tool": "none",
    "depends_on": "none",
    "collaborates_with": "none",
}

# Patterns that indicate one thing is replacing another
# NOTE: Built-in patterns have moved to config._BUILTIN_PATTERNS.
# This module-level list is kept ONLY for backwards compatibility when
# detect_conflicts / check_conflict is called without a config object.
_FALLBACK_SUPERSEDE_SIGNALS = [
    r"\bswitch(ed|ing)?\s+(to|from)\b",
    r"\bmigrat(e|ed|ing)\s+(to|from)\b",
    r"\breplac(e|ed|ing)\b",
    r"\bno longer\s+us(e|ing)\b",
    r"\bmoved?\s+(to|away from)\b",
    r"\bdrop(ped|ping)?\b",
    r"\bstopp(ed|ing)\s+using\b",
    r"\babandoned?\b",
    r"\bdeprecated?\b",
    r"\bphased?\s+out\b",
    r"\bchanged?\s+(to|from)\b",
    r"\bupgrad(e|ed|ing)\s+(to|from)\b",
    r"\bconverted?\s+to\b",
]

# Predicates that represent opinions/preferences
OPINION_PREDICATES = {
    "likes", "dislikes", "prefers", "preference",
    "thinks_about", "feels_about", "opinion_on",
    "favorite", "hates",
}


@dataclass
class Fact:
    """Represents a single fact (triple)."""
    subject: str
    predicate: str
    object: str
    since: str = ""
    confidence: float = 1.0
    source: str = ""
    expired_at: str = ""
    reason: str = ""


@dataclass
class Conflict:
    """Represents a detected conflict between two facts."""
    old_fact: Fact
    new_fact: Fact
    conflict_type: str  # temporal_succession | implicit_supersede | opinion_change | hard_contradiction
    source_text: str = ""  # The text that triggered the new fact
    resolution: Optional[str] = None  # "old_wins" | "new_wins" | "both_valid" | None (unresolved)
    resolution_method: Optional[str] = None  # "auto_temporal" | "auto_supersede" | "auto_opinion" | "user"
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())


def _normalize_predicate(predicate: str) -> str:
    """Normalize a predicate for comparison (lowercase, underscores)."""
    return predicate.lower().replace(" ", "_").strip()


def _is_exclusive(predicate: str, exclusive_predicates: Dict[str, str]) -> bool:
    """Check whether a predicate is exclusive (only one object per subject)."""
    norm = _normalize_predicate(predicate)
    scope = exclusive_predicates.get(norm)
    # If the predicate is known and has scope "subject", it's exclusive.
    # If the predicate is unknown (not in the dict), default to exclusive —
    # it's safer to flag a potential conflict than to silently ignore it.
    if scope is None:
        return True  # unknown predicate → treat as exclusive (safe default)
    return scope == "subject"


def _is_active(fact: Fact) -> bool:
    """Check if a fact is currently active (not expired)."""
    return not fact.expired_at


def _has_supersede_signal(text: str, signals: Optional[List[str]] = None) -> bool:
    """Check if text contains language implying replacement/switching."""
    if not text:
        return False
    patterns = signals if signals is not None else _FALLBACK_SUPERSEDE_SIGNALS
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _is_opinion_predicate(predicate: str) -> bool:
    """Check if a predicate represents an opinion or preference."""
    return _normalize_predicate(predicate) in OPINION_PREDICATES


def _compare_dates(date_a: str, date_b: str) -> int:
    """
    Compare two ISO date strings lexicographically.

    Returns:
        -1 if a < b (a is earlier)
         0 if a == b
         1 if a > b (a is later)

    Empty strings are treated as "unknown" (earliest possible).
    """
    # Treat empty/None as the earliest possible date
    a = date_a or ""
    b = date_b or ""

    if a == b:
        return 0
    if not a:
        return -1  # unknown is "earlier" than any known date
    if not b:
        return 1  # any known date is "later" than unknown
    # ISO format sorts lexicographically
    if a < b:
        return -1
    return 1


def check_conflict(
    new_fact: Fact,
    existing_facts: List[Fact],
    exclusive_predicates: Optional[Dict[str, str]] = None,
    supersede_signals: Optional[List[str]] = None,
) -> Optional[Conflict]:
    """
    Check if a new fact conflicts with any existing facts.

    Only checks ACTIVE facts (not expired). For exclusive predicates,
    a conflict exists when the same subject+predicate already has a
    different object. For non-exclusive predicates, no conflict is raised
    (e.g., someone can love chess AND swimming).

    Args:
        new_fact: The fact being added
        existing_facts: List of currently active facts (not expired)
        exclusive_predicates: Override predicate exclusivity rules

    Returns:
        Conflict if found, None if no conflict
    """
    preds = exclusive_predicates if exclusive_predicates is not None else DEFAULT_EXCLUSIVE_PREDICATES

    new_pred = _normalize_predicate(new_fact.predicate)

    # Non-exclusive predicates never conflict
    if not _is_exclusive(new_pred, preds):
        return None

    new_subj = new_fact.subject.lower().strip()
    new_obj = new_fact.object.lower().strip()

    for existing in existing_facts:
        # Only check active facts
        if not _is_active(existing):
            continue

        existing_pred = _normalize_predicate(existing.predicate)
        existing_subj = existing.subject.lower().strip()
        existing_obj = existing.object.lower().strip()

        # Same subject + same predicate?
        if existing_subj == new_subj and existing_pred == new_pred:
            # Same object → not a conflict (duplicate, not our problem)
            if existing_obj == new_obj:
                continue

            # Different object on an exclusive predicate → CONFLICT
            conflict_type = classify_conflict(
                old_fact=existing,
                new_fact=new_fact,
                source_text=new_fact.source,
                supersede_signals=supersede_signals,
            )
            return Conflict(
                old_fact=existing,
                new_fact=new_fact,
                conflict_type=conflict_type,
                source_text=new_fact.source,
            )

    return None


def classify_conflict(
    old_fact: Fact,
    new_fact: Fact,
    source_text: str = "",
    supersede_signals: Optional[List[str]] = None,
) -> str:
    """
    Classify a conflict into one of 4 types.

    Logic:
    1. If both have temporal info AND new is later → "temporal_succession"
    2. If source_text matches supersede signals → "implicit_supersede"
    3. If predicate is in OPINION_PREDICATES → "opinion_change"
    4. Otherwise → "hard_contradiction"

    Returns: conflict type string
    """
    # 1. Temporal succession: both have dates and the new one is later
    if old_fact.since and new_fact.since:
        if _compare_dates(new_fact.since, old_fact.since) > 0:
            return "temporal_succession"

    # 2. Implicit supersede: the source text uses replacement language
    if _has_supersede_signal(source_text, supersede_signals):
        return "implicit_supersede"

    # 3. Opinion change: the predicate is an opinion/preference
    if _is_opinion_predicate(old_fact.predicate) or _is_opinion_predicate(new_fact.predicate):
        return "opinion_change"

    # 4. Hard contradiction: can't determine which is correct
    return "hard_contradiction"


def resolve_conflict(conflict: Conflict) -> dict:
    """
    Attempt to automatically resolve a conflict.

    Returns:
        {
            "resolved": bool,
            "action": "expire_old" | "expire_old_lower_confidence" | "expire_old_keep_both" | "defer",
            "old_fact_updates": {...},  # changes to apply to old fact
            "new_fact_updates": {...},  # changes to apply to new fact
            "resolution_reason": str,
        }

    Resolution strategies:
    - temporal_succession: expire old fact, new fact takes over (confidence 1.0)
    - implicit_supersede: expire old fact, new fact at confidence 0.85
    - opinion_change: expire old fact, new fact takes over, note "opinion changed"
    - hard_contradiction: don't resolve, lower both to confidence 0.5, defer to user
    """
    now = datetime.now().isoformat()
    ctype = conflict.conflict_type

    if ctype == "temporal_succession":
        # Clear-cut: newer date wins. Expire the old fact.
        conflict.resolution = "new_wins"
        conflict.resolution_method = "auto_temporal"
        return {
            "resolved": True,
            "action": "expire_old",
            "old_fact_updates": {
                "expired_at": now,
                "reason": (
                    f"Superseded by newer fact: {conflict.new_fact.subject} "
                    f"{conflict.new_fact.predicate} {conflict.new_fact.object} "
                    f"(since {conflict.new_fact.since})"
                ),
            },
            "new_fact_updates": {
                "confidence": 1.0,
            },
            "resolution_reason": (
                f"Temporal succession: {conflict.old_fact.object} "
                f"(since {conflict.old_fact.since}) → "
                f"{conflict.new_fact.object} "
                f"(since {conflict.new_fact.since})"
            ),
        }

    elif ctype == "implicit_supersede":
        # Language implies replacement, but no explicit date proof → slightly
        # lower confidence on the new fact.
        conflict.resolution = "new_wins"
        conflict.resolution_method = "auto_supersede"
        return {
            "resolved": True,
            "action": "expire_old_lower_confidence",
            "old_fact_updates": {
                "expired_at": now,
                "reason": (
                    f"Implicitly superseded (detected replacement language): "
                    f"{conflict.new_fact.subject} {conflict.new_fact.predicate} "
                    f"→ {conflict.new_fact.object}"
                ),
            },
            "new_fact_updates": {
                "confidence": 0.85,
            },
            "resolution_reason": (
                f"Implicit supersede detected in source text: "
                f"{conflict.old_fact.object} → {conflict.new_fact.object}"
            ),
        }

    elif ctype == "opinion_change":
        # Both are valid data points, but the old opinion is no longer current.
        # Expire the old one, keep both in the record.
        conflict.resolution = "both_valid"
        conflict.resolution_method = "auto_opinion"
        return {
            "resolved": True,
            "action": "expire_old_keep_both",
            "old_fact_updates": {
                "expired_at": now,
                "reason": (
                    f"Opinion changed: {conflict.old_fact.subject} "
                    f"{conflict.old_fact.predicate} {conflict.old_fact.object} "
                    f"→ {conflict.new_fact.object}"
                ),
            },
            "new_fact_updates": {
                "confidence": 1.0,
                "reason": "opinion changed",
            },
            "resolution_reason": (
                f"Opinion/preference changed: "
                f"{conflict.old_fact.object} → {conflict.new_fact.object}"
            ),
        }

    else:
        # hard_contradiction: we can't safely resolve this automatically.
        # Lower confidence on both and defer to user.
        conflict.resolution = None  # unresolved
        conflict.resolution_method = "user"
        return {
            "resolved": False,
            "action": "defer",
            "old_fact_updates": {
                "confidence": 0.5,
            },
            "new_fact_updates": {
                "confidence": 0.5,
            },
            "resolution_reason": (
                f"Hard contradiction: {conflict.old_fact.subject} "
                f"{conflict.old_fact.predicate} is "
                f"'{conflict.old_fact.object}' vs '{conflict.new_fact.object}' — "
                f"cannot auto-resolve, deferring to user"
            ),
        }


def format_conflict_report(conflicts: List[Conflict]) -> str:
    """
    Format unresolved conflicts as a human-readable string for L1 Working Set.

    Only includes unresolved conflicts (resolution is None).
    Returns empty string if there are no unresolved conflicts.

    Example output:
    ⚠️ UNRESOLVED CONFLICTS:
      - saas-app → database: 'Postgres' vs 'MongoDB' (detected 2026-03-15)
      - Maya → preference: 'dislikes TypeScript' vs 'loves TypeScript' (detected 2026-02-01)
      → Ask user to resolve when relevant.
    """
    unresolved = [c for c in conflicts if c.resolution is None]

    if not unresolved:
        return ""

    lines = ["⚠️ UNRESOLVED CONFLICTS:"]
    for conflict in unresolved:
        # Extract just the date portion from the ISO timestamp
        detected_date = conflict.detected_at[:10] if conflict.detected_at else "unknown"

        lines.append(
            f"  - {conflict.old_fact.subject} → {conflict.old_fact.predicate}: "
            f"'{conflict.old_fact.object}' vs '{conflict.new_fact.object}' "
            f"(detected {detected_date})"
        )

    lines.append("  → Ask user to resolve when relevant.")
    return "\n".join(lines)
