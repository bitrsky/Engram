"""
learn.py — Adaptive pattern learning for Engram.

Observes LLM fact-extraction results and discovers new keywords/phrases
that should be added to quality gate and conflict detection patterns.

**Zero extra LLM calls** — works entirely by comparing the structured
output that ``extract_facts_via_callback`` already produced with what
the heuristic patterns matched (or missed).

Learned patterns go through a two-stage promotion pipeline:

    1. candidate  — first seen, stored in ``[_candidates]``, hits = 1
    2. active     — hits >= PROMOTION_THRESHOLD → promoted to real pattern

Storage: ``~/.engram/learned_patterns.toml``
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from .config import EngramConfig
    from .extract import FactCandidate

logger = logging.getLogger(__name__)

# ── Promotion threshold ─────────────────────────────────────────────────────
# A candidate keyword must be seen this many times before it becomes an
# active pattern.  Prevents one-off anomalies from polluting patterns.
PROMOTION_THRESHOLD = 3

# ── Predicate → pattern category mapping ────────────────────────────────────
# Maps LLM-extracted predicate names to which quality-gate category they
# correspond to.  Only these predicates trigger learning.
_PREDICATE_TO_CATEGORY: Dict[str, str] = {
    # decision markers
    "decision": "decision_markers",
    "decided": "decision_markers",
    "chose": "decision_markers",
    "choice": "decision_markers",
    "selected": "decision_markers",
    "picked": "decision_markers",
    # milestone markers
    "deployed": "milestone_markers",
    "shipped": "milestone_markers",
    "launched": "milestone_markers",
    "released": "milestone_markers",
    "completed": "milestone_markers",
    "finished": "milestone_markers",
    "status": "milestone_markers",  # status=deployed/live → milestone
    # problem markers
    "fixed": "problem_markers",
    "resolved": "problem_markers",
    "debugged": "problem_markers",
    "workaround": "problem_markers",
    "root_cause": "problem_markers",
}

# Status values that map to milestones vs problems
_MILESTONE_STATUSES = {"deployed", "live", "running", "complete", "finished", "done", "ready", "stable", "shipped", "launched", "released"}
_PROBLEM_STATUSES = {"broken", "down", "failing", "crashed", "offline", "blocked", "stalled", "stuck"}

# Minimum keyword length (skip single characters, articles, etc.)
_MIN_KEYWORD_LEN = 2

# Words to never learn as patterns (too generic)
_STOP_WORDS: Set[str] = {
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "and",
    "but", "or", "nor", "not", "so", "very", "just", "about", "up",
    "it", "its", "we", "our", "they", "their", "he", "she", "you",
    "i", "me", "my", "this", "that", "these", "those", "here", "there",
    # Chinese particles / too-generic
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "们",
    "和", "与", "或", "但", "也", "都", "就", "把", "被", "让",
    "给", "对", "从", "到", "用", "以", "而", "及", "等", "着",
    "过", "地", "得", "不", "没", "有", "这", "那", "个", "些",
    "吧", "呢", "啊", "哦", "嗯", "吗",
}


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """A candidate keyword awaiting promotion."""
    keyword: str
    category: str  # "decision_markers", "milestone_markers", etc.
    section: str   # "quality" or "conflicts"
    hits: int = 1
    first_seen: str = ""
    last_seen: str = ""

    def to_dict(self) -> dict:
        return {
            "keyword": self.keyword,
            "category": self.category,
            "section": self.section,
            "hits": self.hits,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @staticmethod
    def from_dict(d: dict) -> "Candidate":
        return Candidate(
            keyword=d.get("keyword", ""),
            category=d.get("category", ""),
            section=d.get("section", "quality"),
            hits=d.get("hits", 1),
            first_seen=d.get("first_seen", ""),
            last_seen=d.get("last_seen", ""),
        )


@dataclass
class LearnResult:
    """What the learner discovered in this round."""
    new_candidates: int = 0
    promoted: int = 0
    already_covered: int = 0
    details: List[str] = field(default_factory=list)


# ── Core learning function ──────────────────────────────────────────────────

def learn_from_extraction(
    content: str,
    facts: List["FactCandidate"],
    matched_categories: Set[str],
    config: "EngramConfig",
) -> LearnResult:
    """
    Compare LLM extraction results with heuristic pattern matches.

    This is the main entry point.  Called from ``remember.py`` after
    ``extract_facts()`` and ``quality_gate()`` have both run.

    Args:
        content: The original text that was remembered.
        facts: FactCandidates returned by LLM extraction.
        matched_categories: Set of pattern categories that quality_gate
            matched (e.g. {"decision_markers", "milestone_markers"}).
        config: EngramConfig instance.

    Returns:
        LearnResult summarizing what was learned.
    """
    result = LearnResult()

    if not facts:
        return result

    # Load current learned state
    learned_path = config.base_dir / "learned_patterns.toml"
    state = _load_learned_state(learned_path)

    today = date.today().isoformat()
    changed = False

    for fact in facts:
        predicate = fact.predicate.lower().strip().replace(" ", "_")

        # ── Check 1: Decision/Milestone/Problem mismatch ────────────────
        category = _resolve_category(predicate, fact)
        if category and category not in matched_categories:
            # LLM sees this category, but heuristic pattern missed it
            keywords = _extract_keywords(content, fact, config)
            for kw in keywords:
                if _keyword_already_covered(kw, "quality", category, config, state):
                    result.already_covered += 1
                    continue
                did_change = _upsert_candidate(
                    state, kw, "quality", category, today, result
                )
                if did_change:
                    changed = True

        # ── Check 2: Supersede mismatch ─────────────────────────────────
        if fact.conflicts_with:
            # LLM detected a supersede, check if signals matched
            supersede_matched = _check_supersede_match(content, config)
            if not supersede_matched:
                keywords = _extract_keywords(content, fact, config)
                for kw in keywords:
                    if _keyword_already_covered(kw, "conflicts", "supersede_signals", config, state):
                        result.already_covered += 1
                        continue
                    did_change = _upsert_candidate(
                        state, kw, "conflicts", "supersede_signals", today, result
                    )
                    if did_change:
                        changed = True

    # Promote candidates that crossed the threshold
    promoted = _promote_candidates(state, today)
    result.promoted = len(promoted)
    if promoted:
        changed = True
        for p in promoted:
            result.details.append(
                f"promoted '{p.keyword}' → [{p.section}].{p.category}"
            )

    # Save if anything changed
    if changed:
        _save_learned_state(learned_path, state)

    return result


# ── Category resolution ─────────────────────────────────────────────────────

def _resolve_category(predicate: str, fact: "FactCandidate") -> Optional[str]:
    """
    Map a fact's predicate to a pattern category.

    Special handling for "status" predicate: check the object value
    to determine if it's a milestone or problem status.
    """
    if predicate == "status":
        obj_lower = fact.object.lower().strip()
        if obj_lower in _MILESTONE_STATUSES:
            return "milestone_markers"
        if obj_lower in _PROBLEM_STATUSES:
            return "problem_markers"
        return None

    return _PREDICATE_TO_CATEGORY.get(predicate)


# ── Keyword extraction ──────────────────────────────────────────────────────

def _extract_keywords(
    content: str,
    fact: "FactCandidate",
    config: "EngramConfig",
) -> List[str]:
    """
    Extract candidate keywords from content that might serve as new patterns.

    Strategy: remove known entities (subject, object) from the relevant
    sentence, then extract remaining meaningful n-grams (1-3 words).
    """
    # Find the sentence in content that's most relevant to this fact
    sentence = _find_relevant_sentence(content, fact)
    if not sentence:
        return []

    # Remove the known entities so we isolate the "signal" words
    cleaned = sentence
    for entity in [fact.subject, fact.object]:
        if entity and len(entity) >= 2:
            # Case-insensitive removal
            cleaned = re.sub(re.escape(entity), " ", cleaned, flags=re.IGNORECASE)

    # Tokenize
    tokens = _tokenize(cleaned)

    # Filter stop words and too-short tokens
    tokens = [t for t in tokens if t.lower() not in _STOP_WORDS and len(t) >= _MIN_KEYWORD_LEN]

    if not tokens:
        return []

    # Generate n-grams (1, 2, and 3-grams)
    keywords: List[str] = []

    # 1-grams
    for t in tokens:
        keywords.append(t)

    # 2-grams (for phrases like "决定用", "switched to")
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + tokens[i + 1]  # joined for CJK
        # Also space-joined for western languages
        bigram_spaced = tokens[i] + " " + tokens[i + 1]
        if _is_cjk(tokens[i]) or _is_cjk(tokens[i + 1]):
            keywords.append(bigram)
        else:
            keywords.append(bigram_spaced)

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique.append(kw)

    return unique


def _find_relevant_sentence(content: str, fact: "FactCandidate") -> str:
    """Find the sentence most relevant to a fact (by entity overlap)."""
    # If fact has source_text, use that directly
    if fact.source_text:
        return fact.source_text

    # Otherwise, search content for the sentence mentioning the object
    sentences = re.split(r"[.!?\n。！？\n]+", content)
    best = ""
    best_score = 0

    obj_lower = fact.object.lower()
    subj_lower = fact.subject.lower()

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        s_lower = s.lower()
        score = 0
        if obj_lower in s_lower:
            score += 2
        if subj_lower in s_lower:
            score += 1
        if score > best_score:
            best_score = score
            best = s

    return best


def _tokenize(text: str) -> List[str]:
    """
    Simple tokenizer that handles both CJK and Western text.

    For CJK: splits into individual characters (good enough for
    our 1-2 gram keyword extraction).
    For Western: splits on whitespace and punctuation.
    """
    tokens: List[str] = []
    current_western = ""
    current_cjk = ""

    for char in text:
        if _is_cjk_char(char):
            # Flush western buffer
            if current_western.strip():
                tokens.extend(re.split(r"[\s,;:!?()\"']+", current_western.strip()))
                current_western = ""
            # Each CJK char is a token
            tokens.append(char)
        elif char.isspace() or char in ",.;:!?()\"'":
            # Flush western buffer on separator
            if current_western.strip():
                tokens.extend(re.split(r"[\s,;:!?()\"']+", current_western.strip()))
                current_western = ""
        else:
            current_western += char

    # Flush remaining
    if current_western.strip():
        tokens.extend(re.split(r"[\s,;:!?()\"']+", current_western.strip()))

    return [t for t in tokens if t]


def _is_cjk_char(char: str) -> bool:
    """Check if a character is in a CJK Unicode range."""
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF)       # CJK Unified Ideographs
        or (0x3400 <= cp <= 0x4DBF)    # CJK Extension A
        or (0x20000 <= cp <= 0x2A6DF)  # CJK Extension B
        or (0xF900 <= cp <= 0xFAFF)    # CJK Compatibility Ideographs
        or (0x3000 <= cp <= 0x303F)    # CJK Symbols
        or (0x3040 <= cp <= 0x309F)    # Hiragana
        or (0x30A0 <= cp <= 0x30FF)    # Katakana
        or (0xAC00 <= cp <= 0xD7AF)    # Hangul Syllables
    )


def _is_cjk(text: str) -> bool:
    """Check if text contains any CJK characters."""
    return any(_is_cjk_char(c) for c in text)


# ── Pattern coverage check ──────────────────────────────────────────────────

def _keyword_already_covered(
    keyword: str,
    section: str,
    category: str,
    config: "EngramConfig",
    state: dict,
) -> bool:
    """
    Check if a keyword is already covered by existing patterns (builtin,
    user, or previously learned).
    """
    # Build a test string containing just the keyword
    test = keyword.lower()

    # Get all current patterns for this category
    patterns = _get_all_patterns(section, category, config)

    # Also include already-promoted learned patterns
    active = state.get("active", {})
    section_active = active.get(section, {})
    learned_keywords = section_active.get(category, [])

    for pattern in patterns:
        try:
            if re.search(pattern, test, re.IGNORECASE):
                return True
        except re.error:
            continue

    # Check against learned keywords (plain substring match)
    for lk in learned_keywords:
        if isinstance(lk, str) and lk.lower() in test:
            return True

    return False


def _get_all_patterns(section: str, category: str, config: "EngramConfig") -> List[str]:
    """Get all current patterns for a section.category from config."""
    if section == "quality":
        mapping = {
            "decision_markers": config.quality_decision_markers,
            "milestone_markers": config.quality_milestone_markers,
            "problem_markers": config.quality_problem_markers,
            "noise_patterns": config.quality_noise_patterns,
        }
    elif section == "conflicts":
        mapping = {
            "supersede_signals": config.conflict_supersede_signals,
        }
    else:
        return []
    return mapping.get(category, [])


def _check_supersede_match(content: str, config: "EngramConfig") -> bool:
    """Check if any supersede signal pattern matches the content."""
    lower = content.lower()
    for pattern in config.conflict_supersede_signals:
        try:
            if re.search(pattern, lower, re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


# ── Candidate management ────────────────────────────────────────────────────

def _upsert_candidate(
    state: dict,
    keyword: str,
    section: str,
    category: str,
    today: str,
    result: LearnResult,
) -> bool:
    """
    Insert or update a candidate keyword.  Returns True if state changed.
    """
    candidates = state.setdefault("candidates", [])

    # Check if this keyword already exists as a candidate
    for cand in candidates:
        if (cand.get("keyword", "").lower() == keyword.lower()
                and cand.get("section") == section
                and cand.get("category") == category):
            cand["hits"] = cand.get("hits", 1) + 1
            cand["last_seen"] = today
            return True

    # New candidate
    candidates.append({
        "keyword": keyword,
        "category": category,
        "section": section,
        "hits": 1,
        "first_seen": today,
        "last_seen": today,
    })
    result.new_candidates += 1
    result.details.append(f"new candidate: '{keyword}' for [{section}].{category}")
    return True


def _promote_candidates(state: dict, today: str) -> List[Candidate]:
    """
    Promote candidates with hits >= PROMOTION_THRESHOLD to active patterns.
    Returns the list of promoted candidates.
    """
    candidates = state.get("candidates", [])
    active = state.setdefault("active", {})
    promoted: List[Candidate] = []

    remaining: List[dict] = []
    for cand_dict in candidates:
        if cand_dict.get("hits", 0) >= PROMOTION_THRESHOLD:
            cand = Candidate.from_dict(cand_dict)
            # Add to active patterns
            section_active = active.setdefault(cand.section, {})
            cat_list = section_active.setdefault(cand.category, [])
            if cand.keyword not in cat_list:
                cat_list.append(cand.keyword)
                promoted.append(cand)
        else:
            remaining.append(cand_dict)

    state["candidates"] = remaining
    return promoted


# ── Persistence ─────────────────────────────────────────────────────────────

def _load_learned_state(path: Path) -> dict:
    """
    Load learned patterns state from TOML file.

    The file has two logical sections:
    - active patterns (promoted, used by config)
    - candidates (not yet promoted, tracked here)

    We store as JSON-in-TOML for the candidates (complex structure),
    and plain TOML arrays for active patterns.
    """
    if not path.exists():
        return {"active": {}, "candidates": []}

    try:
        from .config import _load_toml
        data = _load_toml(path)
    except Exception:
        return {"active": {}, "candidates": []}

    # Extract active patterns (everything except _candidates)
    active: Dict[str, dict] = {}
    candidates: List[dict] = []

    for key, value in data.items():
        if key == "_candidates":
            # Candidates stored as JSON string inside data key
            if isinstance(value, dict):
                raw = value.get("data", "")
                if isinstance(raw, str):
                    try:
                        candidates = json.loads(raw)
                    except json.JSONDecodeError:
                        candidates = []
            elif isinstance(value, str):
                try:
                    candidates = json.loads(value)
                except json.JSONDecodeError:
                    candidates = []
            elif isinstance(value, list):
                candidates = value
        elif key == "replace":
            continue
        elif isinstance(value, dict):
            active[key] = dict(value)

    return {"active": active, "candidates": candidates}


def _save_learned_state(path: Path, state: dict) -> None:
    """Save learned patterns state to TOML file."""
    lines: List[str] = [
        "# Auto-generated by Engram pattern learner.",
        "# These patterns were discovered from your usage.",
        "# You can freely edit or delete entries.",
        "",
    ]

    active = state.get("active", {})
    candidates = state.get("candidates", [])

    # Write active patterns as plain TOML
    for section, categories in sorted(active.items()):
        if not categories:
            continue
        lines.append(f"[{section}]")
        for category, keywords in sorted(categories.items()):
            if keywords:
                kw_strs = ", ".join(f'"{_escape_toml(kw)}"' for kw in keywords)
                lines.append(f"{category} = [{kw_strs}]")
        lines.append("")

    # Write candidates as JSON in a TOML string (complex nested structure)
    if candidates:
        lines.append("[_candidates]")
        # Store as a JSON string for easy round-tripping
        json_str = json.dumps(candidates, ensure_ascii=False, indent=2)
        # TOML multi-line basic string
        lines.append(f'data = """{json_str}"""')
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Best-effort permission setting
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _escape_toml(s: str) -> str:
    """Escape a string for TOML basic string value."""
    return s.replace("\\", "\\\\").replace('"', '\\"')
