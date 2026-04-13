"""
config.py — Configuration management for Engram.

Manages ~/.engram/ directory structure, config.toml, and patterns.toml.
Priority: env vars > config file > built-in defaults.
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ── TOML compat shim ────────────────────────────────────────────────────────
# Python 3.11+ has tomllib in stdlib.  3.10 needs the tomli backport.
if sys.version_info >= (3, 11):
    import tomllib as _tomllib
else:
    try:
        import tomli as _tomllib  # type: ignore[import-not-found,no-redef]
    except ImportError:
        _tomllib = None  # type: ignore[assignment]

# pyyaml is still used by store.py / facts.py / projects.py for Markdown
# frontmatter, but config.py itself no longer depends on it.

DEFAULT_BASE_DIR = os.path.expanduser("~/.engram")

# ── Exclusive predicates (scope = "subject" means one value per subject) ────
_DEFAULT_EXCLUSIVE_PREDICATES: Dict[str, str] = {
    "uses_database": "subject",
    "uses_auth": "subject",
    "uses_framework": "subject",
    "deployed_on": "subject",
    "assigned_to": "subject",
    "status": "subject",
    "led_by": "subject",
    "managed_by": "subject",
}

_DEFAULT_NON_EXCLUSIVE_PREDICATES: Dict[str, str] = {
    "loves": "none",
    "knows": "none",
    "works_with": "none",
    "interested_in": "none",
    "uses_tool": "none",
}

# ── Built-in patterns (English) ─────────────────────────────────────────────
# These are the defaults.  Users can EXTEND or REPLACE them via patterns.toml.

_BUILTIN_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "quality": {
        "decision_markers": [
            r"\b(decided|chose|went with|picked|settled on|switched to)\b",
            r"\b(instead of|rather than|over.*because)\b",
            r"\b(trade-?off|pros and cons)\b",
        ],
        "milestone_markers": [
            r"\b(shipped|launched|deployed|released|it works|got it working)\b",
            r"\b(breakthrough|figured out|nailed it|cracked it)\b",
            r"\b(finally|first time|first ever)\b",
        ],
        "problem_markers": [
            r"\b(the fix was|root cause|the problem was|solved by)\b",
            r"\b(workaround|the answer was|resolved)\b",
        ],
        "noise_patterns": [
            r"^(ok|okay|sure|got it|thanks|thank you|yes|no|right|fine|cool|nice|great|yep|nope|alright)\s*[.!?]?\s*$",
            r"^(sounds good|makes sense|understood|will do|on it|roger|ack|noted)\s*[.!?]?\s*$",
            r"^(here'?s|let me|sure,?\s*i'?ll|i'?ll help|certainly|of course|absolutely)[,!.\s]",
            r"^(i'?d be happy to|i can help|great question|good point)\s",
        ],
    },
    "conflicts": {
        "supersede_signals": [
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
        ],
    },
    "temporal": {
        "markers": [
            r"\b(when did|when was|when were|when is|when are)\b",
            r"\b(how long ago|how many days|how many weeks|how many months|how many years)\b",
            r"\b(days ago|weeks ago|months ago|years ago)\b",
            r"\b(since when|last time|first time)\b",
            r"\b(before|after|during|between .+ and)\b",
            r"\b(timeline|chronolog|sequence of events)\b",
        ],
    },
}

# ── Default config.toml content ─────────────────────────────────────────────
_DEFAULT_CONFIG_TOML = """\
# ─────────────────────────────────────────────────────────────────────────────
# Engram configuration
# ─────────────────────────────────────────────────────────────────────────────
# Priority: environment variables > this file > built-in defaults.

# ── LLM features ────────────────────────────────────────────────────────────
# LLM capability is injected by the host agent via think_fn callback.
# These toggles control which LLM features are enabled.
[llm]
# temporal_reasoning = true      # LLM reasoning for time-related questions

# ── Exclusive predicates ────────────────────────────────────────────────────
# Predicates listed here trigger conflict detection: when a new fact is filed
# for the same subject + predicate, the old fact is superseded.
# Scope "subject" = one value per subject.  "none" = no exclusivity.
#
# [exclusive_predicates]
# uses_database = "subject"
# uses_auth = "subject"
# status = "subject"
# loves = "none"                 # non-exclusive — multiple values allowed

# ── Pattern learning ────────────────────────────────────────────────────
# Controls adaptive pattern learning (see learned_patterns.toml).
# [learning]
# promotion_threshold = 2        # hits needed before a candidate becomes active (default: 2)
"""

# ── Default patterns.toml content ───────────────────────────────────────────
_DEFAULT_PATTERNS_TOML = """\
# ─────────────────────────────────────────────────────────────────────────────
# Engram pattern configuration
# ─────────────────────────────────────────────────────────────────────────────
# User-defined regex patterns EXTEND the built-in English defaults.
# Set replace = true to use ONLY your patterns (discard built-in English ones).
#
# Each section corresponds to a module:
#   [quality]   — quality gate (importance boosting & noise filtering)
#   [conflicts] — conflict detection (technology/tool transition signals)
#   [temporal]  — temporal reasoning trigger (time-related query detection)
#
# Example: adding Chinese patterns
#
# replace = false
#
# [quality]
# decision_markers = [
#     '\\b(决定|选择|改用|换成|采用)\\b',
#     '\\b(而不是|权衡)\\b',
# ]
# milestone_markers = [
#     '\\b(上线了|发布了|部署了|搞定了|终于)\\b',
# ]
# problem_markers = [
#     '\\b(修复方法是|根本原因|问题出在|解决了)\\b',
# ]
# noise_patterns = [
#     '^(好的|收到|明白|了解|嗯|行|可以)\\s*[。！？]?\\s*$',
# ]
#
# [conflicts]
# supersede_signals = [
#     '\\b(迁移到|替换为|不再使用|弃用|升级到)\\b',
# ]
#
# [temporal]
# markers = [
#     '\\b(什么时候|多久以前|几天前|上次|第一次|之前|之后)\\b',
# ]
"""

# ── Default identity.md template ────────────────────────────────────────────
_DEFAULT_IDENTITY_MD = """\
---
name: ""
role: ""
created: "{created}"
---

# Identity

<!-- Describe the owner / persona of this Engram instance. -->
<!-- This file is read by agents to understand who they are working for. -->

## About

(fill in)

## Preferences

- (fill in)

## Notes

- (fill in)
"""


# ── TOML helpers ────────────────────────────────────────────────────────────

def _load_toml(path: Path) -> dict:
    """Load a TOML file, returning {} on any error or if tomllib is unavailable."""
    if _tomllib is None:
        return {}
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return _tomllib.load(f)
    except Exception:
        return {}


def _dump_toml_simple(data: dict) -> str:
    """
    Minimal TOML serializer — handles the flat/shallow dicts we use in config.

    Supports: str, int, float, bool, list[str], and one level of [section].
    NOT a general-purpose TOML writer — just enough for Engram config.
    """
    lines: list = []
    # Top-level scalars first
    for k, v in data.items():
        if isinstance(v, dict):
            continue
        lines.append(_toml_kv(k, v))

    # Then sections
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append("")
            lines.append(f"[{k}]")
            for sk, sv in v.items():
                lines.append(_toml_kv(sk, sv))

    return "\n".join(lines) + "\n"


def _toml_kv(key: str, value) -> str:
    """Format a single TOML key = value pair."""
    if isinstance(value, bool):
        return f"{key} = {'true' if value else 'false'}"
    if isinstance(value, int):
        return f"{key} = {value}"
    if isinstance(value, float):
        return f"{key} = {value}"
    if isinstance(value, str):
        # Use basic string (escape backslashes and quotes)
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'{key} = "{escaped}"'
    if isinstance(value, list):
        items = ", ".join(_toml_value(item) for item in value)
        return f"{key} = [{items}]"
    return f"# {key} = <unsupported type {type(value).__name__}>"


def _toml_value(value) -> str:
    """Format a single TOML value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return repr(value)


# ── Main config class ───────────────────────────────────────────────────────

class EngramConfig:
    """
    Configuration manager for Engram.

    Reads from ``~/.engram/config.toml`` with env var overrides.
    Patterns are loaded from a separate ``~/.engram/patterns.toml``.

    Attributes (all Path properties):
        base_dir       — ~/.engram/
        memories_dir   — ~/.engram/memories/
        projects_dir   — ~/.engram/projects/
        facts_dir      — ~/.engram/facts/
        index_dir      — ~/.engram/.index/
        identity_path  — ~/.engram/identity.md
        config_path    — ~/.engram/config.toml
        patterns_path  — ~/.engram/patterns.toml

    LLM feature toggles:
        rerank_enabled           -- enable LLM reranking (default: True)
        rerank_candidates        -- how many vector candidates to rerank (default: 20)
        query_rewrite_enabled    -- expand queries before search (default: False)
        temporal_reasoning_enabled -- LLM time reasoning (default: True)
    """

    def __init__(self, base_dir=None):
        """
        Args:
            base_dir: Override base directory.
                      Defaults to ~/.engram or ENGRAM_BASE_DIR env var.
        """
        env_base = os.environ.get("ENGRAM_BASE_DIR")
        self._base_dir = Path(base_dir or env_base or DEFAULT_BASE_DIR).expanduser()
        self._config: dict = {}
        self._patterns_data: dict = {}
        self._learned_data: dict = {}
        self._config_path = self._base_dir / "config.toml"
        self._patterns_path = self._base_dir / "patterns.toml"
        self._learned_path = self._base_dir / "learned_patterns.toml"

        # Load main config
        self._config = _load_toml(self._config_path)

        # Fallback: try legacy config.yaml if config.toml doesn't exist
        if not self._config and (self._base_dir / "config.yaml").exists():
            try:
                import yaml
                with open(self._base_dir / "config.yaml", "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
            except Exception:
                self._config = {}

        # Load patterns (user-configured)
        self._patterns_data = _load_toml(self._patterns_path)

        # Load learned patterns (auto-discovered)
        self._learned_data = _load_toml(self._learned_path)

    # ── Directory properties ────────────────────────────────────────────────

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def memories_dir(self) -> Path:
        return self._base_dir / "memories"

    @property
    def projects_dir(self) -> Path:
        return self._base_dir / "projects"

    @property
    def facts_dir(self) -> Path:
        return self._base_dir / "facts"

    @property
    def index_dir(self) -> Path:
        return self._base_dir / ".index"

    @property
    def identity_path(self) -> Path:
        return self._base_dir / "identity.md"

    @property
    def config_path(self) -> Path:
        return self._config_path

    @property
    def patterns_path(self) -> Path:
        return self._patterns_path

    @property
    def learned_patterns_path(self) -> Path:
        return self._learned_path

    def reload_learned_patterns(self) -> None:
        """Reload learned_patterns.toml from disk (e.g. after learn.py writes new patterns)."""
        self._learned_data = _load_toml(self._learned_path)

    # ── LLM feature toggles ────────────────────────────────────────────────

    def _llm_section(self) -> dict:
        """Return the ``[llm]`` section of the config (may be empty)."""
        return self._config.get("llm", {}) or {}

    # ── Temporal reasoning config ────────────────────────────────────────────

    @property
    def temporal_reasoning_enabled(self) -> bool:
        """
        True if LLM temporal reasoning is enabled.

        Default: True when think_fn is provided.
        Override with ``temporal_reasoning = false`` under ``[llm]``
        or env var ENGRAM_TEMPORAL_REASONING=0.
        """
        env_val = os.environ.get("ENGRAM_TEMPORAL_REASONING")
        if env_val is not None:
            return env_val.strip().lower() not in ("0", "false", "no", "off")
        configured = self._llm_section().get("temporal_reasoning")
        if configured is not None:
            return bool(configured)
        return True

    # ── Learning configuration ───────────────────────────────────────────────

    @property
    def promotion_threshold(self) -> int:
        """
        Number of times a candidate keyword must be seen before promotion.

        Default: 2.  Override with ``promotion_threshold`` under ``[learning]``
        or env var ENGRAM_PROMOTION_THRESHOLD.
        """
        env_val = os.environ.get("ENGRAM_PROMOTION_THRESHOLD")
        if env_val:
            try:
                return max(1, int(env_val))
            except (ValueError, TypeError):
                pass
        learning = self._config.get("learning", {}) or {}
        configured = learning.get("promotion_threshold")
        if configured is not None:
            try:
                return max(1, int(configured))
            except (ValueError, TypeError):
                pass
        return 2

    # ── Pattern configuration (from patterns.toml) ──────────────────────────

    def _get_patterns(self, category: str, key: str) -> List[str]:
        """
        Merge built-in + user-configured + learned patterns.

        Merge order: builtin → patterns.toml → learned_patterns.toml
        If ``replace = true`` in patterns.toml, builtins are skipped
        (but learned patterns are still appended).

        Args:
            category: Top-level TOML section ("quality", "conflicts", "temporal")
            key: Pattern key within section ("decision_markers", etc.)

        Returns:
            List of regex pattern strings.
        """
        builtin = _BUILTIN_PATTERNS.get(category, {}).get(key, [])

        replace_mode = bool(self._patterns_data.get("replace", False))

        cat_section = self._patterns_data.get(category, {}) or {}
        user_patterns = cat_section.get(key)

        # Start with builtins (unless replace mode)
        if replace_mode and user_patterns and isinstance(user_patterns, list):
            result = [p for p in user_patterns if isinstance(p, str)]
        elif user_patterns and isinstance(user_patterns, list):
            valid_user = [p for p in user_patterns if isinstance(p, str)]
            result = list(builtin) + valid_user
        else:
            result = list(builtin)

        # Append learned patterns (auto-discovered keywords → wrapped as regex)
        learned_section = self._learned_data.get(category, {}) or {}
        learned_keywords = learned_section.get(key)
        if learned_keywords and isinstance(learned_keywords, list):
            for kw in learned_keywords:
                if isinstance(kw, str) and kw:
                    # Wrap plain keyword as regex: keyword → (?:keyword)
                    # Escape regex special chars in the keyword
                    escaped = re.escape(kw)
                    result.append(f"(?:{escaped})")

        return result

    # ── Quality gate patterns ────────────────────────────────────────────────

    @property
    def quality_decision_markers(self) -> List[str]:
        """Regex patterns that identify decision-related content (importance +1.0)."""
        return self._get_patterns("quality", "decision_markers")

    @property
    def quality_milestone_markers(self) -> List[str]:
        """Regex patterns that identify milestone content (importance +1.0)."""
        return self._get_patterns("quality", "milestone_markers")

    @property
    def quality_problem_markers(self) -> List[str]:
        """Regex patterns that identify problem-solving content (importance +0.5)."""
        return self._get_patterns("quality", "problem_markers")

    @property
    def quality_noise_patterns(self) -> List[str]:
        """Regex patterns for noise content that should be filtered out."""
        return self._get_patterns("quality", "noise_patterns")

    # ── Conflict detection patterns ──────────────────────────────────────────

    @property
    def conflict_supersede_signals(self) -> List[str]:
        """Regex patterns that signal technology/tool transitions."""
        return self._get_patterns("conflicts", "supersede_signals")

    # ── Temporal detection patterns ──────────────────────────────────────────

    @property
    def temporal_markers(self) -> List[str]:
        """Regex patterns that identify time-related queries."""
        return self._get_patterns("temporal", "markers")

    # ── Exclusive predicates ────────────────────────────────────────────────

    @property
    def exclusive_predicates(self) -> Dict[str, str]:
        """
        Dict of predicate -> scope for conflict detection.

        Merged from built-in defaults + config.toml overrides.
        """
        merged: Dict[str, str] = {}
        merged.update(_DEFAULT_EXCLUSIVE_PREDICATES)
        merged.update(_DEFAULT_NON_EXCLUSIVE_PREDICATES)

        overrides = self._config.get("exclusive_predicates")
        if isinstance(overrides, dict):
            for pred, scope in overrides.items():
                merged[str(pred)] = str(scope)

        return merged

    # ── Initialisation ──────────────────────────────────────────────────────

    def init(self) -> Path:
        """
        Create the complete directory structure and default files.

        Creates:
            ~/.engram/
            ~/.engram/memories/
            ~/.engram/projects/
            ~/.engram/facts/
            ~/.engram/.index/
            ~/.engram/config.toml    (if doesn't exist)
            ~/.engram/patterns.toml  (if doesn't exist)
            ~/.engram/identity.md    (if doesn't exist, with template)

        Returns:
            Path to config.toml.
        """
        dirs = [
            self.base_dir,
            self.memories_dir,
            self.projects_dir,
            self.facts_dir,
            self.index_dir,
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            _chmod_safe(d, 0o700)

        # config.toml
        if not self._config_path.exists():
            self._config_path.write_text(_DEFAULT_CONFIG_TOML, encoding="utf-8")
            _chmod_safe(self._config_path, 0o600)
            # Re-read the freshly-written config
            self._config = _load_toml(self._config_path)

        # patterns.toml
        if not self._patterns_path.exists():
            self._patterns_path.write_text(_DEFAULT_PATTERNS_TOML, encoding="utf-8")
            _chmod_safe(self._patterns_path, 0o600)
            self._patterns_data = _load_toml(self._patterns_path)

        # identity.md
        if not self.identity_path.exists():
            content = _DEFAULT_IDENTITY_MD.format(
                created=datetime.now().strftime("%Y-%m-%d"),
            )
            self.identity_path.write_text(content, encoding="utf-8")
            _chmod_safe(self.identity_path, 0o600)

        return self._config_path

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self):
        """Write current config back to config.toml."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        toml_text = _dump_toml_simple(self._config)
        self._config_path.write_text(toml_text, encoding="utf-8")
        _chmod_safe(self._config_path, 0o600)

    # ── Dunder helpers ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"EngramConfig(base_dir={str(self._base_dir)!r})"


# ── Module-level helpers ────────────────────────────────────────────────────


def _chmod_safe(path: Path, mode: int) -> None:
    """Set *path* permissions to *mode*, silently ignoring errors on Windows."""
    try:
        path.chmod(mode)
    except OSError:
        # Windows doesn't support Unix permission bits — that's fine.
        pass
