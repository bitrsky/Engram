"""
config.py — Configuration management for Engram.

Manages ~/.engram/ directory structure and config.yaml.
Priority: env vars > config file > defaults.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml


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

# ── Default config.yaml content (with commented examples) ───────────────────
_DEFAULT_CONFIG_YAML = """\
# ─────────────────────────────────────────────────────────────────────────────
# Engram configuration
# ─────────────────────────────────────────────────────────────────────────────
# Priority: environment variables > this file > built-in defaults.

# ── LLM features ────────────────────────────────────────────────────────────
# LLM capability is injected by the host agent via think_fn callback.
# These toggles control which LLM features are enabled.
llm:
  # rerank: true                 # enable LLM reranking (default: true when think_fn provided)
  # rerank_candidates: 20        # number of vector candidates to rerank
  # query_rewrite: false          # rewrite vague queries before search (adds ~200ms)
  # temporal_reasoning: true      # LLM reasoning for time-related questions

# ── Exclusive predicates ────────────────────────────────────────────────────
# Predicates listed here trigger conflict detection: when a new fact is filed
# for the same subject + predicate, the old fact is superseded.
# Scope "subject" = one value per subject.  "none" = no exclusivity.
#
# exclusive_predicates:
#   uses_database: "subject"
#   uses_auth: "subject"
#   status: "subject"
#   loves: "none"               # non-exclusive — multiple values allowed
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


class EngramConfig:
    """
    Configuration manager for Engram.

    Reads from ~/.engram/config.yaml with env var overrides.

    Attributes (all Path properties):
        base_dir      — ~/.engram/
        memories_dir  — ~/.engram/memories/
        projects_dir  — ~/.engram/projects/
        facts_dir     — ~/.engram/facts/
        index_dir     — ~/.engram/.index/
        identity_path — ~/.engram/identity.md
        config_path   — ~/.engram/config.yaml

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
        self._config_path = self._base_dir / "config.yaml"

        if self._config_path.exists():
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
            except (yaml.YAMLError, OSError):
                self._config = {}

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

    # ── LLM feature toggles ────────────────────────────────────────────────

    def _llm_section(self) -> dict:
        """Return the ``llm:`` section of the config (may be empty)."""
        return self._config.get("llm", {}) or {}

    # ── Rerank config properties ─────────────────────────────────────────────

    @property
    def rerank_enabled(self) -> bool:
        """
        True if LLM reranking is enabled.

        Can be explicitly disabled with ``llm.rerank: false`` in config
        or env var ENGRAM_RERANK=0.  Default: True.

        Note: actual reranking only happens when a think_fn callback is
        provided at runtime.
        """
        env_val = os.environ.get("ENGRAM_RERANK")
        if env_val is not None:
            return env_val.strip().lower() not in ("0", "false", "no", "off")
        configured = self._llm_section().get("rerank")
        if configured is not None:
            return bool(configured)
        return True

    @property
    def rerank_candidates(self) -> int:
        """
        Number of vector search candidates to fetch for reranking.

        Default: 20. Override with ``llm.rerank_candidates`` in config
        or env var ENGRAM_RERANK_CANDIDATES.
        """
        env_val = os.environ.get("ENGRAM_RERANK_CANDIDATES")
        if env_val:
            try:
                return max(1, int(env_val))
            except (ValueError, TypeError):
                pass
        configured = self._llm_section().get("rerank_candidates")
        if configured is not None:
            try:
                return max(1, int(configured))
            except (ValueError, TypeError):
                pass
        return 20

    # ── Query rewrite config ─────────────────────────────────────────────────

    @property
    def query_rewrite_enabled(self) -> bool:
        """
        True if LLM query rewrite is enabled.

        Rewrites vague queries into more specific search terms before
        vector search.  Adds ~200ms latency per search.

        Default: False (opt-in).  Override with ``llm.query_rewrite: true``
        in config or env var ENGRAM_QUERY_REWRITE=1.
        """
        env_val = os.environ.get("ENGRAM_QUERY_REWRITE")
        if env_val is not None:
            return env_val.strip().lower() in ("1", "true", "yes", "on")
        configured = self._llm_section().get("query_rewrite")
        if configured is not None:
            return bool(configured)
        return False

    # ── Temporal reasoning config ────────────────────────────────────────────

    @property
    def temporal_reasoning_enabled(self) -> bool:
        """
        True if LLM temporal reasoning is enabled.

        Detects time-related questions and uses LLM to reason about dates
        and durations from memory timestamps.  Only triggers when temporal
        markers are detected in the query (low cost gate).

        Default: True when think_fn is provided.
        Override with ``llm.temporal_reasoning: false`` in config
        or env var ENGRAM_TEMPORAL_REASONING=0.
        """
        env_val = os.environ.get("ENGRAM_TEMPORAL_REASONING")
        if env_val is not None:
            return env_val.strip().lower() not in ("0", "false", "no", "off")
        configured = self._llm_section().get("temporal_reasoning")
        if configured is not None:
            return bool(configured)
        # Default: enabled (actual gating is done by temporal marker detection)
        return True

    # ── Exclusive predicates ────────────────────────────────────────────────

    @property
    def exclusive_predicates(self) -> Dict[str, str]:
        """
        Dict of predicate -> scope for conflict detection.

        Merged from built-in defaults + config.yaml overrides.

        Default exclusives (scope ``"subject"``):
            uses_database, uses_auth, uses_framework, deployed_on,
            assigned_to, status, led_by, managed_by

        Default non-exclusive (scope ``"none"``):
            loves, knows, works_with, interested_in, uses_tool
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
        Create the complete directory structure and default config.yaml.

        Creates:
            ~/.engram/
            ~/.engram/memories/
            ~/.engram/projects/
            ~/.engram/facts/
            ~/.engram/.index/
            ~/.engram/config.yaml  (if doesn't exist)
            ~/.engram/identity.md  (if doesn't exist, with template)

        Returns:
            Path to config.yaml.
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

        # config.yaml
        if not self._config_path.exists():
            self._config_path.write_text(_DEFAULT_CONFIG_YAML, encoding="utf-8")
            _chmod_safe(self._config_path, 0o600)
            # Re-read the freshly-written config
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    self._config = yaml.safe_load(f) or {}
            except (yaml.YAMLError, OSError):
                self._config = {}

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
        """Write current config back to config.yaml."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self._config,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
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
