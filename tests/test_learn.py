"""
test_learn.py — Tests for the adaptive pattern learning system.

Tests the full pipeline:
1. quality_gate_detailed returns matched categories
2. learn_from_extraction discovers new keywords from LLM facts
3. Candidate promotion after promotion_threshold hits
4. Config picks up learned patterns (builtin + user + learned merge)
5. End-to-end: Chinese content → learning → pattern becomes effective
6. Persistence: save → load round-trip for candidates and active patterns
"""

import re
from pathlib import Path
from typing import Set

import pytest

from engram.config import EngramConfig, _BUILTIN_PATTERNS
from engram.extract import FactCandidate
from engram.learn import (
    _DEFAULT_PROMOTION_THRESHOLD,
    LearnResult,
    _extract_keywords,
    _find_relevant_sentence,
    _is_cjk,
    _keyword_already_covered,
    _load_learned_state,
    _promote_candidates,
    _resolve_category,
    _save_learned_state,
    _tokenize,
    _upsert_candidate,
    learn_from_extraction,
)
from engram.quality import quality_gate, quality_gate_detailed


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def learn_config(tmp_path):
    """Config with isolated dir for learning tests."""
    base = tmp_path / ".engram"
    base.mkdir()
    (base / "memories").mkdir()
    (base / "projects").mkdir()
    (base / "facts").mkdir()
    (base / ".index").mkdir()

    config_path = base / "config.toml"
    config_path.write_text("[llm]\n", encoding="utf-8")

    return EngramConfig(base_dir=str(base))


def _make_fact(
    subject="app",
    predicate="decision",
    obj="PostgreSQL",
    source_text="",
    conflicts_with="",
) -> FactCandidate:
    """Helper to create a FactCandidate."""
    return FactCandidate(
        subject=subject,
        predicate=predicate,
        object=obj,
        confidence=0.9,
        source_text=source_text,
        conflicts_with=conflicts_with,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Part 1: quality_gate_detailed — returns matched categories
# ═════════════════════════════════════════════════════════════════════════════


class TestQualityGateDetailed:
    """Test that quality_gate_detailed reports which categories matched."""

    def test_english_decision_returns_category(self):
        ok, imp, cats = quality_gate_detailed(
            "We decided to use PostgreSQL for the database."
        )
        assert ok is True
        assert imp > 3.0
        assert "decision_markers" in cats

    def test_english_milestone_returns_category(self):
        ok, imp, cats = quality_gate_detailed(
            "We finally shipped the new version to production."
        )
        assert ok is True
        assert "milestone_markers" in cats

    def test_english_problem_returns_category(self):
        ok, imp, cats = quality_gate_detailed(
            "The root cause was a race condition in the connection pool."
        )
        assert ok is True
        assert "problem_markers" in cats

    def test_noise_returns_noise_category(self):
        ok, imp, cats = quality_gate_detailed("ok")
        assert ok is False
        assert "noise_patterns" in cats

    def test_chinese_no_match_empty_categories(self):
        """Chinese text should NOT match English-only builtin patterns."""
        ok, imp, cats = quality_gate_detailed(
            "我们决定使用 PostgreSQL 作为数据库"
        )
        assert ok is True
        # No English pattern matches Chinese text
        assert "decision_markers" not in cats

    def test_backward_compat_quality_gate(self):
        """quality_gate() still returns (bool, float) tuple."""
        result = quality_gate("We decided to use PostgreSQL for the database.")
        assert len(result) == 2
        assert result[0] is True
        assert result[1] > 3.0


# ═════════════════════════════════════════════════════════════════════════════
# Part 2: Keyword extraction and tokenization
# ═════════════════════════════════════════════════════════════════════════════


class TestTokenization:
    """Test CJK-aware tokenizer."""

    def test_english_tokenize(self):
        tokens = _tokenize("We decided to use PostgreSQL")
        assert "We" in tokens
        assert "decided" in tokens
        assert "PostgreSQL" in tokens

    def test_chinese_tokenize(self):
        tokens = _tokenize("我们决定用")
        # CJK chars tokenized individually
        assert "我" in tokens
        assert "们" in tokens
        assert "决" in tokens
        assert "定" in tokens
        assert "用" in tokens

    def test_mixed_tokenize(self):
        tokens = _tokenize("我们决定用PostgreSQL")
        # Both CJK chars and western word
        assert "PostgreSQL" in tokens
        assert "决" in tokens

    def test_cjk_detection(self):
        assert _is_cjk("决定") is True
        assert _is_cjk("decided") is False
        assert _is_cjk("决定了abc") is True


class TestKeywordExtraction:
    """Test keyword extraction from content."""

    def test_extract_from_english(self, learn_config):
        fact = _make_fact(
            subject="app",
            predicate="decision",
            obj="PostgreSQL",
            source_text="We decided to use PostgreSQL",
        )
        keywords = _extract_keywords(
            "We decided to use PostgreSQL", fact, learn_config
        )
        # Should contain "decided" or similar (after removing subject/object)
        kw_lower = [k.lower() for k in keywords]
        assert any("decided" in k for k in kw_lower)

    def test_extract_from_chinese(self, learn_config):
        fact = _make_fact(
            subject="app",
            predicate="decision",
            obj="PostgreSQL",
            source_text="我们决定用PostgreSQL",
        )
        keywords = _extract_keywords(
            "我们决定用PostgreSQL", fact, learn_config
        )
        # After removing "app" (subject not in text) and "PostgreSQL",
        # should find Chinese chars like 决, 定
        assert len(keywords) > 0

    def test_find_relevant_sentence_with_source_text(self, learn_config):
        fact = _make_fact(source_text="We chose Redis")
        result = _find_relevant_sentence("Some intro. We chose Redis. Goodbye.", fact)
        assert result == "We chose Redis"

    def test_find_relevant_sentence_by_object(self, learn_config):
        fact = _make_fact(
            subject="app", obj="Redis", source_text=""
        )
        result = _find_relevant_sentence(
            "First sentence. We are using Redis now. Last sentence.", fact
        )
        assert "Redis" in result


# ═════════════════════════════════════════════════════════════════════════════
# Part 3: Category resolution
# ═════════════════════════════════════════════════════════════════════════════


class TestCategoryResolution:
    """Test predicate → pattern category mapping."""

    def test_decision_predicates(self):
        fact = _make_fact(predicate="decision")
        assert _resolve_category("decision", fact) == "decision_markers"
        assert _resolve_category("chose", fact) == "decision_markers"
        assert _resolve_category("decided", fact) == "decision_markers"

    def test_milestone_predicates(self):
        fact = _make_fact(predicate="deployed")
        assert _resolve_category("deployed", fact) == "milestone_markers"
        assert _resolve_category("shipped", fact) == "milestone_markers"
        assert _resolve_category("launched", fact) == "milestone_markers"

    def test_problem_predicates(self):
        fact = _make_fact(predicate="fixed")
        assert _resolve_category("fixed", fact) == "problem_markers"
        assert _resolve_category("resolved", fact) == "problem_markers"

    def test_status_milestone(self):
        fact = _make_fact(predicate="status", obj="deployed")
        assert _resolve_category("status", fact) == "milestone_markers"

    def test_status_problem(self):
        fact = _make_fact(predicate="status", obj="broken")
        assert _resolve_category("status", fact) == "problem_markers"

    def test_unknown_predicate(self):
        fact = _make_fact(predicate="unknown_stuff")
        assert _resolve_category("unknown_stuff", fact) is None


# ═════════════════════════════════════════════════════════════════════════════
# Part 4: Candidate management and promotion
# ═════════════════════════════════════════════════════════════════════════════


class TestCandidateManagement:
    """Test candidate upsert and promotion logic."""

    def test_new_candidate(self):
        state = {"active": {}, "candidates": []}
        result = LearnResult()
        changed = _upsert_candidate(
            state, "决定", "quality", "decision_markers", "2025-06-20", result
        )
        assert changed is True
        assert result.new_candidates == 1
        assert len(state["candidates"]) == 1
        assert state["candidates"][0]["keyword"] == "决定"
        assert state["candidates"][0]["hits"] == 1

    def test_existing_candidate_increments(self):
        state = {
            "active": {},
            "candidates": [{
                "keyword": "决定",
                "category": "decision_markers",
                "section": "quality",
                "hits": 1,
                "first_seen": "2025-06-20",
                "last_seen": "2025-06-20",
            }],
        }
        result = LearnResult()
        _upsert_candidate(
            state, "决定", "quality", "decision_markers", "2025-06-21", result
        )
        assert state["candidates"][0]["hits"] == 2
        assert state["candidates"][0]["last_seen"] == "2025-06-21"
        # Not a NEW candidate
        assert result.new_candidates == 0

    def test_promotion_at_threshold(self):
        state = {
            "active": {},
            "candidates": [{
                "keyword": "决定",
                "category": "decision_markers",
                "section": "quality",
                "hits": _DEFAULT_PROMOTION_THRESHOLD,
                "first_seen": "2025-06-20",
                "last_seen": "2025-06-22",
            }],
        }
        promoted = _promote_candidates(state, "2025-06-22")
        assert len(promoted) == 1
        assert promoted[0].keyword == "决定"
        # Candidate removed from candidates list
        assert len(state["candidates"]) == 0
        # Added to active
        assert "决定" in state["active"]["quality"]["decision_markers"]

    def test_no_promotion_below_threshold(self):
        state = {
            "active": {},
            "candidates": [{
                "keyword": "决定",
                "category": "decision_markers",
                "section": "quality",
                "hits": _DEFAULT_PROMOTION_THRESHOLD - 1,
                "first_seen": "2025-06-20",
                "last_seen": "2025-06-21",
            }],
        }
        promoted = _promote_candidates(state, "2025-06-22")
        assert len(promoted) == 0
        assert len(state["candidates"]) == 1  # still a candidate


# ═════════════════════════════════════════════════════════════════════════════
# Part 5: Persistence round-trip
# ═════════════════════════════════════════════════════════════════════════════


class TestPersistence:
    """Test save/load round-trip for learned patterns."""

    def test_save_and_load_active(self, tmp_path):
        path = tmp_path / "learned_patterns.toml"
        state = {
            "active": {
                "quality": {
                    "decision_markers": ["决定", "选择了"],
                    "milestone_markers": ["上线了"],
                },
            },
            "candidates": [],
        }
        _save_learned_state(path, state)
        assert path.exists()

        loaded = _load_learned_state(path)
        assert loaded["active"]["quality"]["decision_markers"] == ["决定", "选择了"]
        assert loaded["active"]["quality"]["milestone_markers"] == ["上线了"]
        assert loaded["candidates"] == []

    def test_save_and_load_candidates(self, tmp_path):
        path = tmp_path / "learned_patterns.toml"
        state = {
            "active": {},
            "candidates": [
                {
                    "keyword": "替换",
                    "category": "supersede_signals",
                    "section": "conflicts",
                    "hits": 2,
                    "first_seen": "2025-06-20",
                    "last_seen": "2025-06-21",
                },
            ],
        }
        _save_learned_state(path, state)
        loaded = _load_learned_state(path)
        assert len(loaded["candidates"]) == 1
        assert loaded["candidates"][0]["keyword"] == "替换"
        assert loaded["candidates"][0]["hits"] == 2

    def test_load_nonexistent_file(self, tmp_path):
        path = tmp_path / "does_not_exist.toml"
        state = _load_learned_state(path)
        assert state == {"active": {}, "candidates": []}

    def test_round_trip_mixed(self, tmp_path):
        """Full round-trip with both active and candidates."""
        path = tmp_path / "learned_patterns.toml"
        state = {
            "active": {
                "quality": {"decision_markers": ["决定"]},
                "conflicts": {"supersede_signals": ["迁移到"]},
            },
            "candidates": [
                {
                    "keyword": "选择",
                    "category": "decision_markers",
                    "section": "quality",
                    "hits": 1,
                    "first_seen": "2025-06-20",
                    "last_seen": "2025-06-20",
                },
            ],
        }
        _save_learned_state(path, state)
        loaded = _load_learned_state(path)

        assert loaded["active"]["quality"]["decision_markers"] == ["决定"]
        assert loaded["active"]["conflicts"]["supersede_signals"] == ["迁移到"]
        assert len(loaded["candidates"]) == 1
        assert loaded["candidates"][0]["keyword"] == "选择"


# ═════════════════════════════════════════════════════════════════════════════
# Part 6: Config integration — learned patterns merge into config
# ═════════════════════════════════════════════════════════════════════════════


class TestConfigIntegration:
    """Test that config correctly merges builtin + user + learned patterns."""

    def test_config_loads_learned_patterns(self, tmp_path):
        """Config should include learned patterns in its pattern properties."""
        base = tmp_path / ".engram"
        base.mkdir()
        for d in ["memories", "projects", "facts", ".index"]:
            (base / d).mkdir()

        (base / "config.toml").write_text("[llm]\n", encoding="utf-8")

        # Write a learned_patterns.toml with an active pattern
        (base / "learned_patterns.toml").write_text(
            '[quality]\ndecision_markers = ["决定"]\n',
            encoding="utf-8",
        )

        config = EngramConfig(base_dir=str(base))
        markers = config.quality_decision_markers

        # Should contain builtins + learned
        builtin_count = len(_BUILTIN_PATTERNS["quality"]["decision_markers"])
        assert len(markers) == builtin_count + 1

        # The last pattern should match "决定"
        assert any(re.search(p, "决定") for p in markers)

    def test_reload_learned_patterns(self, tmp_path):
        """reload_learned_patterns() should pick up newly written patterns."""
        base = tmp_path / ".engram"
        base.mkdir()
        for d in ["memories", "projects", "facts", ".index"]:
            (base / d).mkdir()

        (base / "config.toml").write_text("[llm]\n", encoding="utf-8")
        config = EngramConfig(base_dir=str(base))

        # Initially no learned patterns
        markers_before = config.quality_decision_markers
        builtin_count = len(_BUILTIN_PATTERNS["quality"]["decision_markers"])
        assert len(markers_before) == builtin_count

        # Write learned patterns
        (base / "learned_patterns.toml").write_text(
            '[quality]\ndecision_markers = ["决定", "选择"]\n',
            encoding="utf-8",
        )

        # Reload
        config.reload_learned_patterns()
        markers_after = config.quality_decision_markers
        assert len(markers_after) == builtin_count + 2


# ═════════════════════════════════════════════════════════════════════════════
# Part 7: End-to-end learning flow
# ═════════════════════════════════════════════════════════════════════════════


class TestEndToEndLearning:
    """Test the full learning pipeline: Chinese content → new patterns."""

    def test_chinese_decision_creates_candidate(self, learn_config):
        """When LLM says 'decision' but heuristic patterns miss → candidate."""
        fact = _make_fact(
            subject="app",
            predicate="decision",
            obj="PostgreSQL",
            source_text="我们决定用PostgreSQL",
        )

        # No decision_markers matched (Chinese text, English patterns)
        matched_categories: Set[str] = set()

        result = learn_from_extraction(
            content="我们决定用PostgreSQL作为数据库",
            facts=[fact],
            matched_categories=matched_categories,
            config=learn_config,
        )

        assert result.new_candidates > 0
        # Verify candidates were saved
        learned_path = learn_config.base_dir / "learned_patterns.toml"
        assert learned_path.exists()

    def test_english_decision_no_learning(self, learn_config):
        """When English patterns already match → nothing to learn."""
        fact = _make_fact(
            subject="app",
            predicate="decision",
            obj="PostgreSQL",
            source_text="We decided to use PostgreSQL",
        )

        # decision_markers already matched
        matched_categories: Set[str] = {"decision_markers"}

        result = learn_from_extraction(
            content="We decided to use PostgreSQL",
            facts=[fact],
            matched_categories=matched_categories,
            config=learn_config,
        )

        assert result.new_candidates == 0
        assert result.promoted == 0

    def test_supersede_mismatch_creates_candidate(self, learn_config):
        """When LLM detects conflict but supersede_signals miss → candidate."""
        fact = _make_fact(
            subject="app",
            predicate="uses",
            obj="PostgreSQL",
            source_text="我们把MySQL换成了PostgreSQL",
            conflicts_with="app → uses → MySQL",
        )

        matched_categories: Set[str] = set()

        result = learn_from_extraction(
            content="我们把MySQL换成了PostgreSQL",
            facts=[fact],
            matched_categories=matched_categories,
            config=learn_config,
        )

        # Should have candidates for supersede_signals
        assert result.new_candidates > 0

    def test_full_promotion_cycle(self, learn_config):
        """Simulate N rounds of same keyword → promotion → config picks it up."""
        fact = _make_fact(
            subject="app",
            predicate="decision",
            obj="Redis",
            source_text="我们决定用Redis",
        )
        matched_categories: Set[str] = set()

        # Run promotion_threshold times
        for i in range(_DEFAULT_PROMOTION_THRESHOLD):
            result = learn_from_extraction(
                content=f"我们决定用Redis来做缓存 (round {i})",
                facts=[fact],
                matched_categories=matched_categories,
                config=learn_config,
            )

        # After enough rounds, should have promoted
        assert result.promoted > 0

        # Reload and verify config picks it up
        learn_config.reload_learned_patterns()
        markers = learn_config.quality_decision_markers
        builtin_count = len(_BUILTIN_PATTERNS["quality"]["decision_markers"])
        assert len(markers) > builtin_count

    def test_promoted_pattern_actually_matches(self, learn_config):
        """After promotion, the learned keyword should match in quality_gate."""
        # Manually write a learned pattern
        learned_path = learn_config.base_dir / "learned_patterns.toml"
        learned_path.write_text(
            '[quality]\ndecision_markers = ["决定"]\n',
            encoding="utf-8",
        )
        learn_config.reload_learned_patterns()

        # Now quality_gate should boost Chinese decision text
        ok, imp, cats = quality_gate_detailed(
            "我们决定使用 PostgreSQL 作为数据库",
            config=learn_config,
        )
        assert ok is True
        assert "decision_markers" in cats
        assert imp > 3.0  # boosted by decision_marker match

    def test_no_learning_without_facts(self, learn_config):
        """Empty facts list → nothing to learn."""
        result = learn_from_extraction(
            content="some content",
            facts=[],
            matched_categories=set(),
            config=learn_config,
        )
        assert result.new_candidates == 0
        assert result.promoted == 0

    def test_already_covered_keyword_skipped(self, learn_config):
        """If a keyword is already covered by existing patterns, skip it."""
        # "decided" is already in builtin decision_markers
        fact = _make_fact(
            subject="app",
            predicate="decision",
            obj="Redis",
            source_text="We decided on Redis",
        )

        # Pretend patterns didn't match (even though they would)
        result = learn_from_extraction(
            content="We decided on Redis",
            facts=[fact],
            matched_categories=set(),
            config=learn_config,
        )

        # "decided" should be skipped as already covered
        assert result.already_covered > 0


# ═════════════════════════════════════════════════════════════════════════════
# Part 8: Edge cases
# ═════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_empty_content(self, learn_config):
        fact = _make_fact(predicate="decision", source_text="")
        result = learn_from_extraction(
            content="",
            facts=[fact],
            matched_categories=set(),
            config=learn_config,
        )
        # Should not crash
        assert isinstance(result, LearnResult)

    def test_fact_with_no_source_text(self, learn_config):
        fact = _make_fact(
            predicate="decision",
            obj="Redis",
            source_text="",
        )
        result = learn_from_extraction(
            content="We should use Redis for caching.",
            facts=[fact],
            matched_categories=set(),
            config=learn_config,
        )
        assert isinstance(result, LearnResult)

    def test_unicode_keywords_in_patterns(self, learn_config):
        """Ensure regex-escaped unicode keywords work as patterns."""
        # Manually add a learned CJK keyword
        learned_path = learn_config.base_dir / "learned_patterns.toml"
        learned_path.write_text(
            '[conflicts]\nsupersede_signals = ["换成"]\n',
            encoding="utf-8",
        )
        learn_config.reload_learned_patterns()

        signals = learn_config.conflict_supersede_signals
        # Should match Chinese text
        test_text = "我们把MySQL换成了PostgreSQL"
        matched = any(
            re.search(p, test_text, re.IGNORECASE) for p in signals
        )
        assert matched is True

    def test_concurrent_category_learning(self, learn_config):
        """Multiple categories can be learned in one call."""
        facts = [
            _make_fact(
                predicate="decision",
                obj="PostgreSQL",
                source_text="我们决定用PostgreSQL",
            ),
            _make_fact(
                predicate="deployed",
                obj="v2.0",
                source_text="v2.0上线了",
            ),
        ]

        result = learn_from_extraction(
            content="我们决定用PostgreSQL，v2.0上线了",
            facts=facts,
            matched_categories=set(),
            config=learn_config,
        )

        assert result.new_candidates > 0
