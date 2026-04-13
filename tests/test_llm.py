"""
test_llm.py -- Unit tests for the LLM callback protocol and prompt builders.

Tests query rewrite, reranking via callback, temporal reasoning, and fact
extraction via callback â€” all with mocked LLM functions (no real LLM needed).
"""

import pytest
from dataclasses import dataclass, field
from typing import Optional


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


@dataclass
class MockSearchHit:
    """Minimal SearchHit mock for testing."""
    id: str
    content: str
    similarity: float = 0.9
    project: str = ""
    topics: list = field(default_factory=list)
    memory_type: str = "note"
    importance: float = 3.0
    created: str = "2026-01-15"
    file_path: str = ""


def make_think_fn(response: str):
    """Create a mock LLM callback that returns a fixed response."""
    def think_fn(prompt: str, system: str = "", **kwargs) -> Optional[str]:
        return response
    return think_fn


def make_failing_think_fn():
    """Create a mock LLM callback that raises an exception."""
    def think_fn(prompt: str, system: str = "", **kwargs) -> Optional[str]:
        raise RuntimeError("LLM service unavailable")
    return think_fn


def make_none_think_fn():
    """Create a mock LLM callback that returns None."""
    def think_fn(prompt: str, system: str = "", **kwargs) -> Optional[str]:
        return None
    return think_fn


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ThinkFn Protocol
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestThinkFnProtocol:
    """Test the ThinkFn protocol."""

    def test_protocol_is_runtime_checkable(self):
        from engram.llm import ThinkFn

        def my_fn(prompt: str, system: str = "", **kwargs) -> Optional[str]:
            return "hello"

        assert isinstance(my_fn, ThinkFn)

    def test_lambda_satisfies_protocol(self):
        from engram.llm import ThinkFn

        fn = lambda prompt, system="", **kw: "hello"
        assert isinstance(fn, ThinkFn)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Query Rewrite
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestRewriteQuery:
    """Test query rewrite via LLM callback."""

    def test_basic_rewrite(self):
        from engram.llm import rewrite_query

        think_fn = make_think_fn("authentication provider choices Clerk Auth0 comparison")
        result = rewrite_query("what auth?", think_fn)
        assert "Clerk" in result or "auth" in result.lower()
        assert result != "what auth?"

    def test_rewrite_strips_quotes(self):
        from engram.llm import rewrite_query

        think_fn = make_think_fn('"expanded query about databases"')
        result = rewrite_query("db?", think_fn)
        assert not result.startswith('"')
        assert not result.endswith('"')

    def test_rewrite_returns_original_on_failure(self):
        from engram.llm import rewrite_query

        think_fn = make_failing_think_fn()
        result = rewrite_query("what auth?", think_fn)
        assert result == "what auth?"

    def test_rewrite_returns_original_on_none(self):
        from engram.llm import rewrite_query

        think_fn = make_none_think_fn()
        result = rewrite_query("what auth?", think_fn)
        assert result == "what auth?"

    def test_rewrite_returns_original_on_empty(self):
        from engram.llm import rewrite_query

        think_fn = make_think_fn("")
        result = rewrite_query("what auth?", think_fn)
        assert result == "what auth?"

    def test_rewrite_returns_original_on_too_long(self):
        from engram.llm import rewrite_query

        think_fn = make_think_fn("x" * 600)
        result = rewrite_query("what auth?", think_fn)
        assert result == "what auth?"

    def test_rewrite_returns_original_on_too_short(self):
        from engram.llm import rewrite_query

        think_fn = make_think_fn("ab")
        result = rewrite_query("what auth?", think_fn)
        assert result == "what auth?"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Reranking via Callback
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestRerankWithLLM:
    """Test reranking via LLM callback."""

    def test_basic_rerank(self):
        from engram.llm import rerank_with_llm

        candidates = [
            MockSearchHit(id="a", content="Document A"),
            MockSearchHit(id="b", content="Document B"),
            MockSearchHit(id="c", content="Document C"),
            MockSearchHit(id="d", content="Document D"),
            MockSearchHit(id="e", content="Document E"),
        ]

        think_fn = make_think_fn("[3, 5, 1]")
        result = rerank_with_llm("test query", candidates, think_fn, top_k=3)

        assert len(result) == 3
        assert result[0].id == "c"  # index 3 -> 0-based 2
        assert result[1].id == "e"  # index 5 -> 0-based 4
        assert result[2].id == "a"  # index 1 -> 0-based 0

    def test_rerank_fallback_on_failure(self):
        from engram.llm import rerank_with_llm

        candidates = [
            MockSearchHit(id="a", content="Doc A"),
            MockSearchHit(id="b", content="Doc B"),
            MockSearchHit(id="c", content="Doc C"),
        ]

        think_fn = make_failing_think_fn()
        result = rerank_with_llm("test", candidates, think_fn, top_k=2)

        assert len(result) == 2
        assert result[0].id == "a"
        assert result[1].id == "b"

    def test_rerank_fallback_on_none(self):
        from engram.llm import rerank_with_llm

        candidates = [
            MockSearchHit(id="a", content="Doc A"),
            MockSearchHit(id="b", content="Doc B"),
        ]

        think_fn = make_none_think_fn()
        result = rerank_with_llm("test", candidates, think_fn, top_k=2)

        assert len(result) == 2
        assert result[0].id == "a"

    def test_rerank_fills_remaining(self):
        from engram.llm import rerank_with_llm

        candidates = [
            MockSearchHit(id="a", content="Doc A"),
            MockSearchHit(id="b", content="Doc B"),
            MockSearchHit(id="c", content="Doc C"),
            MockSearchHit(id="d", content="Doc D"),
        ]

        # LLM returns only 2 indices but we want 4
        think_fn = make_think_fn("[3, 1]")
        result = rerank_with_llm("test", candidates, think_fn, top_k=4)

        assert len(result) == 4
        assert result[0].id == "c"
        assert result[1].id == "a"
        # Remaining filled from original order
        assert result[2].id == "b"
        assert result[3].id == "d"

    def test_rerank_empty_candidates(self):
        from engram.llm import rerank_with_llm

        think_fn = make_think_fn("[1, 2]")
        result = rerank_with_llm("test", [], think_fn, top_k=2)
        assert result == []

    def test_rerank_markdown_wrapped_response(self):
        from engram.llm import rerank_with_llm

        candidates = [
            MockSearchHit(id="a", content="Doc A"),
            MockSearchHit(id="b", content="Doc B"),
            MockSearchHit(id="c", content="Doc C"),
        ]

        think_fn = make_think_fn("```json\n[2, 3, 1]\n```")
        result = rerank_with_llm("test", candidates, think_fn, top_k=3)

        assert result[0].id == "b"
        assert result[1].id == "c"
        assert result[2].id == "a"


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Parse Rerank Indices
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestParseRerankIndices:
    """Test the internal index parser."""

    def test_clean_json(self):
        from engram.llm import _parse_rerank_indices

        result = _parse_rerank_indices("[3, 1, 5, 2, 4]", n_candidates=10, top_k=5)
        assert result == [2, 0, 4, 1, 3]

    def test_text_with_numbers(self):
        from engram.llm import _parse_rerank_indices

        result = _parse_rerank_indices(
            "Most relevant: 3, 1, 5", n_candidates=10, top_k=5
        )
        assert result == [2, 0, 4]

    def test_out_of_range_filtered(self):
        from engram.llm import _parse_rerank_indices

        result = _parse_rerank_indices("[3, 15, 1]", n_candidates=10, top_k=5)
        assert result == [2, 0]

    def test_empty_response(self):
        from engram.llm import _parse_rerank_indices

        assert _parse_rerank_indices("", n_candidates=10, top_k=5) is None

    def test_no_numbers(self):
        from engram.llm import _parse_rerank_indices

        assert _parse_rerank_indices("I cannot rank", n_candidates=10, top_k=5) is None


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Temporal Query Detection
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestTemporalQuery:
    """Test temporal marker detection."""

    def test_when_did(self):
        from engram.llm import is_temporal_query

        assert is_temporal_query("when did we switch to Clerk?") is True

    def test_how_long_ago(self):
        from engram.llm import is_temporal_query

        assert is_temporal_query("how long ago was the migration?") is True

    def test_days_ago(self):
        from engram.llm import is_temporal_query

        assert is_temporal_query("what happened 3 days ago?") is True

    def test_timeline(self):
        from engram.llm import is_temporal_query

        assert is_temporal_query("show me the timeline of events") is True

    def test_non_temporal(self):
        from engram.llm import is_temporal_query

        assert is_temporal_query("what database do we use?") is False

    def test_non_temporal_simple(self):
        from engram.llm import is_temporal_query

        assert is_temporal_query("explain the auth architecture") is False


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Temporal Reasoning
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestAnswerTemporal:
    """Test temporal reasoning via LLM callback."""

    def test_temporal_answer(self):
        from engram.llm import answer_temporal

        hits = [
            MockSearchHit(
                id="a",
                content="We switched to Clerk for authentication.",
                created="2026-01-15",
            ),
        ]

        think_fn = make_think_fn("The switch to Clerk happened on January 15, 2026.")
        result = answer_temporal("when did we switch auth?", hits, think_fn)

        assert result is not None
        assert "2026" in result

    def test_non_temporal_query_returns_none(self):
        from engram.llm import answer_temporal

        hits = [MockSearchHit(id="a", content="some content")]
        think_fn = make_think_fn("This is not a temporal answer.")
        result = answer_temporal("what database do we use?", hits, think_fn)

        assert result is None

    def test_unable_to_determine_returns_none(self):
        from engram.llm import answer_temporal

        hits = [MockSearchHit(id="a", content="some content", created="2026-01-15")]
        think_fn = make_think_fn("Unable to determine from available memories.")
        result = answer_temporal("when did we start?", hits, think_fn)

        assert result is None

    def test_empty_hits_returns_none(self):
        from engram.llm import answer_temporal

        think_fn = make_think_fn("answer")
        result = answer_temporal("when did we start?", [], think_fn)

        assert result is None

    def test_llm_failure_returns_none(self):
        from engram.llm import answer_temporal

        hits = [MockSearchHit(id="a", content="content", created="2026-01-15")]
        think_fn = make_failing_think_fn()
        result = answer_temporal("when did we start?", hits, think_fn)

        assert result is None


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Fact Extraction via Callback
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestExtractFactsViaCallback:
    """Test fact extraction via LLM callback."""

    def test_basic_extraction(self):
        from engram.llm import extract_facts_via_callback

        llm_response = '''[
            {"subject": "saas-app", "predicate": "uses", "object": "Clerk", "confidence": 0.95, "temporal": "2026-01", "conflicts_with": ""}
        ]'''

        think_fn = make_think_fn(llm_response)
        facts = extract_facts_via_callback(
            "We decided to use Clerk for auth.",
            think_fn,
            project="saas-app",
        )

        assert len(facts) == 1
        assert facts[0].subject == "saas-app"
        assert facts[0].predicate == "uses"
        assert facts[0].object == "Clerk"
        assert facts[0].confidence == 0.95

    def test_extraction_with_existing_facts(self):
        from engram.llm import extract_facts_via_callback

        @dataclass
        class MockFact:
            subject: str
            predicate: str
            object: str

        existing = [MockFact("saas-app", "uses", "Auth0")]

        llm_response = '''[
            {"subject": "saas-app", "predicate": "uses", "object": "Clerk", "confidence": 0.9, "temporal": "", "conflicts_with": "saas-app â†’ uses â†’ Auth0"}
        ]'''

        think_fn = make_think_fn(llm_response)
        facts = extract_facts_via_callback(
            "We switched from Auth0 to Clerk.",
            think_fn,
            project="saas-app",
            existing_facts=existing,
        )

        assert len(facts) == 1
        assert facts[0].conflicts_with == "saas-app â†’ uses â†’ Auth0"

    def test_extraction_on_failure_returns_empty(self):
        from engram.llm import extract_facts_via_callback

        think_fn = make_failing_think_fn()
        facts = extract_facts_via_callback("some text", think_fn)
        assert facts == []

    def test_extraction_on_none_returns_empty(self):
        from engram.llm import extract_facts_via_callback

        think_fn = make_none_think_fn()
        facts = extract_facts_via_callback("some text", think_fn)
        assert facts == []

    def test_extraction_on_garbage_returns_empty(self):
        from engram.llm import extract_facts_via_callback

        think_fn = make_think_fn("This is not JSON at all")
        facts = extract_facts_via_callback("some text", think_fn)
        assert facts == []


# ===========================================================================
# deep_search tests
# ===========================================================================


class TestDeepSearch:
    """Test the deep_search() function."""

    def test_deep_search_finds_answer(self):
        from engram.llm import deep_search_simple as deep_search

        hits = [MockSearchHit(id="session_5", content="Alice went to the park")]
        listing = [
            {"filename": "session_1.md", "id": "session_1", "created": "2024-01-01",
             "preview": "Hi there!", "content": "Hi there! Long time no see."},
            {"filename": "session_2.md", "id": "session_2", "created": "2024-01-15",
             "preview": "Caroline research", "content": "Caroline told me about her research on climate change."},
        ]

        def mock_think(prompt, system="", **kw):
            return "Caroline researched climate change. Evidence found in session_2.md."

        result = deep_search("What did Caroline research?", mock_think, hits, listing)
        assert result is not None
        assert "climate change" in result

    def test_deep_search_returns_none_on_not_found(self):
        from engram.llm import deep_search_simple as deep_search

        listing = [
            {"filename": "session_1.md", "id": "s1", "created": "2024-01-01",
             "preview": "Hello", "content": "Hello world"},
        ]

        def mock_think(prompt, system="", **kw):
            return "NOT_FOUND"

        result = deep_search("What is the meaning of life?", mock_think, [], listing)
        assert result is None

    def test_deep_search_returns_none_on_empty_listing(self):
        from engram.llm import deep_search_simple as deep_search

        def mock_think(prompt, system="", **kw):
            return "Some answer"

        result = deep_search("query", mock_think, [], [])
        assert result is None

    def test_deep_search_returns_none_on_failure(self):
        from engram.llm import deep_search_simple as deep_search

        listing = [{"filename": "a.md", "id": "a", "created": "", "preview": "x", "content": "x"}]

        def failing_think(prompt, system="", **kw):
            raise RuntimeError("boom")

        result = deep_search("query", failing_think, [], listing)
        assert result is None
