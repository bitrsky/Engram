"""
test_llm.py -- Unit tests for the LLM callback protocol and prompt builders.

Tests temporal reasoning and fact
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


class TestBuildDeepSearchPrompt:
    def test_returns_system_and_user(self):
        from engram.llm import build_deep_search_prompt

        system, user = build_deep_search_prompt(
            query="What database does Alice use?",
            base_dir="/home/user/.engram",
            vector_hits=[],
        )
        assert "/home/user/.engram" in system
        assert "memories/" in system
        assert "facts/" in system
        assert "projects/" in system
        assert "What database does Alice use?" in user

    def test_includes_vector_hints(self):
        from engram.llm import build_deep_search_prompt
        from engram.index import SearchHit

        hits = [
            SearchHit(
                id="session_5", content="Alice uses PostgreSQL for the main DB",
                similarity=0.85, project="myproj", topics=["db"],
                memory_type="conversation", importance=3.0,
                created="2025-01-15", file_path="/path/to/session_5.md",
            ),
            SearchHit(
                id="fact_myproj_abc123", content="Alice uses_database PostgreSQL",
                similarity=0.92, project="myproj", topics=[],
                memory_type="fact", importance=5.0,
                created="2025-01", file_path="/path/to/facts/myproj.md",
            ),
        ]

        system, user = build_deep_search_prompt(
            query="What database?",
            base_dir="/tmp/engram",
            vector_hits=hits,
        )
        # Should have both hits in the prompt
        assert "session_5" in user
        assert "type=conversation" in user
        assert "type=fact" in user
        assert "fact: Alice uses_database PostgreSQL" in user
        assert "preview:" in user  # conversation hit has preview


class TestParseDeepSearchResponse:
    def test_parse_found(self):
        from engram.llm import parse_deep_search_response

        response = (
            "Based on the memories, Alice switched to PostgreSQL.\n\n"
            "ANSWER: PostgreSQL\n"
            "EVIDENCE: session_5, session_12"
        )
        parsed = parse_deep_search_response(response)
        assert parsed["found"] is True
        assert parsed["answer"] == "PostgreSQL"
        assert parsed["evidence"] == ["session_5", "session_12"]

    def test_parse_not_found(self):
        from engram.llm import parse_deep_search_response

        response = "ANSWER: NOT_FOUND\nEVIDENCE: none"
        parsed = parse_deep_search_response(response)
        assert parsed["found"] is False
        assert parsed["evidence"] == []

    def test_parse_empty(self):
        from engram.llm import parse_deep_search_response

        parsed = parse_deep_search_response("")
        assert parsed["found"] is False
        assert parsed["answer"] == "NOT_FOUND"

    def test_parse_none(self):
        from engram.llm import parse_deep_search_response

        parsed = parse_deep_search_response(None)
        assert parsed["found"] is False