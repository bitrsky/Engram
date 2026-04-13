"""
test_rerank.py -- Unit tests for LLM reranking module.

Tests prompt building, response parsing, and fallback behavior
without requiring an actual LLM.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field


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
    created: str = ""
    file_path: str = ""


@dataclass
class MockConfig:
    """Minimal EngramConfig mock for testing."""
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = "test-key"
    llm_base_url: str = ""
    llm_available: bool = True
    rerank_enabled: bool = True
    rerank_candidates: int = 20


# ===========================================================================
# _parse_rerank_response tests
# ===========================================================================

class TestParseRerankResponse:
    """Test response parsing from various LLM output formats."""

    def setup_method(self):
        from engram.rerank import _parse_rerank_response
        self.parse = _parse_rerank_response

    def test_clean_json_array(self):
        """Standard JSON array output."""
        result = self.parse("[3, 1, 5, 2, 4]", n_candidates=10, top_k=5)
        assert result == [2, 0, 4, 1, 3]  # 1-based -> 0-based

    def test_json_with_spaces(self):
        """JSON with extra whitespace."""
        result = self.parse("  [ 3 , 1 , 5 ]  ", n_candidates=10, top_k=5)
        assert result == [2, 0, 4]

    def test_markdown_wrapped_json(self):
        """JSON wrapped in markdown code block."""
        result = self.parse("```json\n[3, 1, 5]\n```", n_candidates=10, top_k=5)
        assert result == [2, 0, 4]

    def test_numbers_with_text(self):
        """LLM returns text with numbers mixed in."""
        result = self.parse(
            "The most relevant documents are 3, 1, and 5.",
            n_candidates=10, top_k=5,
        )
        assert result == [2, 0, 4]

    def test_out_of_range_indices_filtered(self):
        """Indices outside valid range are filtered out."""
        result = self.parse("[3, 15, 1, 0, 5]", n_candidates=10, top_k=5)
        # 15 -> idx 14, out of range for 10 candidates
        # 0 -> idx -1, out of range
        assert result == [2, 0, 4]

    def test_duplicates_removed(self):
        """Duplicate indices are deduplicated."""
        result = self.parse("[3, 3, 1, 1, 5]", n_candidates=10, top_k=5)
        assert result == [2, 0, 4]

    def test_empty_response(self):
        """Empty response returns None."""
        assert self.parse("", n_candidates=10, top_k=5) is None

    def test_no_numbers(self):
        """Response with no numbers returns None."""
        assert self.parse("I cannot rank these.", n_candidates=10, top_k=5) is None

    def test_top_k_truncation(self):
        """Result is truncated to top_k."""
        result = self.parse("[1, 2, 3, 4, 5, 6, 7]", n_candidates=10, top_k=3)
        assert len(result) == 3
        assert result == [0, 1, 2]


# ===========================================================================
# _build_rerank_prompt tests
# ===========================================================================

class TestBuildRerankPrompt:
    """Test prompt construction."""

    def test_basic_prompt(self):
        from engram.rerank import _build_rerank_prompt

        candidates = [
            MockSearchHit(id="a", content="The auth provider is Clerk."),
            MockSearchHit(id="b", content="Database migration to Postgres."),
            MockSearchHit(id="c", content="Sprint planning meeting notes."),
        ]
        prompt = _build_rerank_prompt("What auth provider?", candidates, top_k=2)

        assert "What auth provider?" in prompt
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt
        assert "auth provider is Clerk" in prompt
        assert "top 2" in prompt

    def test_long_content_truncated(self):
        from engram.rerank import _build_rerank_prompt, _MAX_DOC_CHARS

        long_content = "x" * 1000
        candidates = [MockSearchHit(id="a", content=long_content)]
        prompt = _build_rerank_prompt("query", candidates, top_k=1)

        # The content in the prompt should be truncated
        assert "x" * _MAX_DOC_CHARS in prompt
        assert "..." in prompt


# ===========================================================================
# rerank() integration tests (with mocked LLM)
# ===========================================================================

class TestRerank:
    """Test the full rerank() function with mocked LLM calls."""

    def test_rerank_reorders_candidates(self):
        from engram.rerank import rerank

        candidates = [
            MockSearchHit(id="a", content="Document A"),
            MockSearchHit(id="b", content="Document B"),
            MockSearchHit(id="c", content="Document C"),
            MockSearchHit(id="d", content="Document D"),
            MockSearchHit(id="e", content="Document E"),
        ]
        config = MockConfig()

        # Mock LLM to return reordered indices
        with patch("engram.rerank._call_llm_for_rerank", return_value="[3, 5, 1]"):
            result = rerank("test query", candidates, config, top_k=3)

        assert len(result) == 3
        assert result[0].id == "c"  # index 3 -> 0-based 2
        assert result[1].id == "e"  # index 5 -> 0-based 4
        assert result[2].id == "a"  # index 1 -> 0-based 0

    def test_rerank_fallback_on_llm_failure(self):
        from engram.rerank import rerank

        candidates = [
            MockSearchHit(id="a", content="Document A"),
            MockSearchHit(id="b", content="Document B"),
            MockSearchHit(id="c", content="Document C"),
        ]
        config = MockConfig()

        # Mock LLM failure
        with patch("engram.rerank._call_llm_for_rerank", return_value=None):
            result = rerank("test query", candidates, config, top_k=2)

        # Should return original order
        assert len(result) == 2
        assert result[0].id == "a"
        assert result[1].id == "b"

    def test_rerank_fallback_on_parse_failure(self):
        from engram.rerank import rerank

        candidates = [
            MockSearchHit(id="a", content="Document A"),
            MockSearchHit(id="b", content="Document B"),
        ]
        config = MockConfig()

        # Mock LLM returns garbage
        with patch("engram.rerank._call_llm_for_rerank", return_value="I don't know"):
            result = rerank("test query", candidates, config, top_k=2)

        assert len(result) == 2
        assert result[0].id == "a"

    def test_rerank_disabled_returns_original(self):
        from engram.rerank import rerank

        candidates = [
            MockSearchHit(id="a", content="Doc A"),
            MockSearchHit(id="b", content="Doc B"),
        ]
        config = MockConfig(rerank_enabled=False)

        result = rerank("test", candidates, config, top_k=1)
        assert len(result) == 1
        assert result[0].id == "a"

    def test_rerank_fills_remaining_from_original(self):
        from engram.rerank import rerank

        candidates = [
            MockSearchHit(id="a", content="Doc A"),
            MockSearchHit(id="b", content="Doc B"),
            MockSearchHit(id="c", content="Doc C"),
            MockSearchHit(id="d", content="Doc D"),
        ]
        config = MockConfig()

        # LLM only returns 2 indices but we want top 4
        with patch("engram.rerank._call_llm_for_rerank", return_value="[3, 1]"):
            result = rerank("test", candidates, config, top_k=4)

        assert len(result) == 4
        assert result[0].id == "c"  # from LLM
        assert result[1].id == "a"  # from LLM
        # Remaining filled from original order
        assert result[2].id == "b"
        assert result[3].id == "d"

    def test_empty_candidates(self):
        from engram.rerank import rerank

        config = MockConfig()
        result = rerank("test", [], config, top_k=5)
        assert result == []

    def test_rerank_with_llm_fn_callback(self):
        """When llm_fn is provided, it should be used instead of HTTP calls."""
        from engram.rerank import rerank

        candidates = [
            MockSearchHit(id="a", content="Document A"),
            MockSearchHit(id="b", content="Document B"),
            MockSearchHit(id="c", content="Document C"),
        ]
        config = MockConfig()

        # llm_fn that returns reranked indices
        def mock_llm(prompt, system="", **kwargs):
            return "[2, 3, 1]"

        result = rerank("test query", candidates, config, top_k=3, llm_fn=mock_llm)

        assert len(result) == 3
        assert result[0].id == "b"  # index 2 -> 0-based 1
        assert result[1].id == "c"  # index 3 -> 0-based 2
        assert result[2].id == "a"  # index 1 -> 0-based 0

    def test_rerank_with_llm_fn_fallback_on_failure(self):
        """When llm_fn fails, should fall back to original order."""
        from engram.rerank import rerank

        candidates = [
            MockSearchHit(id="a", content="Doc A"),
            MockSearchHit(id="b", content="Doc B"),
        ]
        config = MockConfig()

        def failing_llm(prompt, system="", **kwargs):
            raise RuntimeError("boom")

        result = rerank("test", candidates, config, top_k=2, llm_fn=failing_llm)

        assert len(result) == 2
        assert result[0].id == "a"
        assert result[1].id == "b"


# ===========================================================================
# Config tests
# ===========================================================================

class TestRerankConfig:
    """Test rerank config properties."""

    def test_rerank_enabled_when_llm_available(self):
        from engram.config import EngramConfig
        import tempfile, os

        tmpdir = tempfile.mkdtemp()
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("llm:\n  provider: openai\n  api_key: test\n")

        config = EngramConfig(base_dir=tmpdir)
        assert config.llm_available is True
        assert config.rerank_enabled is True

    def test_rerank_disabled_when_no_llm(self):
        from engram.config import EngramConfig
        import tempfile, os

        tmpdir = tempfile.mkdtemp()
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("llm:\n  provider: none\n")

        config = EngramConfig(base_dir=tmpdir)
        assert config.rerank_enabled is False

    def test_rerank_explicitly_disabled(self):
        from engram.config import EngramConfig
        import tempfile, os

        tmpdir = tempfile.mkdtemp()
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("llm:\n  provider: openai\n  api_key: test\n  rerank: false\n")

        config = EngramConfig(base_dir=tmpdir)
        assert config.llm_available is True
        assert config.rerank_enabled is False

    def test_rerank_candidates_default(self):
        from engram.config import EngramConfig
        import tempfile

        tmpdir = tempfile.mkdtemp()
        config = EngramConfig(base_dir=tmpdir)
        assert config.rerank_candidates == 20

    def test_rerank_candidates_custom(self):
        from engram.config import EngramConfig
        import tempfile, os

        tmpdir = tempfile.mkdtemp()
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("llm:\n  provider: openai\n  rerank_candidates: 30\n")

        config = EngramConfig(base_dir=tmpdir)
        assert config.rerank_candidates == 30
