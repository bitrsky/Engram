"""
test_rerank.py -- Unit tests for LLM reranking module.

Tests the callback-based reranking and fallback behavior.
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
    rerank_enabled: bool = True
    rerank_candidates: int = 20


# ===========================================================================
# rerank() tests
# ===========================================================================

class TestRerank:
    """Test the rerank() function with think_fn callbacks."""

    def test_rerank_with_think_fn_reorders(self):
        """When think_fn is provided, it should reorder candidates."""
        from engram.rerank import rerank

        candidates = [
            MockSearchHit(id="a", content="Document A"),
            MockSearchHit(id="b", content="Document B"),
            MockSearchHit(id="c", content="Document C"),
            MockSearchHit(id="d", content="Document D"),
            MockSearchHit(id="e", content="Document E"),
        ]
        config = MockConfig()

        def mock_llm(prompt, system="", **kwargs):
            return "[3, 5, 1]"

        result = rerank("test query", candidates, config, top_k=3, think_fn=mock_llm)

        assert len(result) == 3
        assert result[0].id == "c"  # index 3 -> 0-based 2
        assert result[1].id == "e"  # index 5 -> 0-based 4
        assert result[2].id == "a"  # index 1 -> 0-based 0

    def test_rerank_with_think_fn_fallback_on_failure(self):
        """When think_fn fails, should fall back to original order."""
        from engram.rerank import rerank

        candidates = [
            MockSearchHit(id="a", content="Doc A"),
            MockSearchHit(id="b", content="Doc B"),
        ]
        config = MockConfig()

        def failing_llm(prompt, system="", **kwargs):
            raise RuntimeError("boom")

        result = rerank("test", candidates, config, top_k=2, think_fn=failing_llm)

        assert len(result) == 2
        assert result[0].id == "a"
        assert result[1].id == "b"

    def test_rerank_no_think_fn_returns_original_order(self):
        """Without think_fn, returns original vector order."""
        from engram.rerank import rerank

        candidates = [
            MockSearchHit(id="a", content="Doc A"),
            MockSearchHit(id="b", content="Doc B"),
            MockSearchHit(id="c", content="Doc C"),
        ]
        config = MockConfig()

        result = rerank("test", candidates, config, top_k=2)

        assert len(result) == 2
        assert result[0].id == "a"
        assert result[1].id == "b"

    def test_rerank_disabled_no_think_fn_returns_original(self):
        """When rerank disabled and no think_fn, returns original order."""
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
        """When LLM returns fewer than top_k, fill from original order."""
        from engram.rerank import rerank

        candidates = [
            MockSearchHit(id="a", content="Doc A"),
            MockSearchHit(id="b", content="Doc B"),
            MockSearchHit(id="c", content="Doc C"),
            MockSearchHit(id="d", content="Doc D"),
        ]
        config = MockConfig()

        def mock_llm(prompt, system="", **kwargs):
            return "[3, 1]"

        result = rerank("test", candidates, config, top_k=4, think_fn=mock_llm)

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

    def test_rerank_think_fn_returns_garbage(self):
        """When think_fn returns unparseable text, fall back to original order."""
        from engram.rerank import rerank

        candidates = [
            MockSearchHit(id="a", content="Doc A"),
            MockSearchHit(id="b", content="Doc B"),
        ]
        config = MockConfig()

        def garbage_llm(prompt, system="", **kwargs):
            return "I don't know how to rank these"

        result = rerank("test", candidates, config, top_k=2, think_fn=garbage_llm)

        # llm.rerank_with_llm will parse "don't" -> no valid indices -> fill from original
        assert len(result) == 2


# ===========================================================================
# Config tests
# ===========================================================================

class TestRerankConfig:
    """Test rerank config properties."""

    def test_rerank_enabled_default(self):
        """Rerank should be enabled by default."""
        from engram.config import EngramConfig
        import tempfile

        tmpdir = tempfile.mkdtemp()
        config = EngramConfig(base_dir=tmpdir)
        assert config.rerank_enabled is True

    def test_rerank_explicitly_disabled(self):
        from engram.config import EngramConfig
        import tempfile, os

        tmpdir = tempfile.mkdtemp()
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as f:
            f.write("llm:\n  rerank: false\n")

        config = EngramConfig(base_dir=tmpdir)
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
            f.write("llm:\n  rerank_candidates: 30\n")

        config = EngramConfig(base_dir=tmpdir)
        assert config.rerank_candidates == 30
