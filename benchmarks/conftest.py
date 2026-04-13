"""
conftest.py -- Shared fixtures for Engram benchmarks.

Provides an isolated Engram instance with synthetic data pre-ingested.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# Benchmark data directory (relative to this file)
BENCHMARKS_DIR = Path(__file__).parent
DATASETS_DIR = BENCHMARKS_DIR / "datasets"


# ===========================================================================
# Data loading
# ===========================================================================


def load_conversations() -> List[dict]:
    """Load synthetic conversations from JSON."""
    path = DATASETS_DIR / "synthetic_conversations.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions() -> List[dict]:
    """Load synthetic questions from JSON."""
    path = DATASETS_DIR / "synthetic_questions.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_questions_by_split(split: str) -> List[dict]:
    """Get questions filtered by split ('dev' or 'test')."""
    return [q for q in load_questions() if q["split"] == split]


def get_questions_by_category(category: str) -> List[dict]:
    """Get questions filtered by category."""
    return [q for q in load_questions() if q["category"] == category]


def conversation_to_text(session: dict) -> str:
    """Convert a conversation session to a flat text string for ingestion."""
    lines = []
    for turn in session.get("turns", []):
        role = turn.get("role", "user")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="session")
def engram_bench_dir(tmp_path_factory):
    """
    Create an isolated Engram directory for benchmarks.

    Session-scoped: created once and reused across all benchmark tests.
    """
    base = tmp_path_factory.mktemp("engram_bench")
    engram_dir = base / ".engram"
    engram_dir.mkdir()
    (engram_dir / "memories").mkdir()
    (engram_dir / "projects").mkdir()
    (engram_dir / "facts").mkdir()
    (engram_dir / ".index").mkdir()

    # Create minimal config (no LLM -- heuristic mode only)
    config_path = engram_dir / "config.yaml"
    config_path.write_text(
        "llm:\n  provider: none\n",
        encoding="utf-8",
    )

    # Create identity
    identity_path = engram_dir / "identity.md"
    identity_path.write_text(
        "---\nname: Benchmark User\n---\n\nI am a benchmark test user.\n",
        encoding="utf-8",
    )

    return engram_dir


@pytest.fixture(scope="session")
def bench_config(engram_bench_dir):
    """Provide an EngramConfig for benchmarks."""
    from engram.config import EngramConfig
    return EngramConfig(base_dir=str(engram_bench_dir))


@pytest.fixture(scope="session")
def bench_index(bench_config):
    """
    Provide an IndexManager with all synthetic conversations ingested.

    Session-scoped: ingests once, reused by all retrieval tests.
    """
    from engram.index import IndexManager
    from engram.store import write_memory

    mgr = IndexManager(
        index_dir=bench_config.index_dir,
        memories_dir=bench_config.memories_dir,
    )

    conversations = load_conversations()

    for session in conversations:
        text = conversation_to_text(session)
        session_id = session["session_id"]
        project = session.get("project", "saas-app")
        topics = session.get("topics", [])
        timestamp = session.get("timestamp", "")

        # Write memory file
        filepath = write_memory(
            content=text,
            project=project,
            topics=topics,
            memory_type="conversation",
            source="benchmark",
            importance=3.0,
            memories_dir=bench_config.memories_dir,
            memory_id=session_id,
        )

        # Index it
        mgr.index_memory(filepath)

    yield mgr
    mgr.close()


@pytest.fixture(scope="session")
def all_questions():
    """Load all synthetic questions."""
    return load_questions()


@pytest.fixture(scope="session")
def test_questions():
    """Load test-split questions only."""
    return get_questions_by_split("test")


@pytest.fixture(scope="session")
def dev_questions():
    """Load dev-split questions only."""
    return get_questions_by_split("dev")


@pytest.fixture(scope="session")
def conversations():
    """Load all synthetic conversations."""
    return load_conversations()
