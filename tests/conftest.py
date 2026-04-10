"""Shared fixtures for Engram tests."""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def engram_dir(tmp_path):
    """
    Provide an isolated ~/.engram/ directory for testing.

    Creates the full directory structure and returns the base path.
    Tests can use this with EngramConfig(base_dir=engram_dir).
    """
    base = tmp_path / ".engram"
    base.mkdir()
    (base / "memories").mkdir()
    (base / "projects").mkdir()
    (base / "facts").mkdir()
    (base / ".index").mkdir()

    # Create minimal config
    config_path = base / "config.yaml"
    config_path.write_text(
        "llm:\n  provider: none\n",
        encoding="utf-8",
    )

    # Create minimal identity
    identity_path = base / "identity.md"
    identity_path.write_text(
        "---\nname: Test User\n---\n\nI am a test user.\n",
        encoding="utf-8",
    )

    return base


@pytest.fixture
def config(engram_dir):
    """Provide an EngramConfig pointing to the isolated test directory."""
    from engram.config import EngramConfig
    return EngramConfig(base_dir=str(engram_dir))


@pytest.fixture
def index_manager(config):
    """Provide an IndexManager using the test directory."""
    from engram.index import IndexManager
    mgr = IndexManager(
        index_dir=config.index_dir,
        memories_dir=config.memories_dir,
    )
    yield mgr
    mgr.close()
