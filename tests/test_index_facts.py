"""Tests for facts and projects indexing in IndexManager."""

import json
import tempfile
from pathlib import Path

import pytest

from engram.index import IndexManager


@pytest.fixture()
def tmp_dirs(tmp_path):
    """Create temp directories mimicking engram structure."""
    memories_dir = tmp_path / "memories"
    facts_dir = tmp_path / "facts"
    projects_dir = tmp_path / "projects"
    index_dir = tmp_path / ".index"
    memories_dir.mkdir()
    facts_dir.mkdir()
    projects_dir.mkdir()
    return {
        "index_dir": index_dir,
        "memories_dir": memories_dir,
        "facts_dir": facts_dir,
        "projects_dir": projects_dir,
    }


@pytest.fixture()
def manager(tmp_dirs):
    mgr = IndexManager(**tmp_dirs)
    yield mgr
    mgr.close()


# ---------------------------------------------------------------------------
# Facts indexing
# ---------------------------------------------------------------------------


def _write_facts_file(facts_dir: Path, project: str, facts: list[tuple]):
    """Write a minimal facts file.

    facts: list of (subject, predicate, object, since) tuples.
    """
    lines = [f"# Facts: {project}\n\n## Current\n"]
    for subj, pred, obj, since in facts:
        lines.append(f"- {subj} | {pred} | {obj} | since: {since}\n")
    lines.append("\n## Expired\n\n## Conflicts\n")

    path = facts_dir / f"{project}.md"
    path.write_text("".join(lines), encoding="utf-8")
    return path


class TestIndexFactsFile:
    def test_basic_indexing(self, manager, tmp_dirs):
        """Facts should be indexed and searchable."""
        _write_facts_file(
            tmp_dirs["facts_dir"],
            "myproj",
            [
                ("Alice", "uses_database", "PostgreSQL", "2025-01"),
                ("Bob", "prefers_language", "Python", "2025-03"),
            ],
        )
        count = manager.index_facts_file(tmp_dirs["facts_dir"] / "myproj.md")
        assert count == 2

        # Search for a fact
        hits = manager.vector_search("what database does Alice use", n=5)
        fact_hits = [h for h in hits if h.memory_type == "fact"]
        assert len(fact_hits) >= 1
        assert "PostgreSQL" in fact_hits[0].content

    def test_fact_metadata(self, manager, tmp_dirs):
        """Fact entries should have correct metadata."""
        _write_facts_file(
            tmp_dirs["facts_dir"],
            "proj",
            [("System", "uses_framework", "Django", "2025-06")],
        )
        manager.index_facts_file(tmp_dirs["facts_dir"] / "proj.md")

        hits = manager.vector_search("Django framework", n=5)
        fact_hits = [h for h in hits if h.memory_type == "fact"]
        assert len(fact_hits) >= 1
        h = fact_hits[0]
        assert h.project == "proj"
        assert h.importance == 5.0
        assert h.id.startswith("fact_proj_")

    def test_reindex_replaces_old(self, manager, tmp_dirs):
        """Re-indexing a facts file should replace old entries."""
        path = _write_facts_file(
            tmp_dirs["facts_dir"],
            "proj",
            [("Alice", "uses_database", "MySQL", "2025-01")],
        )
        manager.index_facts_file(path)

        # Update the fact
        _write_facts_file(
            tmp_dirs["facts_dir"],
            "proj",
            [("Alice", "uses_database", "PostgreSQL", "2025-06")],
        )
        count = manager.index_facts_file(path)
        assert count == 1

        hits = manager.vector_search("what database does Alice use", n=10)
        fact_hits = [h for h in hits if h.memory_type == "fact"]
        # Should only have the new fact, not the old one
        contents = [h.content for h in fact_hits]
        assert any("PostgreSQL" in c for c in contents)
        # MySQL should NOT be in any fact hit
        assert not any("MySQL" in c for c in contents)

    def test_nonexistent_file(self, manager, tmp_dirs):
        """Indexing a nonexistent file returns 0."""
        count = manager.index_facts_file(tmp_dirs["facts_dir"] / "nope.md")
        assert count == 0


# ---------------------------------------------------------------------------
# Projects indexing
# ---------------------------------------------------------------------------


def _write_project_file(projects_dir: Path, project_id: str, **kwargs):
    """Write a minimal project file."""
    import yaml

    meta = {
        "id": project_id,
        "display_name": kwargs.get("display_name", project_id),
        "status": kwargs.get("status", "active"),
        "description": kwargs.get("description", ""),
        "created": kwargs.get("created", "2025-01-01T00:00:00"),
        "aliases": kwargs.get("aliases", []),
        "tags": kwargs.get("tags", []),
    }
    yaml_text = yaml.dump(meta, default_flow_style=False, allow_unicode=True)
    body = kwargs.get("description", "")
    content = f"---\n{yaml_text}---\n\n{body}\n"

    path = projects_dir / f"{project_id}.md"
    path.write_text(content, encoding="utf-8")
    return path


class TestIndexProjectFile:
    def test_basic_indexing(self, manager, tmp_dirs):
        """Projects should be indexed and searchable."""
        _write_project_file(
            tmp_dirs["projects_dir"],
            "saas-app",
            display_name="My SaaS App",
            description="A B2B platform for invoicing",
            tags=["web", "billing"],
        )
        pid = manager.index_project_file(tmp_dirs["projects_dir"] / "saas-app.md")
        assert pid == "saas-app"

        hits = manager.vector_search("invoicing platform", n=5)
        proj_hits = [h for h in hits if h.memory_type == "project"]
        assert len(proj_hits) >= 1
        assert proj_hits[0].project == "saas-app"

    def test_project_metadata(self, manager, tmp_dirs):
        """Project entries should have correct metadata."""
        _write_project_file(
            tmp_dirs["projects_dir"],
            "ml-pipeline",
            display_name="ML Pipeline",
            tags=["ml", "data"],
        )
        manager.index_project_file(tmp_dirs["projects_dir"] / "ml-pipeline.md")

        hits = manager.vector_search("machine learning pipeline", n=5)
        proj_hits = [h for h in hits if h.memory_type == "project"]
        assert len(proj_hits) >= 1
        h = proj_hits[0]
        assert h.id == "project_ml-pipeline"
        assert h.importance == 4.0


# ---------------------------------------------------------------------------
# Rebuild / incremental with all three types
# ---------------------------------------------------------------------------


class TestRebuildWithFactsAndProjects:
    def test_rebuild_indexes_all(self, manager, tmp_dirs):
        """rebuild() should index memories, facts, and projects."""
        # Write a memory
        mem_path = tmp_dirs["memories_dir"] / "2025-01-01_chat.md"
        mem_path.write_text(
            "---\nid: session_1\nproject: proj\ntopics: []\nmemory_type: conversation\nimportance: 3.0\ncreated: 2025-01-01\n---\nHello world\n",
            encoding="utf-8",
        )

        # Write facts
        _write_facts_file(
            tmp_dirs["facts_dir"],
            "proj",
            [("Alice", "likes", "chess", "2025-01")],
        )

        # Write project
        _write_project_file(
            tmp_dirs["projects_dir"],
            "proj",
            display_name="My Project",
        )

        count = manager.rebuild()
        # 1 memory + 1 fact + 1 project = 3
        assert count == 3

        # All types should be searchable
        hits = manager.vector_search("chess", n=5)
        types = {h.memory_type for h in hits}
        assert "fact" in types

    def test_incremental_picks_up_new_facts(self, manager, tmp_dirs):
        """incremental_update() should index new/modified facts files."""
        # First build with just a memory
        mem_path = tmp_dirs["memories_dir"] / "2025-01-01_chat.md"
        mem_path.write_text(
            "---\nid: session_1\nproject: proj\ntopics: []\nmemory_type: conversation\nimportance: 3.0\ncreated: 2025-01-01\n---\nHello world\n",
            encoding="utf-8",
        )
        manager.rebuild()

        # Now add a facts file
        import time
        time.sleep(0.1)  # ensure mtime is newer
        _write_facts_file(
            tmp_dirs["facts_dir"],
            "proj",
            [("Bob", "uses", "Vim", "2025-06")],
        )

        updated = manager.incremental_update()
        assert updated >= 1

        hits = manager.vector_search("Vim editor", n=5)
        fact_hits = [h for h in hits if h.memory_type == "fact"]
        assert len(fact_hits) >= 1
