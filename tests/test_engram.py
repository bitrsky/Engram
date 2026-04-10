"""Smoke tests to validate the package is importable and functional."""

import pytest
from pathlib import Path


class TestImports:
    """Verify all modules are importable."""

    def test_import_engram(self):
        import engram
        assert hasattr(engram, "__version__")
        assert engram.__version__ == "0.1.0"

    def test_import_config(self):
        from engram.config import EngramConfig
        assert EngramConfig is not None

    def test_import_store(self):
        from engram.store import (
            parse_frontmatter, write_memory, read_memory,
            list_memories, update_frontmatter, slugify,
            generate_memory_id,
        )

    def test_import_quality(self):
        from engram.quality import quality_gate

    def test_import_conflicts(self):
        from engram.conflicts import (
            Fact, Conflict, check_conflict,
            classify_conflict, resolve_conflict,
            format_conflict_report,
        )

    def test_import_dedup(self):
        from engram.dedup import DedupResult, check_duplicate, normalize_for_hash

    def test_import_extract(self):
        from engram.extract import (
            FactCandidate, extract_facts,
            extract_facts_heuristic,
        )

    def test_import_facts(self):
        from engram.facts import (
            parse_facts_file, write_facts_file, add_fact,
            expire_fact, get_active_facts, get_facts_for_entity,
            get_unresolved_conflicts,
        )

    def test_import_index(self):
        from engram.index import IndexManager, SearchHit

    def test_import_projects(self):
        from engram.projects import (
            create_project, get_project, list_projects,
            update_project, archive_project, resolve_project,
        )

    def test_import_search(self):
        from engram.search import search, format_search_results

    def test_import_remember(self):
        from engram.remember import remember, remember_batch, RememberResult

    def test_import_layers(self):
        from engram.layers import MemoryStack

    def test_import_decay(self):
        from engram.decay import run_decay, DecayResult

    def test_import_cli(self):
        from engram.cli import main

    def test_import_ingest(self):
        from engram.ingest import (
            ingest_file, ingest_directory, ingest_text,
            chunk_markdown, chunk_text, chunk_code,
        )


class TestConfig:
    """Test configuration management."""

    def test_init_creates_dirs(self, config, engram_dir):
        config.init()
        assert (engram_dir / "memories").is_dir()
        assert (engram_dir / "projects").is_dir()
        assert (engram_dir / "facts").is_dir()
        assert (engram_dir / ".index").is_dir()

    def test_directory_properties(self, config, engram_dir):
        assert config.base_dir == engram_dir
        assert config.memories_dir == engram_dir / "memories"
        assert config.projects_dir == engram_dir / "projects"
        assert config.facts_dir == engram_dir / "facts"
        assert config.index_dir == engram_dir / ".index"

    def test_llm_default_none(self, config):
        assert config.llm_provider == "none"
        assert config.llm_available is False


class TestStore:
    """Test Markdown store operations."""

    def test_slugify(self):
        from engram.store import slugify
        assert slugify("Auth Provider Decision") == "auth-provider-decision"
        assert slugify("A" * 100) != ""
        assert len(slugify("A" * 100)) <= 40

    def test_generate_memory_id(self):
        from engram.store import generate_memory_id
        mid = generate_memory_id("test content")
        assert mid.startswith("mem_")
        assert len(mid) > 10

    def test_write_and_read_memory(self, config):
        from engram.store import write_memory, read_memory

        filepath = write_memory(
            content="We decided to use Clerk for auth.",
            project="test-project",
            topics=["auth"],
            memory_type="decision",
            memories_dir=config.memories_dir,
        )

        assert filepath.exists()
        assert filepath.suffix == ".md"

        data = read_memory(filepath)
        assert data["content"] == "We decided to use Clerk for auth."
        assert data["project"] == "test-project"
        assert data["topics"] == ["auth"]
        assert data["memory_type"] == "decision"

    def test_parse_frontmatter(self, tmp_path):
        from engram.store import parse_frontmatter

        md = tmp_path / "test.md"
        md.write_text(
            "---\nid: test\nproject: foo\n---\n\nBody here.\n",
            encoding="utf-8",
        )
        fm, body = parse_frontmatter(md)
        assert fm["id"] == "test"
        assert fm["project"] == "foo"
        assert "Body here." in body

    def test_update_frontmatter(self, tmp_path):
        from engram.store import parse_frontmatter, update_frontmatter

        md = tmp_path / "test.md"
        md.write_text(
            "---\nid: test\nimportance: 3.0\n---\n\nBody.\n",
            encoding="utf-8",
        )
        update_frontmatter(md, {"importance": 5.0})
        fm, body = parse_frontmatter(md)
        assert fm["importance"] == 5.0
        assert "Body." in body


class TestQuality:
    """Test quality gate."""

    def test_reject_empty(self):
        from engram.quality import quality_gate
        ok, imp = quality_gate("")
        assert ok is False

    def test_reject_noise(self):
        from engram.quality import quality_gate
        ok, imp = quality_gate("ok")
        assert ok is False

    def test_accept_decision(self):
        from engram.quality import quality_gate
        ok, imp = quality_gate("We decided to use PostgreSQL for the database.")
        assert ok is True
        assert imp > 3.0  # decision marker boost


class TestConflicts:
    """Test conflict detection."""

    def test_no_conflict_non_exclusive(self):
        from engram.conflicts import Fact, check_conflict
        existing = [Fact(subject="Max", predicate="loves", object="chess")]
        new_fact = Fact(subject="Max", predicate="loves", object="swimming")
        result = check_conflict(new_fact, existing)
        assert result is None  # loves is non-exclusive

    def test_conflict_exclusive(self):
        from engram.conflicts import Fact, check_conflict
        existing = [Fact(subject="app", predicate="database", object="Postgres")]
        new_fact = Fact(subject="app", predicate="database", object="MongoDB")
        result = check_conflict(new_fact, existing)
        assert result is not None
        assert result.old_fact.object == "Postgres"
        assert result.new_fact.object == "MongoDB"

    def test_temporal_succession(self):
        from engram.conflicts import Fact, classify_conflict
        old = Fact(subject="app", predicate="database", object="Postgres", since="2025-01")
        new = Fact(subject="app", predicate="database", object="MongoDB", since="2026-03")
        ctype = classify_conflict(old, new)
        assert ctype == "temporal_succession"

    def test_implicit_supersede(self):
        from engram.conflicts import Fact, classify_conflict
        old = Fact(subject="app", predicate="database", object="Postgres")
        new = Fact(subject="app", predicate="database", object="MongoDB")
        ctype = classify_conflict(old, new, source_text="We migrated to MongoDB")
        assert ctype == "implicit_supersede"


class TestProjects:
    """Test project management."""

    def test_create_and_get(self, config):
        from engram.projects import create_project, get_project

        path = create_project(
            "test-proj",
            "Test Project",
            description="A test",
            projects_dir=config.projects_dir,
        )
        assert path.exists()

        proj = get_project("test-proj", projects_dir=config.projects_dir)
        assert proj is not None
        assert proj["id"] == "test-proj"
        assert proj["display_name"] == "Test Project"
        assert proj["status"] == "active"

    def test_list_projects(self, config):
        from engram.projects import create_project, list_projects

        create_project("proj-a", "Project A", projects_dir=config.projects_dir)
        create_project("proj-b", "Project B", projects_dir=config.projects_dir)

        projects = list_projects(projects_dir=config.projects_dir)
        assert len(projects) == 2

    def test_resolve_explicit(self, config):
        from engram.projects import create_project, resolve_project

        create_project("my-app", "My App", projects_dir=config.projects_dir)
        result = resolve_project(explicit="my-app", projects_dir=config.projects_dir)
        assert result == "my-app"

    def test_resolve_alias(self, config):
        from engram.projects import create_project, resolve_project

        create_project(
            "my-app", "My App",
            aliases=["the app", "saas"],
            projects_dir=config.projects_dir,
        )
        result = resolve_project(
            message="let's discuss the app",
            projects_dir=config.projects_dir,
        )
        assert result == "my-app"


class TestCLI:
    """Test CLI commands."""

    def test_help(self):
        from engram.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_version(self):
        from engram.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
