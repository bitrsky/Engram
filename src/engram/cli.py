"""
cli.py — Command-line interface for Engram.

Uses argparse only — no click/typer dependencies.

Commands:
    engram init                          Initialize ~/.engram/
    engram remember <text> --project X   Remember something
    engram search <query> --project X    Semantic search
    engram wake-up --project X           L0+L1 session startup
    engram recall <message>              L2 contextual recall
    engram project create <id> <name>    Create project
    engram project list                  List projects
    engram project archive <id>          Archive project
    engram facts <project>               Show facts
    engram conflicts                     Show unresolved conflicts
    engram rebuild-index                 Rebuild search index
    engram decay                         Run decay engine
    engram status                        System status
"""

import argparse
import sys

from . import __version__


def main(argv=None):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="engram",
        description="Engram — Markdown-First AI Memory System",
    )
    parser.add_argument("--version", action="version", version=f"engram {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- init ---
    sub_init = subparsers.add_parser("init", help="Initialize ~/.engram/ directory")

    # --- remember ---
    sub_remember = subparsers.add_parser("remember", help="Remember something")
    sub_remember.add_argument("text", nargs="+", help="Text to remember")
    sub_remember.add_argument("--project", "-p", help="Project tag")
    sub_remember.add_argument("--topics", "-t", nargs="*", help="Topic tags")
    sub_remember.add_argument("--type", dest="memory_type", default="note",
                              choices=["note", "decision", "milestone", "problem", "preference", "emotional"])

    # --- search ---
    sub_search = subparsers.add_parser("search", help="Semantic search")
    sub_search.add_argument("query", nargs="+", help="Search query")
    sub_search.add_argument("--project", "-p", help="Filter by project")
    sub_search.add_argument("--topics", "-t", nargs="*", help="Filter by topics")
    sub_search.add_argument("-n", type=int, default=5, help="Max results")

    # --- wake-up ---
    sub_wakeup = subparsers.add_parser("wake-up", help="Session startup (L0+L1)")
    sub_wakeup.add_argument("--project", "-p", help="Project")

    # --- recall ---
    sub_recall = subparsers.add_parser("recall", help="Contextual recall (L2)")
    sub_recall.add_argument("message", nargs="+", help="Context message")
    sub_recall.add_argument("--project", "-p", help="Project")

    # --- project ---
    sub_project = subparsers.add_parser("project", help="Project management")
    project_sub = sub_project.add_subparsers(dest="project_command")

    proj_create = project_sub.add_parser("create", help="Create project")
    proj_create.add_argument("id", help="Project ID slug")
    proj_create.add_argument("name", help="Display name")
    proj_create.add_argument("--description", "-d", default="", help="Description")
    proj_create.add_argument("--aliases", "-a", nargs="*", help="Aliases")
    proj_create.add_argument("--paths", nargs="*", help="Associated paths")

    proj_list = project_sub.add_parser("list", help="List projects")
    proj_list.add_argument("--status", "-s", help="Filter by status")

    proj_archive = project_sub.add_parser("archive", help="Archive project")
    proj_archive.add_argument("id", help="Project ID")

    # --- facts ---
    sub_facts = subparsers.add_parser("facts", help="Show project facts")
    sub_facts.add_argument("project", help="Project ID")

    # --- conflicts ---
    sub_conflicts = subparsers.add_parser("conflicts", help="Show unresolved conflicts")
    sub_conflicts.add_argument("--project", "-p", help="Filter by project")

    # --- rebuild-index ---
    sub_rebuild = subparsers.add_parser("rebuild-index", help="Rebuild search index")

    # --- decay ---
    sub_decay = subparsers.add_parser("decay", help="Run decay engine")
    sub_decay.add_argument("--dry-run", action="store_true", help="Preview without changes")

    # --- status ---
    sub_status = subparsers.add_parser("status", help="System status")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch
    try:
        if args.command == "init":
            return cmd_init(args)
        elif args.command == "remember":
            return cmd_remember(args)
        elif args.command == "search":
            return cmd_search(args)
        elif args.command == "wake-up":
            return cmd_wakeup(args)
        elif args.command == "recall":
            return cmd_recall(args)
        elif args.command == "project":
            return cmd_project(args)
        elif args.command == "facts":
            return cmd_facts(args)
        elif args.command == "conflicts":
            return cmd_conflicts(args)
        elif args.command == "rebuild-index":
            return cmd_rebuild_index(args)
        elif args.command == "decay":
            return cmd_decay(args)
        elif args.command == "status":
            return cmd_status(args)
    except KeyboardInterrupt:
        print("\nAborted.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Command implementations
# ═══════════════════════════════════════════════════════════════════════════


def cmd_init(args):
    """Initialize ~/.engram/ directory."""
    from .config import EngramConfig

    config = EngramConfig()
    path = config.init()
    print(f"✅ Engram initialized at {config.base_dir}")
    print(f"   Config: {path}")
    print(f"   Identity: {config.identity_path}")
    print(f"\nEdit {config.identity_path} to set your identity.")
    return 0


def cmd_remember(args):
    """Remember something — runs the full remember pipeline."""
    from .config import EngramConfig
    from .remember import remember

    text = " ".join(args.text)
    config = EngramConfig()

    result = remember(
        content=text,
        project=args.project,
        topics=args.topics,
        memory_type=args.memory_type,
        source="cli",
        config=config,
    )

    if not result.success:
        reason = result.rejected_reason
        if reason == "low_quality":
            print("⚠️  Rejected: content did not pass quality gate.")
        elif reason.startswith("duplicate:"):
            dup_type = reason.split(":", 1)[1]
            print(f"⚠️  Duplicate detected ({dup_type}): already exists as {result.id}")
        else:
            print(f"⚠️  Not stored: {reason}")
        return 1

    print(f"✅ Remembered: {result.id}")
    print(f"   File: {result.file_path}")

    if result.facts_extracted > 0:
        print(f"   Facts extracted: {result.facts_extracted}, added: {result.facts_added}")

    if result.conflicts_detected > 0:
        print(f"   ⚠️  {result.conflicts_detected} conflict(s) detected:")
        for detail in result.conflict_details:
            old = detail.get("old", "?")
            new = detail.get("new", "?")
            resolution = detail.get("resolution", "unresolved")
            print(f"      {old}  →  {new}  [{resolution}]")

    return 0


def cmd_search(args):
    """Semantic search across memories."""
    from .config import EngramConfig
    from .index import IndexManager
    from .search import search, format_search_results

    query = " ".join(args.query)
    config = EngramConfig()

    index_manager = IndexManager(
        index_dir=config.index_dir,
        memories_dir=config.memories_dir,
        facts_dir=config.facts_dir,
        projects_dir=config.projects_dir,
    )

    try:
        results = search(
            query=query,
            index_manager=index_manager,
            project=args.project,
            topics=args.topics,
            n=args.n,
            config=config,
        )
        print(format_search_results(results))
    finally:
        index_manager.close()

    return 0


def cmd_wakeup(args):
    """Session startup — output L0 (identity) + L1 (working set)."""
    from .layers import MemoryStack

    stack = MemoryStack()
    try:
        context = stack.wake_up(project=args.project)
        if context:
            print(context)
        else:
            print("(no session context available)")
    finally:
        stack.close()

    return 0


def cmd_recall(args):
    """Contextual recall (L2) — find memories relevant to a message."""
    from .layers import MemoryStack

    message = " ".join(args.message)
    stack = MemoryStack()
    try:
        context = stack.recall(message=message, project=args.project)
        if context:
            print(context)
        else:
            print("(no relevant memories found)")
    finally:
        stack.close()

    return 0


def cmd_project(args):
    """Project management subcommands."""
    if args.project_command is None:
        # No subcommand given — print help
        print("Usage: engram project {create,list,archive} ...")
        print("\nSubcommands:")
        print("  create   Create a new project")
        print("  list     List all projects")
        print("  archive  Archive a project")
        return 1

    if args.project_command == "create":
        return _project_create(args)
    elif args.project_command == "list":
        return _project_list(args)
    elif args.project_command == "archive":
        return _project_archive(args)

    return 1


def _project_create(args):
    """Create a new project."""
    from .projects import create_project

    try:
        filepath = create_project(
            project_id=args.id,
            display_name=args.name,
            description=args.description,
            aliases=args.aliases,
            associated_paths=args.paths,
        )
    except FileExistsError:
        print(f"❌ Project '{args.id}' already exists.", file=sys.stderr)
        return 1

    print(f"✅ Project created: {args.id}")
    print(f"   Name: {args.name}")
    print(f"   File: {filepath}")

    if args.description:
        print(f"   Description: {args.description}")
    if args.aliases:
        print(f"   Aliases: {', '.join(args.aliases)}")
    if args.paths:
        print(f"   Paths: {', '.join(args.paths)}")

    return 0


def _project_list(args):
    """List all projects."""
    from .projects import list_projects

    projects = list_projects(status=args.status)

    if not projects:
        status_msg = f" with status '{args.status}'" if args.status else ""
        print(f"No projects found{status_msg}.")
        return 0

    status_filter = f" (status: {args.status})" if args.status else ""
    print(f"📂 Projects{status_filter}:\n")

    for proj in projects:
        pid = proj.get("id", "?")
        display = proj.get("display_name", pid)
        status = proj.get("status", "?")
        desc = proj.get("description", "")
        last_active = proj.get("last_active")
        last_active_str = str(last_active)[:10] if last_active else "—"

        # Status indicator
        status_icon = {"active": "🟢", "paused": "🟡", "archived": "⚫"}.get(status, "⚪")

        print(f"  {status_icon} {display} ({pid})")
        print(f"    Status: {status} | Last active: {last_active_str}")
        if desc:
            print(f"    {desc}")
        print()

    print(f"Total: {len(projects)} project(s)")
    return 0


def _project_archive(args):
    """Archive a project."""
    from .projects import archive_project

    try:
        archive_project(args.id)
    except FileNotFoundError:
        print(f"❌ Project '{args.id}' not found.", file=sys.stderr)
        return 1

    print(f"✅ Project '{args.id}' archived.")
    return 0


def cmd_facts(args):
    """Show facts for a project — reads the facts file directly."""
    from .config import EngramConfig
    from .facts import parse_facts_file

    config = EngramConfig()
    facts_path = config.facts_dir / f"{args.project}.md"

    if not facts_path.exists():
        print(f"No facts file for project '{args.project}'.")
        print(f"  Expected: {facts_path}")
        return 1

    data = parse_facts_file(args.project, facts_dir=config.facts_dir)
    current = data["current"]
    expired = data["expired"]
    conflicts = data["conflicts"]

    print(f"📌 Facts: {args.project}\n")

    # Current facts
    print(f"## Current ({len(current)})\n")
    if current:
        for fact in current:
            since_str = f" (since {fact.since})" if fact.since else ""
            conf_str = f" [{fact.confidence:.0%}]" if fact.confidence < 1.0 else ""
            print(f"  • {fact.subject} → {fact.predicate} → {fact.object}{since_str}{conf_str}")
            if fact.source:
                print(f"    source: {fact.source}")
    else:
        print("  (none)")
    print()

    # Expired facts
    print(f"## Expired ({len(expired)})\n")
    if expired:
        for fact in expired:
            range_str = ""
            if fact.since and fact.expired_at:
                range_str = f" ({fact.since} to {fact.expired_at[:7]})"
            elif fact.expired_at:
                range_str = f" (expired {fact.expired_at[:7]})"
            print(f"  • {fact.subject} → {fact.predicate} → {fact.object}{range_str}")
            if fact.reason:
                print(f"    reason: {fact.reason}")
    else:
        print("  (none)")
    print()

    # Conflicts
    unresolved = [c for c in conflicts if c.get("status") == "unresolved"]
    resolved = [c for c in conflicts if c.get("status") != "unresolved"]
    print(f"## Conflicts ({len(unresolved)} unresolved, {len(resolved)} resolved)\n")
    if conflicts:
        for c in conflicts:
            status_icon = "⚠️" if c.get("status") == "unresolved" else "✅"
            print(f"  {status_icon} {c.get('description', '?')}")
            print(f"    status: {c.get('status', '?')} | detected: {c.get('detected', '?')}")
    else:
        print("  (none)")

    return 0


def cmd_conflicts(args):
    """Show unresolved conflicts across projects."""
    from .config import EngramConfig
    from .facts import get_unresolved_conflicts

    config = EngramConfig()
    conflicts = get_unresolved_conflicts(
        project=args.project,
        facts_dir=config.facts_dir,
    )

    if not conflicts:
        scope = f" in project '{args.project}'" if args.project else ""
        print(f"✅ No unresolved conflicts{scope}.")
        return 0

    scope = f" in project '{args.project}'" if args.project else " (all projects)"
    print(f"⚠️  Unresolved conflicts{scope}:\n")

    for i, c in enumerate(conflicts, 1):
        project_tag = f"[{c.get('project', '?')}] " if not args.project else ""
        print(f"  {i}. {project_tag}{c.get('description', '?')}")
        print(f"     Detected: {c.get('detected', '?')}")
        print(f"     Sources: {c.get('fact_a_source', '?')} vs {c.get('fact_b_source', '?')}")
        print()

    print(f"Total: {len(conflicts)} unresolved conflict(s)")
    return 0


def cmd_rebuild_index(args):
    """Rebuild the search index from Markdown files."""
    from .config import EngramConfig
    from .index import IndexManager

    config = EngramConfig()

    if not config.memories_dir.is_dir():
        print(f"❌ Memories directory not found: {config.memories_dir}", file=sys.stderr)
        print("Run 'engram init' first.")
        return 1

    print(f"Rebuilding index from {config.memories_dir} ...")

    index_manager = IndexManager(
        index_dir=config.index_dir,
        memories_dir=config.memories_dir,
        facts_dir=config.facts_dir,
        projects_dir=config.projects_dir,
    )

    try:
        count = index_manager.rebuild(memories_dir=config.memories_dir)
    finally:
        index_manager.close()

    print(f"✅ Index rebuilt: {count} memories indexed.")
    stats_line = f"   ChromaDB: {config.index_dir / 'vectors.chroma'}"
    print(stats_line)
    print(f"   SQLite:   {config.index_dir / 'meta.sqlite3'}")
    return 0


def cmd_decay(args):
    """Run the decay engine."""
    from .decay import run_decay

    result = run_decay(dry_run=args.dry_run)

    if args.dry_run:
        print("🧪 Decay engine (dry run):\n")
    else:
        print("🔄 Decay engine results:\n")

    print(f"  Total memories scanned: {result.total_scanned}")
    print(f"  Decayed (importance reduced): {result.decayed}")
    print(f"  Promoted (access boosted):    {result.promoted}")
    print(f"  Unchanged:                    {result.unchanged}")
    print(f"  Errors:                       {result.errors}")

    if args.dry_run:
        print("\n  (no changes applied — dry run)")

    return 0


def cmd_status(args):
    """Show system status."""
    from .config import EngramConfig
    from .layers import MemoryStack

    config = EngramConfig()
    stack = MemoryStack(config=config)

    try:
        status = stack.get_status()
    finally:
        stack.close()

    print("📊 Engram Status\n")

    # Base dir
    exists_icon = "✅" if config.base_dir.exists() else "❌"
    print(f"  {exists_icon} Base dir:  {config.base_dir}")

    # Identity
    id_icon = "✅" if config.identity_path.exists() else "⚠️"
    print(f"  {id_icon} Identity:  {config.identity_path}")
    print()

    # Memories
    total_memories = status.get("total_memories", 0)
    print(f"  Memories:  {total_memories}")

    # Projects
    total_projects = status.get("total_projects", 0)
    print(f"  Projects:  {total_projects}")

    # Active project
    active = status.get("active_project")
    print(f"  Active:    {active or '(none)'}")
    print()

    # Index stats
    idx = status.get("index_stats", {})
    chroma_count = idx.get("chroma_count", 0)
    sqlite_count = idx.get("sqlite_count", 0)
    last_rebuild = idx.get("last_rebuild") or "(never)"
    indexed_projects = idx.get("projects", [])

    print("  Index:")
    print(f"    ChromaDB vectors: {chroma_count}")
    print(f"    SQLite rows:      {sqlite_count}")
    print(f"    Last rebuild:     {last_rebuild}")
    if indexed_projects:
        print(f"    Projects indexed: {', '.join(indexed_projects)}")
    print()

    # Conflicts
    n_conflicts = status.get("unresolved_conflicts", 0)
    if n_conflicts > 0:
        print(f"  ⚠️  Unresolved conflicts: {n_conflicts}")
    else:
        print("  ✅ No unresolved conflicts")

    # LLM config
    print()
    print(f"  LLM: via think_fn callback (host agent provides)")

    return 0
