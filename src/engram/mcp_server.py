"""
mcp_server.py — MCP (Model Context Protocol) Server for Engram.

Exposes 12 tools for AI assistants:

Read tools:
  engram_status        — System status and active project info
  engram_search        — Semantic search across memories
  engram_recall        — Contextual recall (L2)
  engram_facts         — Get facts for a project/entity
  engram_timeline      — Recent memories timeline
  engram_conflicts     — List unresolved conflicts
  engram_list_projects — List all projects
  engram_wake_up       — Session startup (L0+L1)

Write tools:
  engram_remember      — Store a new memory
  engram_learn_fact    — Manually add a fact
  engram_forget_fact   — Expire a fact
  engram_resolve_conflict — Resolve a conflict
"""

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from .config import EngramConfig
from .layers import MemoryStack
from .remember import remember
from .search import search as do_search, format_search_results
from .facts import (
    add_fact, expire_fact, get_active_facts, get_facts_for_entity,
    get_unresolved_conflicts, resolve_conflict_manual,
)
from .projects import list_projects, resolve_project

logger = logging.getLogger(__name__)

# ── MCP Server ──────────────────────────────────────────────────────────────

mcp = FastMCP("engram")

# ── Global instances (initialized lazily) ───────────────────────────────────

_config: EngramConfig | None = None
_stack: MemoryStack | None = None


def _get_config() -> EngramConfig:
    global _config
    if _config is None:
        _config = EngramConfig()
    return _config


def _get_stack() -> MemoryStack:
    global _stack
    if _stack is None:
        _stack = MemoryStack(_get_config())
    return _stack


# ═══════════════════════════════════════════════════════════════════════════
# READ TOOLS (8)
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def engram_status() -> str:
    """Get Engram memory system status — active project, total memories,
    project count, index stats, and unresolved conflict count.
    Call this to understand the current state of the memory system."""
    try:
        stack = _get_stack()
        status = stack.get_status()
        return json.dumps(status, indent=2, default=str)
    except Exception as e:
        logger.exception("engram_status failed")
        return json.dumps({"error": str(e)})


@mcp.tool()
def engram_search(
    query: str,
    project: str = None,
    topics: str = None,
    n: int = 5,
) -> str:
    """Semantic search across all memories. Returns enriched results with
    related facts and conflict flags.

    Args:
        query: What to search for (natural language).
        project: Filter to a specific project (optional).
        topics: Comma-separated topic filter, e.g. "auth,backend" (optional).
        n: Maximum number of results (default 5).
    """
    try:
        stack = _get_stack()
        config = _get_config()

        topic_list = None
        if topics:
            topic_list = [t.strip() for t in topics.split(",") if t.strip()]

        results = do_search(
            query=query,
            index_manager=stack.index,
            project=project,
            topics=topic_list,
            n=n,
            config=config,
        )

        return format_search_results(results)
    except Exception as e:
        logger.exception("engram_search failed")
        return f"Search error: {e}"


@mcp.tool()
def engram_recall(
    message: str,
    project: str = None,
    topic: str = None,
    cwd: str = None,
    n: int = 3,
) -> str:
    """Contextual recall (L2) — automatically find memories relevant to
    the current message. Uses project resolution and semantic search.

    Args:
        message: The user's message or current context.
        project: Explicit project override (optional).
        topic: Topic filter (optional).
        cwd: Current working directory for project auto-detection (optional).
        n: Maximum number of results (default 3).
    """
    try:
        stack = _get_stack()
        result = stack.recall(
            message=message,
            project=project,
            topic=topic,
            cwd=cwd,
            n=n,
        )
        return result if result else "No relevant memories found."
    except Exception as e:
        logger.exception("engram_recall failed")
        return f"Recall error: {e}"


@mcp.tool()
def engram_facts(
    project: str = None,
    entity: str = None,
) -> str:
    """Get active facts for a project or entity. Returns structured
    fact triples (subject → predicate → object) with confidence and dates.

    Provide either project or entity (or both):
    - project only: all active facts in that project
    - entity only: facts about that entity across all projects
    - both: facts about that entity within the specified project

    Args:
        project: Project ID to get facts for (optional).
        entity: Entity name to look up, e.g. "MongoDB", "Maya" (optional).
    """
    try:
        config = _get_config()

        if entity:
            facts = get_facts_for_entity(
                entity=entity,
                project=project,
                facts_dir=config.facts_dir,
            )
        elif project:
            facts = get_active_facts(
                project=project,
                facts_dir=config.facts_dir,
            )
        else:
            return "Provide at least one of: project or entity."

        if not facts:
            scope = f" for entity '{entity}'" if entity else f" in project '{project}'"
            return f"No active facts found{scope}."

        lines = []
        for f in facts:
            parts = [f"{f.subject} → {f.predicate} → {f.object}"]
            if f.since:
                parts.append(f"since {f.since}")
            if f.confidence < 1.0:
                parts.append(f"confidence {f.confidence:.2f}")
            lines.append(" | ".join(parts))

        header = f"Active facts ({len(facts)}):"
        return header + "\n" + "\n".join(f"  • {line}" for line in lines)
    except Exception as e:
        logger.exception("engram_facts failed")
        return f"Facts error: {e}"


@mcp.tool()
def engram_timeline(
    project: str = None,
    limit: int = 10,
) -> str:
    """Recent memories timeline — shows the most recent memories
    ordered by creation date. Useful for understanding recent activity.

    Args:
        project: Filter to a specific project (optional).
        limit: Maximum number of entries (default 10).
    """
    try:
        stack = _get_stack()

        recent = stack.index.metadata_query(
            project=project,
            order_by="created",
            limit=limit,
        )

        if not recent:
            scope = f" for project '{project}'" if project else ""
            return f"No memories found{scope}."

        lines = [f"Recent memories ({len(recent)}):"]
        for i, mem in enumerate(recent, 1):
            created = str(mem.get("created", ""))[:10]
            mem_type = mem.get("memory_type", "note")
            mem_id = mem.get("id", "?")
            file_path = mem.get("file_path", "")

            # Build a readable title from file path or ID
            if file_path:
                from pathlib import Path
                import re
                stem = Path(file_path).stem
                title = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", stem)
                title = title.replace("-", " ").replace("_", " ")
            else:
                title = mem_id

            proj = mem.get("project", "")
            proj_part = f" [{proj}]" if proj else ""
            lines.append(f"  {i}. [{created}] ({mem_type}){proj_part} {title}")

        return "\n".join(lines)
    except Exception as e:
        logger.exception("engram_timeline failed")
        return f"Timeline error: {e}"


@mcp.tool()
def engram_conflicts(
    project: str = None,
) -> str:
    """List all unresolved fact conflicts. Conflicts arise when two facts
    contradict each other and couldn't be auto-resolved.

    Returns conflict descriptions with their index numbers — use the index
    with engram_resolve_conflict to resolve them.

    Args:
        project: Filter to a specific project (optional — omit for all projects).
    """
    try:
        config = _get_config()

        conflicts = get_unresolved_conflicts(
            project=project,
            facts_dir=config.facts_dir,
        )

        if not conflicts:
            scope = f" in project '{project}'" if project else ""
            return f"No unresolved conflicts{scope}. ✅"

        lines = [f"⚠️ Unresolved conflicts ({len(conflicts)}):"]
        for i, c in enumerate(conflicts):
            proj = c.get("project", "?")
            desc = c.get("description", "?")
            detected = c.get("detected", "?")
            lines.append(f"  [{i}] [{proj}] {desc} (detected: {detected})")

        lines.append("")
        lines.append("Use engram_resolve_conflict(project, conflict_index, winner='a'|'b') to resolve.")
        return "\n".join(lines)
    except Exception as e:
        logger.exception("engram_conflicts failed")
        return f"Conflicts error: {e}"


@mcp.tool()
def engram_list_projects(
    status: str = None,
) -> str:
    """List all known projects with their status and last activity date.

    Args:
        status: Filter by status — "active", "paused", or "archived" (optional).
    """
    try:
        config = _get_config()

        projects = list_projects(
            status=status,
            projects_dir=config.projects_dir,
        )

        if not projects:
            filter_part = f" with status '{status}'" if status else ""
            return f"No projects found{filter_part}."

        lines = [f"Projects ({len(projects)}):"]
        for p in projects:
            pid = p.get("id", "?")
            display = p.get("display_name", pid)
            p_status = p.get("status", "active")
            desc = p.get("description", "")
            last_active_raw = p.get("last_active")
            last_active = str(last_active_raw)[:10] if last_active_raw else "?"

            line = f"  • {display} ({pid}) [{p_status}] — last active: {last_active}"
            if desc:
                line += f"\n    {desc}"
            lines.append(line)

        return "\n".join(lines)
    except Exception as e:
        logger.exception("engram_list_projects failed")
        return f"List projects error: {e}"


@mcp.tool()
def engram_wake_up(
    project: str = None,
    cwd: str = None,
) -> str:
    """Session startup — call this at the beginning of every session.
    Returns combined L0 (identity) + L1 (working set) context including:
    - Who the user is (identity.md)
    - Active project overview, facts, recent memories, and conflicts.

    Also sets the active project for subsequent calls.

    Args:
        project: Explicit project to activate (optional).
        cwd: Current working directory for project auto-detection (optional).
    """
    try:
        stack = _get_stack()
        context = stack.wake_up(project=project, cwd=cwd)
        return context if context else "Engram is ready. No identity or project context loaded."
    except Exception as e:
        logger.exception("engram_wake_up failed")
        return f"Wake-up error: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# WRITE TOOLS (4)
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def engram_remember(
    content: str,
    project: str = None,
    topics: str = None,
    memory_type: str = "note",
    source: str = "mcp",
) -> str:
    """Store a new memory. Runs the full pipeline: quality gate → dedup →
    write Markdown → index → extract facts → conflict detection.

    Args:
        content: The text to remember (decisions, notes, problems, etc.).
        project: Project to associate with (optional).
        topics: Comma-separated topic tags, e.g. "auth,migration" (optional).
        memory_type: Type of memory — "note", "decision", "milestone",
                     "problem", "preference", or "emotional" (default "note").
        source: Where this memory came from — "mcp", "claude-code",
                "chatgpt", "manual" (default "mcp").
    """
    try:
        config = _get_config()

        topic_list = None
        if topics:
            topic_list = [t.strip() for t in topics.split(",") if t.strip()]

        result = remember(
            content=content,
            project=project,
            topics=topic_list,
            memory_type=memory_type,
            source=source,
            config=config,
        )

        if not result.success:
            reason = result.rejected_reason
            if reason.startswith("duplicate:"):
                return (
                    f"Not stored — duplicate detected ({reason}). "
                    f"Existing memory: {result.id}"
                )
            return f"Not stored — rejected: {reason}"

        # Build a summary of what happened
        parts = [f"✅ Memory stored: {result.id}"]

        if result.file_path:
            parts.append(f"   File: {result.file_path}")

        if result.facts_extracted > 0:
            parts.append(
                f"   Facts: {result.facts_extracted} extracted, "
                f"{result.facts_added} added"
            )

        if result.conflicts_detected > 0:
            parts.append(f"   ⚠️ {result.conflicts_detected} conflict(s) detected:")
            for cd in result.conflict_details:
                parts.append(
                    f"      {cd.get('type', '?')}: "
                    f"{cd.get('old', '?')} vs {cd.get('new', '?')} "
                    f"→ {cd.get('resolution', '?')}"
                )

        return "\n".join(parts)
    except Exception as e:
        logger.exception("engram_remember failed")
        return f"Remember error: {e}"


@mcp.tool()
def engram_learn_fact(
    project: str,
    subject: str,
    predicate: str,
    object_val: str,
    confidence: float = 1.0,
    since: str = "",
) -> str:
    """Manually add a fact to a project's knowledge base.
    Facts are subject → predicate → object triples.

    Examples:
    - ("saas-app", "saas-app", "database", "MongoDB")
    - ("saas-app", "Maya", "role", "lead developer")
    - ("saas-app", "saas-app", "uses_auth", "Clerk")

    Args:
        project: Project ID to store the fact in.
        subject: The entity the fact is about (e.g. "saas-app", "Maya").
        predicate: The relationship (e.g. "database", "uses_auth", "role").
        object_val: The value (e.g. "MongoDB", "Clerk", "lead developer").
        confidence: Confidence level 0.0–1.0 (default 1.0).
        since: When this fact became true, e.g. "2026-03" (optional).
    """
    try:
        config = _get_config()

        result = add_fact(
            project=project,
            subject=subject,
            predicate=predicate,
            object_val=object_val,
            confidence=confidence,
            source_memory_id="",
            since=since,
            source_text="",
            facts_dir=config.facts_dir,
            exclusive_predicates=config.exclusive_predicates,
        )

        if result.get("added"):
            msg = f"✅ Fact added: {subject} → {predicate} → {object_val}"
            conflict = result.get("conflict")
            if conflict:
                resolution = result.get("resolution", {})
                action = resolution.get("action", "?")
                reason = resolution.get("resolution_reason", "")
                msg += f"\n   ⚠️ Conflict resolved ({action}): {reason}"
            return msg
        else:
            conflict = result.get("conflict")
            if conflict is None:
                return (
                    f"Fact already exists: {subject} → {predicate} → {object_val} "
                    f"(confidence updated if higher)"
                )
            return f"Fact not added due to conflict: {conflict}"
    except Exception as e:
        logger.exception("engram_learn_fact failed")
        return f"Learn fact error: {e}"


@mcp.tool()
def engram_forget_fact(
    project: str,
    subject: str,
    predicate: str,
    object_val: str,
    reason: str = "",
) -> str:
    """Expire (soft-delete) a fact. Moves it from Current to Expired.
    The fact is preserved in history but no longer treated as active.

    Args:
        project: Project ID.
        subject: Fact subject (e.g. "saas-app").
        predicate: Fact predicate (e.g. "database").
        object_val: Fact value (e.g. "Postgres").
        reason: Why this fact is being expired (optional).
    """
    try:
        config = _get_config()

        success = expire_fact(
            project=project,
            subject=subject,
            predicate=predicate,
            object_val=object_val,
            reason=reason,
            facts_dir=config.facts_dir,
        )

        if success:
            return f"✅ Fact expired: {subject} → {predicate} → {object_val}"
        else:
            return (
                f"Fact not found: {subject} → {predicate} → {object_val} "
                f"in project '{project}'"
            )
    except Exception as e:
        logger.exception("engram_forget_fact failed")
        return f"Forget fact error: {e}"


@mcp.tool()
def engram_resolve_conflict(
    project: str,
    conflict_index: int,
    winner: str,
) -> str:
    """Manually resolve a fact conflict by choosing which side is correct.

    Use engram_conflicts to see the list of unresolved conflicts and their
    index numbers. Then call this with the index and winner choice.

    Args:
        project: Project ID where the conflict exists.
        conflict_index: Index of the conflict (from engram_conflicts output).
        winner: Which fact to keep — "a" (first/older) or "b" (second/newer).
    """
    try:
        if winner not in ("a", "b"):
            return "Invalid winner — must be 'a' or 'b'."

        config = _get_config()

        success = resolve_conflict_manual(
            project=project,
            conflict_index=conflict_index,
            winner=winner,
            facts_dir=config.facts_dir,
        )

        if success:
            return (
                f"✅ Conflict #{conflict_index} in '{project}' resolved — "
                f"winner: {'fact A (older)' if winner == 'a' else 'fact B (newer)'}."
            )
        else:
            return (
                f"Could not resolve conflict #{conflict_index} in '{project}'. "
                f"Check that the index is valid and the conflict is still unresolved."
            )
    except Exception as e:
        logger.exception("engram_resolve_conflict failed")
        return f"Resolve conflict error: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# Server entry point
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """Run the Engram MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
