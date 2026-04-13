"""
layers.py — 4-layer retrieval stack for Engram.

L0: Identity — who am I? (static, always loaded)
L1: Working Set — what am I working on? (≤500 tokens, auto-generated)
L2: Contextual — what's relevant to this message? (auto-triggered)
L3: Deep Search — explicit semantic search (user-triggered)

MemoryStack is the unified interface.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from .config import EngramConfig
from .index import IndexManager, SearchHit
from .projects import get_project, resolve_project, list_projects
from .facts import get_active_facts, get_unresolved_conflicts

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# MemoryStack
# ═══════════════════════════════════════════════════════════════════════════


class MemoryStack:
    """
    Unified interface for the 4-layer retrieval stack.

    Usage:
        stack = MemoryStack()

        # On session start:
        context = stack.wake_up(project="saas-app")

        # When user mentions something:
        relevant = stack.recall(message="the auth migration issue")

        # Explicit search:
        results = stack.search("MongoDB performance tuning")

    With LLM callback (preferred when hosted by an AI agent)::

        def my_llm(prompt, system="", **kw):
            return call_model(prompt, system_message=system)

        stack = MemoryStack(config=cfg, think_fn=my_think)
    """

    def __init__(self, config: Optional[EngramConfig] = None, think_fn=None):
        self._config = config or EngramConfig()
        self._think_fn = think_fn
        self._index: Optional[IndexManager] = None
        self._active_project: Optional[str] = None

    @property
    def index(self) -> IndexManager:
        """Lazy-init IndexManager."""
        if self._index is None:
            self._index = IndexManager(
                index_dir=self._config.index_dir,
                memories_dir=self._config.memories_dir,
                facts_dir=self._config.facts_dir,
                projects_dir=self._config.projects_dir,
            )
        return self._index

    def close(self):
        """Release resources."""
        if self._index is not None:
            self._index.close()
            self._index = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ──────────────────────────────────────────────
    # L0: Identity
    # ──────────────────────────────────────────────

    def identity(self) -> str:
        """
        L0: Read identity file.

        Returns contents of ~/.engram/identity.md (body only, no frontmatter).
        Returns empty string if file doesn't exist.
        """
        try:
            id_path = self._config.identity_path
            if not id_path.exists():
                return ""

            raw = id_path.read_text(encoding="utf-8")

            # Strip YAML frontmatter (--- ... ---)
            fm_match = re.match(
                r"\A---[ \t]*\r?\n(.*?\r?\n)---[ \t]*\r?\n?",
                raw,
                re.DOTALL,
            )
            if fm_match:
                body = raw[fm_match.end():]
            else:
                body = raw

            return body.strip()
        except Exception as exc:
            logger.debug("L0 identity read failed: %s", exc)
            return ""

    # ──────────────────────────────────────────────
    # L1: Working Set
    # ──────────────────────────────────────────────

    def working_set(self, project: Optional[str] = None, max_tokens: int = 500) -> str:
        """
        L1: Generate working set context.

        If project is specified:
          1. Project overview from projects/{project}.md (first 200 tokens)
          2. Active facts from facts/{project}.md (each fact = 1 line)
          3. Recent 5 memories (titles + first line, from SQLite)
          4. Unresolved conflicts (if any)

        If project is None (cross-project mode):
          1. List of all active projects (one line each)
          2. Global recent memories (last 5)
          3. Cross-project unresolved conflicts

        Truncated to max_tokens.

        Returns: formatted string for AI system prompt injection
        """
        try:
            parts: List[str] = []

            if project is not None:
                parts.append(self._working_set_project(project))
            else:
                parts.append(self._working_set_cross_project())

            text = "\n".join(p for p in parts if p)
            return _truncate_to_tokens(text, max_tokens)
        except Exception as exc:
            logger.debug("L1 working_set failed: %s", exc)
            return ""

    def _working_set_project(self, project: str) -> str:
        """Build working set for a single project."""
        sections: List[str] = []

        # 1. Project overview (first ~200 tokens)
        try:
            proj_data = get_project(
                project,
                projects_dir=self._config.projects_dir,
            )
            if proj_data:
                display = proj_data.get("display_name", project)
                status = proj_data.get("status", "active")
                desc = proj_data.get("description", "")
                header = f"📂 {display} [{status}]"
                if desc:
                    header += f" — {desc}"
                body = proj_data.get("body", "")
                if body:
                    body = _truncate_to_tokens(body, 150)
                    header += f"\n{body}"
                sections.append(_truncate_to_tokens(header, 200))
            else:
                sections.append(f"📂 {project} [not found]")
        except Exception:
            sections.append(f"📂 {project}")

        # 2. Active facts
        try:
            facts = get_active_facts(
                project,
                facts_dir=self._config.facts_dir,
            )
            if facts:
                fact_lines = [_format_fact_line(f) for f in facts]
                sections.append("Facts:\n" + "\n".join(fact_lines))
        except Exception:
            pass

        # 3. Recent memories (last 5)
        try:
            recent = self.index.metadata_query(
                project=project,
                order_by="created",
                limit=5,
            )
            if recent:
                mem_lines = [_format_memory_summary(m) for m in recent]
                sections.append("Recent:\n" + "\n".join(mem_lines))
        except Exception:
            pass

        # 4. Unresolved conflicts
        try:
            conflicts = get_unresolved_conflicts(
                project=project,
                facts_dir=self._config.facts_dir,
            )
            if conflicts:
                conflict_lines = [
                    f"  ⚠️ {c.get('description', '?')}" for c in conflicts
                ]
                sections.append(
                    "Conflicts:\n" + "\n".join(conflict_lines)
                )
        except Exception:
            pass

        return "\n".join(sections)

    def _working_set_cross_project(self) -> str:
        """Build working set for cross-project mode."""
        sections: List[str] = []

        # 1. List active projects
        try:
            projects = list_projects(
                status="active",
                projects_dir=self._config.projects_dir,
            )
            if projects:
                proj_lines: List[str] = []
                for p in projects:
                    display = p.get("display_name", p.get("id", "?"))
                    pid = p.get("id", "?")
                    desc = p.get("description", "")
                    line = f"  • {display} ({pid})"
                    if desc:
                        line += f" — {desc}"
                    proj_lines.append(line)
                sections.append("Active projects:\n" + "\n".join(proj_lines))
            else:
                sections.append("No active projects.")
        except Exception:
            pass

        # 2. Global recent memories
        try:
            recent = self.index.metadata_query(
                order_by="created",
                limit=5,
            )
            if recent:
                mem_lines = [_format_memory_summary(m) for m in recent]
                sections.append("Recent:\n" + "\n".join(mem_lines))
        except Exception:
            pass

        # 3. Cross-project unresolved conflicts
        try:
            conflicts = get_unresolved_conflicts(
                project=None,
                facts_dir=self._config.facts_dir,
            )
            if conflicts:
                conflict_lines = [
                    f"  ⚠️ [{c.get('project', '?')}] {c.get('description', '?')}"
                    for c in conflicts
                ]
                sections.append(
                    "Conflicts:\n" + "\n".join(conflict_lines)
                )
        except Exception:
            pass

        return "\n".join(sections)

    # ──────────────────────────────────────────────
    # L2: Contextual (auto-triggered)
    # ──────────────────────────────────────────────

    def recall(
        self,
        message: Optional[str] = None,
        project: Optional[str] = None,
        topic: Optional[str] = None,
        cwd: Optional[str] = None,
        n: int = 3,
    ) -> str:
        """
        L2: Auto-triggered contextual retrieval.

        Flow:
        1. Resolve project (if not explicitly given):
           resolve_project(cwd=cwd, message=message, explicit=project)
        2. Semantic search with resolved project and topic
        3. Format results as context string

        Args:
            message: User's message (used for project resolution + search query)
            project: Explicit project override
            topic: Topic filter
            cwd: Current working directory
            n: Max results

        Returns: formatted context string, or "" if nothing found
        """
        if not message:
            return ""

        try:
            # 1. Resolve project
            resolved = resolve_project(
                cwd=cwd,
                message=message,
                explicit=project,
                projects_dir=self._config.projects_dir,
            )

            # 2. Semantic search
            topics = [topic] if topic else None
            hits = self.index.vector_search(
                query=message,
                project=resolved,
                topics=topics,
                n=n,
            )

            if not hits:
                return ""

            # 3. Format results
            return _format_recall_results(hits, resolved)
        except Exception as exc:
            logger.debug("L2 recall failed: %s", exc)
            return ""

    # ──────────────────────────────────────────────
    # L3: Deep Search
    # ──────────────────────────────────────────────

    def search(
        self,
        query: str,
        project: Optional[str] = None,
        topics: Optional[List[str]] = None,
        n: int = 5,
    ) -> str:
        """
        L3: Deep semantic search (user-triggered).

        Wraps IndexManager.vector_search with formatting.
        Searches across all projects if project is None.

        Returns: formatted results string
        """
        if not query:
            return "No query provided."

        try:
            if self._config.rerank_enabled:
                hits = self.index.vector_search_reranked(
                    query=query,
                    config=self._config,
                    project=project,
                    topics=topics,
                    n=n,
                    think_fn=self._think_fn,
                )
            else:
                hits = self.index.vector_search(
                    query=query,
                    project=project,
                    topics=topics,
                    n=n,
                )

            if not hits:
                scope = f" in project '{project}'" if project else ""
                return f"No results found for '{query}'{scope}."

            return _format_search_results(hits, query)
        except Exception as exc:
            logger.debug("L3 search failed: %s", exc)
            return f"Search failed: {exc}"

    # ──────────────────────────────────────────────
    # Unified entry points
    # ──────────────────────────────────────────────

    def wake_up(self, project: Optional[str] = None, cwd: Optional[str] = None) -> str:
        """
        Session startup: returns L0 + L1 combined context.

        Also sets active_project for subsequent calls.
        If project is None and cwd is provided, tries to resolve project from cwd.

        Returns: combined context string
        """
        try:
            # Resolve project from cwd if not explicitly given
            resolved = project
            if resolved is None and cwd is not None:
                resolved = resolve_project(
                    cwd=cwd,
                    projects_dir=self._config.projects_dir,
                )

            # Set active project for subsequent calls
            if resolved is not None:
                self._active_project = resolved

            # L0: Identity
            l0 = self.identity()

            # L1: Working Set
            l1 = self.working_set(project=resolved)

            # Combine
            parts: List[str] = []
            if l0:
                parts.append(l0)
            if l1:
                if parts:
                    parts.append("")  # blank separator
                parts.append("─── Working Set ───")
                parts.append(l1)

            return "\n".join(parts) if parts else ""
        except Exception as exc:
            logger.debug("wake_up failed: %s", exc)
            return ""

    def set_active_project(self, project: str):
        """Set the active project for subsequent calls."""
        self._active_project = project

    def get_status(self) -> dict:
        """
        Get overall memory system status.

        Returns:
            {
                "active_project": str or None,
                "total_memories": int,
                "total_projects": int,
                "index_stats": dict,
                "unresolved_conflicts": int,
            }
        """
        try:
            idx_stats = self.index.stats()
        except Exception:
            idx_stats = {
                "total_memories": 0,
                "projects": [],
                "last_rebuild": None,
                "chroma_count": 0,
                "sqlite_count": 0,
            }

        try:
            all_projects = list_projects(
                projects_dir=self._config.projects_dir,
            )
            total_projects = len(all_projects)
        except Exception:
            total_projects = 0

        try:
            conflicts = get_unresolved_conflicts(
                project=None,
                facts_dir=self._config.facts_dir,
            )
            n_conflicts = len(conflicts)
        except Exception:
            n_conflicts = 0

        return {
            "active_project": self._active_project,
            "total_memories": idx_stats.get("total_memories", 0),
            "total_projects": total_projects,
            "index_stats": idx_stats,
            "unresolved_conflicts": n_conflicts,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════════


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Approximate token truncation.
    1 token ≈ 4 chars (rough estimate).
    Truncate at word boundary with "..." suffix.
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text

    # Cut at max_chars, then walk back to the nearest word boundary
    truncated = text[:max_chars]

    # Find last space to avoid cutting mid-word
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]

    return truncated.rstrip() + "..."


def _format_fact_line(fact) -> str:
    """Format a Fact as a compact single line for L1."""
    parts = [fact.subject, fact.predicate, fact.object]

    if fact.confidence < 1.0:
        parts.append(f"conf:{fact.confidence:.1f}")

    return "  • " + " → ".join(parts)


def _format_memory_summary(memory: dict) -> str:
    """Format a memory as a compact single line for L1 (title + first 50 chars)."""
    mem_id = memory.get("id", "?")
    created = str(memory.get("created", ""))[:10]  # YYYY-MM-DD
    mem_type = memory.get("memory_type", "note")

    # Try to get a meaningful title from the content or file_path
    content = ""
    file_path = memory.get("file_path", "")
    if file_path:
        # Use filename stem as a rough title
        stem = Path(file_path).stem
        # Strip date prefix (YYYY-MM-DD_)
        title = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", stem)
        title = title.replace("-", " ").replace("_", " ")
    else:
        title = mem_id

    return f"  • [{created}] ({mem_type}) {title}"


def _format_recall_results(hits: List[SearchHit], project: Optional[str]) -> str:
    """Format L2 recall results as a compact context string."""
    lines: List[str] = []
    scope = f" [{project}]" if project else ""
    lines.append(f"─── Relevant Memories{scope} ───")

    for hit in hits:
        sim_pct = int(hit.similarity * 100)
        created = str(hit.created)[:10] if hit.created else "?"
        topics_str = ", ".join(hit.topics) if hit.topics else ""

        header_parts = [f"[{created}]", f"{sim_pct}%"]
        if hit.project:
            header_parts.append(hit.project)
        if topics_str:
            header_parts.append(topics_str)

        lines.append(" ".join(header_parts))

        # Include content, but truncated to ~100 tokens per result
        content = _truncate_to_tokens(hit.content.strip(), 100)
        lines.append(content)
        lines.append("")  # blank separator

    return "\n".join(lines).rstrip()


def _format_search_results(hits: List[SearchHit], query: str) -> str:
    """Format L3 deep search results with full detail."""
    lines: List[str] = []
    lines.append(f"─── Search: '{query}' ({len(hits)} results) ──��")
    lines.append("")

    for i, hit in enumerate(hits, 1):
        sim_pct = int(hit.similarity * 100)
        created = str(hit.created)[:10] if hit.created else "?"
        topics_str = ", ".join(hit.topics) if hit.topics else "—"

        lines.append(
            f"{i}. [{sim_pct}% match] {hit.id}"
        )
        lines.append(
            f"   Project: {hit.project or '—'} | "
            f"Topics: {topics_str} | "
            f"Type: {hit.memory_type or '—'} | "
            f"Created: {created}"
        )

        # More generous content budget for deep search (~150 tokens)
        content = _truncate_to_tokens(hit.content.strip(), 150)
        # Indent content lines
        for cline in content.split("\n"):
            lines.append(f"   {cline}")
        lines.append("")

    return "\n".join(lines).rstrip()
