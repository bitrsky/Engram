"""
projects.py — Project registry for Engram.

Projects are logical concepts stored as Markdown files.
They are NOT tied to workspace directories.
A project has:
- id: unique slug identifier
- display_name: human-readable name
- status: active | paused | archived
- aliases: alternative names for matching
- associated_paths: optional weak associations to filesystem paths (for auto-detection)
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config import EngramConfig
from .store import parse_frontmatter, update_frontmatter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_projects_dir() -> Path:
    """Get default projects directory from config."""
    return EngramConfig().projects_dir


def _resolve_projects_dir(projects_dir: str | Path = None) -> Path:
    """Resolve projects directory, creating it if needed."""
    if projects_dir is None:
        projects_dir = _default_projects_dir()
    projects_dir = Path(projects_dir)
    projects_dir.mkdir(parents=True, exist_ok=True)
    return projects_dir


def _project_file(project_id: str, projects_dir: Path) -> Path:
    """Return the path to a project's Markdown file."""
    return projects_dir / f"{project_id}.md"


def _parse_last_active(value) -> datetime:
    """
    Parse a last_active value into a datetime for sorting.
    Handles datetime objects (from YAML), ISO strings, and None.
    Returns datetime.min for unparseable values so they sort last.
    """
    if value is None:
        return datetime.min
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return datetime.min


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def create_project(
    project_id: str,
    display_name: str,
    description: str = "",
    aliases: List[str] = None,
    associated_paths: List[str] = None,
    tags: List[str] = None,
    projects_dir: str | Path = None,
) -> Path:
    """
    Create a new project file.

    Args:
        project_id: Unique slug (e.g. "saas-app")
        display_name: Human name (e.g. "My SaaS App")
        description: One-line description
        aliases: Alternative names for matching (e.g. ["saas", "the app"])
        associated_paths: Optional filesystem paths (weak association, not binding)
        tags: Classification tags
        projects_dir: Override directory

    Returns:
        Path to created project file

    Raises:
        FileExistsError if project already exists
    """
    pdir = _resolve_projects_dir(projects_dir)
    filepath = _project_file(project_id, pdir)

    if filepath.exists():
        raise FileExistsError(f"Project '{project_id}' already exists: {filepath}")

    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    frontmatter: Dict = {
        "id": project_id,
        "display_name": display_name,
        "status": "active",
        "description": description,
        "created": now,
        "last_active": now,
        "aliases": aliases if aliases else [],
        "associated_paths": associated_paths if associated_paths else [],
        "tags": tags if tags else [],
    }

    yaml_text = yaml.dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )

    body = ""
    if description:
        body = f"\n{description}\n"
    else:
        body = "\n"

    file_content = f"---\n{yaml_text}---\n{body}"
    filepath.write_text(file_content, encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def get_project(project_id: str, projects_dir: str | Path = None) -> Optional[dict]:
    """
    Get a project by ID.

    Returns:
        Dict with all frontmatter fields + "body" (free-text notes area),
        or None if project doesn't exist.
    """
    pdir = _resolve_projects_dir(projects_dir)
    filepath = _project_file(project_id, pdir)

    if not filepath.exists():
        return None

    meta, body = parse_frontmatter(filepath)

    result = dict(meta)
    result["body"] = body.strip()
    result["file_path"] = str(filepath.resolve())
    return result


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def list_projects(
    status: str = None,
    projects_dir: str | Path = None,
) -> List[dict]:
    """
    List all projects.

    Args:
        status: Filter by status ("active", "paused", "archived").
                None = return all.

    Returns:
        List of project dicts, sorted by last_active DESC
    """
    pdir = _resolve_projects_dir(projects_dir)

    if not pdir.is_dir():
        return []

    results: List[dict] = []

    for md_file in pdir.glob("*.md"):
        # Skip _index.md
        if md_file.name == "_index.md":
            continue

        try:
            meta, body = parse_frontmatter(md_file)
        except (OSError, UnicodeDecodeError):
            continue

        # Must have "id" in frontmatter to be a valid project file
        if "id" not in meta:
            continue

        # Status filter
        if status is not None and meta.get("status") != status:
            continue

        entry = dict(meta)
        entry["body"] = body.strip()
        entry["file_path"] = str(md_file.resolve())
        results.append(entry)

    # Sort by last_active DESC (most recently active first)
    results.sort(key=lambda e: _parse_last_active(e.get("last_active")), reverse=True)

    return results


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


def update_project(
    project_id: str,
    projects_dir: str | Path = None,
    **updates,
) -> None:
    """
    Update project frontmatter fields.

    Common updates:
        last_active="2026-03-15T10:00:00"
        status="archived"
        aliases=["new-alias"]
        associated_paths=["/new/path"]

    Raises:
        FileNotFoundError if project doesn't exist.
    """
    pdir = _resolve_projects_dir(projects_dir)
    filepath = _project_file(project_id, pdir)

    if not filepath.exists():
        raise FileNotFoundError(f"Project '{project_id}' not found: {filepath}")

    update_frontmatter(filepath, updates)


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


def archive_project(project_id: str, projects_dir: str | Path = None) -> None:
    """Set project status to 'archived'."""
    update_project(project_id, projects_dir=projects_dir, status="archived")


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------


def resolve_project(
    cwd: str = None,
    message: str = None,
    explicit: str = None,
    projects_dir: str | Path = None,
) -> Optional[str]:
    """
    Determine which project the current context belongs to.

    Three-level routing with fallback:

    1. Explicit declaration (highest priority)
       → If explicit is not None, return it (after validating it exists)

    2. Path matching (weak association)
       → If cwd is not None, check each project's associated_paths
       → A project matches if cwd starts with any of its associated_paths
       → (or vice versa — the associated_path starts with cwd)
       → Use the LONGEST matching path (most specific)

    3. Alias matching (keyword in message)
       → If message is not None, check each project's aliases + id + display_name
       → Case-insensitive substring match
       → Return first match (by most-recently-active project priority)

    4. Fallback → None (cross-project mode)

    Returns:
        project_id string, or None
    """
    pdir = _resolve_projects_dir(projects_dir)

    # ── 1. Explicit declaration ──────────────────────────────────────────
    if explicit is not None:
        filepath = _project_file(explicit, pdir)
        if filepath.exists():
            return explicit
        # Explicit project doesn't exist — don't fall through, just return None
        return None

    # Load all projects once (sorted by last_active DESC for alias priority)
    all_projects = list_projects(projects_dir=pdir)

    # ── 2. Path matching ────────────────────────────────────────────────
    if cwd is not None:
        try:
            cwd_path = Path(cwd).expanduser().resolve()
        except (OSError, ValueError):
            cwd_path = None

        if cwd_path is not None:
            best_match_id: Optional[str] = None
            best_match_len: int = -1

            for proj in all_projects:
                assoc_paths = proj.get("associated_paths", [])
                if not assoc_paths:
                    continue

                for ap in assoc_paths:
                    try:
                        ap_path = Path(ap).expanduser().resolve()
                    except (OSError, ValueError):
                        continue

                    # Check if cwd is under the associated path
                    # or the associated path is under cwd
                    try:
                        is_match = (
                            cwd_path == ap_path
                            or cwd_path.is_relative_to(ap_path)
                            or ap_path.is_relative_to(cwd_path)
                        )
                    except (ValueError, TypeError):
                        is_match = False

                    if is_match:
                        # Use length of the associated_path string for specificity
                        path_len = len(str(ap_path))
                        if path_len > best_match_len:
                            best_match_len = path_len
                            best_match_id = proj["id"]

            if best_match_id is not None:
                return best_match_id

    # ── 3. Alias matching ───────────────────────────────────────────────
    if message is not None:
        message_lower = message.lower()

        # all_projects is already sorted by last_active DESC
        for proj in all_projects:
            # Build list of matchable terms: id, display_name, aliases
            terms: List[str] = []
            terms.append(proj.get("id", ""))
            terms.append(proj.get("display_name", ""))
            for alias in proj.get("aliases", []):
                terms.append(str(alias))

            for term in terms:
                if term and term.lower() in message_lower:
                    return proj["id"]

    # ── 4. Fallback ─────────────────────────────────────────────────────
    return None


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def update_project_index(projects_dir: str | Path = None) -> Path:
    """
    Rebuild the _index.md file — a human-readable overview of all projects.

    Format:
    ---
    generated: 2026-03-15T10:00:00
    ---

    # Projects

    ## Active

    ### My SaaS App (saas-app)
    Status: active | Last active: 2026-03-15
    One-line description here.

    ### ML Research (ml-research)
    Status: active | Last active: 2026-02-20
    Personal AI research notes.

    ## Archived

    ### Old Project (old-project)
    Status: archived | Last active: 2025-06-01

    Returns:
        Path to _index.md
    """
    pdir = _resolve_projects_dir(projects_dir)
    all_projects = list_projects(projects_dir=pdir)

    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Group projects by status
    groups: Dict[str, List[dict]] = {
        "active": [],
        "paused": [],
        "archived": [],
    }

    for proj in all_projects:
        status = proj.get("status", "active")
        if status not in groups:
            groups[status] = []
        groups[status].append(proj)

    # Build frontmatter
    frontmatter_yaml = yaml.dump(
        {"generated": now},
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )

    lines: List[str] = []
    lines.append(f"---\n{frontmatter_yaml}---\n")
    lines.append("# Projects\n")

    # Render each status group (only if it has projects)
    status_order = ["active", "paused", "archived"]

    for status in status_order:
        projects_in_group = groups.get(status, [])
        if not projects_in_group:
            continue

        lines.append(f"## {status.capitalize()}\n")

        for proj in projects_in_group:
            display_name = proj.get("display_name", proj.get("id", "Unknown"))
            project_id = proj.get("id", "unknown")
            proj_status = proj.get("status", "active")

            # Format last_active — show date only
            last_active_raw = proj.get("last_active")
            if last_active_raw:
                last_active_str = str(last_active_raw)[:10]  # YYYY-MM-DD
            else:
                last_active_str = "unknown"

            lines.append(f"### {display_name} ({project_id})")
            lines.append(f"Status: {proj_status} | Last active: {last_active_str}")

            description = proj.get("description", "")
            if description:
                lines.append(description)

            lines.append("")  # blank line between projects

    # Handle any non-standard statuses
    for status, projects_in_group in groups.items():
        if status in status_order:
            continue
        if not projects_in_group:
            continue

        lines.append(f"## {status.capitalize()}\n")

        for proj in projects_in_group:
            display_name = proj.get("display_name", proj.get("id", "Unknown"))
            project_id = proj.get("id", "unknown")
            proj_status = proj.get("status", "active")

            last_active_raw = proj.get("last_active")
            if last_active_raw:
                last_active_str = str(last_active_raw)[:10]
            else:
                last_active_str = "unknown"

            lines.append(f"### {display_name} ({project_id})")
            lines.append(f"Status: {proj_status} | Last active: {last_active_str}")

            description = proj.get("description", "")
            if description:
                lines.append(description)

            lines.append("")

    index_path = pdir / "_index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path
