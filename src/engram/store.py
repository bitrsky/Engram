"""
store.py — Markdown file read/write for Engram.

Every memory is a Markdown file with YAML frontmatter.
This module handles parsing, writing, listing, and updating these files.
Markdown is the Source of Truth — indexes are derived from these files.
"""

import hashlib
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .config import EngramConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def slugify(text: str, max_length: int = 40) -> str:
    """
    Convert text to a safe filename slug.

    - Lowercase
    - Replace non-alphanumeric with hyphens
    - Collapse multiple hyphens
    - Strip leading/trailing hyphens
    - Truncate to max_length
    - Handle unicode (normalize to ASCII)

    Examples:
        slugify("Auth Provider Decision") → "auth-provider-decision"
        slugify("我们决定用Clerk") → "clerk" (or transliterated)
        slugify("A" * 100) → "aaaa..." (truncated to 40)
    """
    # Normalize unicode → decompose, then strip non-ASCII marks
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Lowercase
    text = text.lower()

    # Replace any non-alphanumeric character with a hyphen
    text = re.sub(r"[^a-z0-9]+", "-", text)

    # Collapse multiple hyphens
    text = re.sub(r"-{2,}", "-", text)

    # Strip leading/trailing hyphens
    text = text.strip("-")

    # Truncate — but try not to cut in the middle of a word/segment
    if len(text) > max_length:
        text = text[:max_length].rstrip("-")

    # If nothing survived (e.g. purely CJK with no Latin), fall back
    if not text:
        text = "memory"

    return text


def generate_memory_id(content: str) -> str:
    """
    Generate a unique memory ID.
    Format: mem_{YYYY-MM-DD}_{sha256_first_6}
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:6]
    return f"mem_{date_str}_{content_hash}"


def _default_memories_dir() -> Path:
    """Return the default memories directory from EngramConfig."""
    return Path(EngramConfig().memories_dir)


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

# Regex that matches the opening/closing --- fences of YAML frontmatter.
# The frontmatter must start at the very beginning of the file.
_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?\r?\n)---[ \t]*\r?\n?",
    re.DOTALL,
)


def parse_frontmatter(filepath: str | Path) -> Tuple[dict, str]:
    """
    Parse a Markdown file with YAML frontmatter.

    Format:
        ---
        key: value
        ---

        Body text here.

    Args:
        filepath: Path to the .md file

    Returns:
        (frontmatter_dict, body_string)
        If no frontmatter found, returns ({}, full_text)
    """
    filepath = Path(filepath)
    raw = filepath.read_text(encoding="utf-8")

    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    yaml_block = match.group(1)
    body = raw[match.end():]

    try:
        meta = yaml.safe_load(yaml_block)
    except yaml.YAMLError:
        # Malformed YAML — treat the whole file as body
        return {}, raw

    # yaml.safe_load returns None for an empty document (e.g. "---\n---\n")
    if meta is None:
        meta = {}

    if not isinstance(meta, dict):
        # Unexpected YAML type (e.g. a bare scalar) — treat as no frontmatter
        return {}, raw

    return meta, body


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_memory(
    content: str,
    project: str = None,
    topics: List[str] = None,
    memory_type: str = "note",
    source: str = "manual",
    importance: float = 3.0,
    memories_dir: str | Path = None,
    memory_id: str = None,
) -> Path:
    """
    Write a new memory as a Markdown file.

    Generates:
      - ID: mem_{YYYY-MM-DD}_{sha256_first_6_of_content}
      - Filename: {YYYY-MM-DD}_{slugified_first_40_chars}.md

    Frontmatter fields:
      id, project, topics, memory_type, importance,
      created (ISO datetime), source, access_count (0), last_accessed (null)

    Args:
        content: The verbatim text to store
        project: Project tag (can be None for cross-project memories)
        topics: List of topic tags
        memory_type: "note" | "decision" | "milestone" | "problem" | "preference" | "emotional"
        source: "manual" | "claude-code" | "chatgpt" | "slack" | "ingest"
        importance: Initial importance weight (default 3.0)
        memories_dir: Override memories directory (default: ~/.engram/memories/)
        memory_id: Override auto-generated ID (useful for benchmarks / imports)

    Returns:
        Path to the written file
    """
    if memories_dir is None:
        memories_dir = _default_memories_dir()
    memories_dir = Path(memories_dir)
    memories_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    if memory_id is None:
        memory_id = generate_memory_id(content)
    date_str = now.strftime("%Y-%m-%d")

    # Build frontmatter -------------------------------------------------------
    frontmatter: Dict = {
        "id": memory_id,
        "project": project,
        "topics": topics if topics else [],
        "memory_type": memory_type,
        "importance": importance,
        "created": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source,
        "access_count": 0,
        "last_accessed": None,
    }

    # Build filename -----------------------------------------------------------
    slug = slugify(content, max_length=40)
    base_name = f"{date_str}_{slug}"
    filepath = memories_dir / f"{base_name}.md"

    # Handle filename collisions — append _1, _2, …
    counter = 1
    while filepath.exists():
        filepath = memories_dir / f"{base_name}_{counter}.md"
        counter += 1

    # Serialize ----------------------------------------------------------------
    # Use yaml.dump with default_flow_style=False for readable output.
    # Ensure None values are rendered as 'null' (YAML default).
    yaml_text = yaml.dump(
        frontmatter,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )

    file_content = f"---\n{yaml_text}---\n\n{content}\n"

    filepath.write_text(file_content, encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_memory(filepath: str | Path) -> dict:
    """
    Read a memory file and return as a dict.

    Returns:
        {
            "id": "mem_2026-01-15_a3f8c2",
            "project": "saas-app",
            "topics": ["auth", "infrastructure"],
            "memory_type": "decision",
            "importance": 4.0,
            "created": "2026-01-15T14:23:00",
            "source": "claude-code",
            "access_count": 3,
            "last_accessed": "2026-03-01T10:00:00",
            "content": "We decided to use Clerk...",
            "file_path": "/path/to/file.md"
        }
    """
    filepath = Path(filepath)
    meta, body = parse_frontmatter(filepath)

    result = {
        "id": meta.get("id"),
        "project": meta.get("project"),
        "topics": meta.get("topics", []),
        "memory_type": meta.get("memory_type"),
        "importance": meta.get("importance"),
        "created": meta.get("created"),
        "source": meta.get("source"),
        "access_count": meta.get("access_count", 0),
        "last_accessed": meta.get("last_accessed"),
        "content": body.strip(),
        "file_path": str(filepath.resolve()),
    }
    return result


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def list_memories(
    memories_dir: str | Path = None,
    project: str = None,
    since: str = None,
    limit: int = None,
) -> List[dict]:
    """
    List memories from the memories directory.

    Args:
        memories_dir: Override memories directory
        project: Filter by project tag
        since: ISO date string, only return memories created after this date
        limit: Max number of results

    Returns:
        List of dicts (same format as read_memory), sorted by created DESC
    """
    if memories_dir is None:
        memories_dir = _default_memories_dir()
    memories_dir = Path(memories_dir)

    if not memories_dir.is_dir():
        return []

    results: List[dict] = []

    for md_file in memories_dir.glob("*.md"):
        try:
            meta, body = parse_frontmatter(md_file)
        except (OSError, UnicodeDecodeError):
            # Skip files that can't be read
            continue

        # If frontmatter has no 'id', it's probably not a memory file — skip
        if "id" not in meta:
            continue

        # Project filter
        if project is not None and meta.get("project") != project:
            continue

        # Since filter
        created_str = meta.get("created")
        if since is not None and created_str is not None:
            try:
                # Handle both date-only and datetime strings
                if "T" in str(since):
                    since_dt = datetime.fromisoformat(str(since))
                else:
                    since_dt = datetime.fromisoformat(str(since) + "T00:00:00")

                if "T" in str(created_str):
                    created_dt = datetime.fromisoformat(str(created_str))
                else:
                    created_dt = datetime.fromisoformat(str(created_str) + "T00:00:00")

                if created_dt <= since_dt:
                    continue
            except (ValueError, TypeError):
                # Can't parse dates — include the memory anyway
                pass

        entry = {
            "id": meta.get("id"),
            "project": meta.get("project"),
            "topics": meta.get("topics", []),
            "memory_type": meta.get("memory_type"),
            "importance": meta.get("importance"),
            "created": created_str,
            "source": meta.get("source"),
            "access_count": meta.get("access_count", 0),
            "last_accessed": meta.get("last_accessed"),
            "content": body.strip(),
            "file_path": str(md_file.resolve()),
        }
        results.append(entry)

    # Sort by created DESC (most recent first)
    def _sort_key(entry: dict):
        created = entry.get("created")
        if created is None:
            return ""
        return str(created)

    results.sort(key=_sort_key, reverse=True)

    # Apply limit
    if limit is not None:
        results = results[:limit]

    return results


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


def update_frontmatter(filepath: str | Path, updates: dict) -> None:
    """
    Update specific frontmatter fields without changing the body.

    Args:
        filepath: Path to the .md file
        updates: Dict of field -> new_value to update

    Example:
        update_frontmatter("memory.md", {"importance": 5.0, "access_count": 4})
    """
    filepath = Path(filepath)
    raw = filepath.read_text(encoding="utf-8")

    match = _FRONTMATTER_RE.match(raw)
    if not match:
        # No frontmatter — create one with just the updates, preserve body
        meta = dict(updates)
        body = raw
    else:
        yaml_block = match.group(1)
        body = raw[match.end():]

        try:
            meta = yaml.safe_load(yaml_block)
        except yaml.YAMLError:
            meta = {}

        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            meta = {}

        meta.update(updates)

    yaml_text = yaml.dump(
        meta,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )

    new_content = f"---\n{yaml_text}---\n{body}"
    filepath.write_text(new_content, encoding="utf-8")
