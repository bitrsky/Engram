"""
facts.py — Facts file management for Engram.

Each project has a facts file: facts/{project}.md
Facts are stored in a structured Markdown format that is both
human-readable and machine-parseable.

Format:
    ---
    project: saas-app
    last_updated: 2026-03-15
    ---
    
    # Facts: saas-app
    
    ## Current
    
    - saas-app | database | MongoDB | since: 2026-03 | confidence: 0.95
      - source: mem_2026-02-20_mongo-migration
      - superseded: Postgres (2025-06 to 2026-03)
    
    ## Expired
    
    - saas-app | database | Postgres | 2025-06 to 2026-03
      - reason: migrated to MongoDB
      - source: mem_2025-06-01_initial-setup
    
    ## Conflicts
    
    - ⚠️ Maya | preference | TypeScript: "doesn't like it" vs "loves it"
      - fact_a: mem_2026-01-10_maya-ts
      - fact_b: mem_2026-03-05_maya-ts-update
      - detected: 2026-03-05
      - status: unresolved
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config import EngramConfig
from .conflicts import (
    Fact,
    Conflict,
    check_conflict,
    resolve_conflict,
    DEFAULT_EXCLUSIVE_PREDICATES,
)


# ── Internal helpers ────────────────────────────────────────────────────────


def _default_facts_dir() -> Path:
    return EngramConfig().facts_dir


def _facts_path(project: str, facts_dir: Optional[str | Path] = None) -> Path:
    """Return the Path for a project's facts file."""
    d = Path(facts_dir) if facts_dir else _default_facts_dir()
    return d / f"{project}.md"


# ── Formatting ──────────────────────────────────────────────────────────────


def _format_fact_line(fact: Fact, expired: bool = False) -> str:
    """
    Format a Fact as a Markdown line (plus sub-lines).

    Current fact:
    - subject | predicate | object | since: YYYY-MM | confidence: 0.95
      - source: mem_xxx
      - superseded: OldValue (YYYY-MM to YYYY-MM)

    Expired fact:
    - subject | predicate | object | YYYY-MM to YYYY-MM
      - reason: migrated to X
      - source: mem_xxx
    """
    parts: List[str] = [fact.subject, fact.predicate, fact.object]

    if expired or fact.expired_at:
        # Expired fact: show date range
        since_part = fact.since or "?"
        expired_part = fact.expired_at[:7] if len(fact.expired_at) >= 7 else (fact.expired_at or "?")
        # Normalise since to YYYY-MM if longer
        if len(since_part) > 7:
            since_part = since_part[:7]
        parts.append(f"{since_part} to {expired_part}")
    else:
        # Current fact: show since + confidence
        if fact.since:
            since_display = fact.since[:7] if len(fact.since) > 7 else fact.since
            parts.append(f"since: {since_display}")
        parts.append(f"confidence: {fact.confidence}")

    line = "- " + " | ".join(str(p) for p in parts)

    # Sub-lines
    sub_lines: List[str] = []
    if fact.reason:
        if fact.expired_at:
            sub_lines.append(f"  - reason: {fact.reason}")
        elif fact.reason.startswith("superseded: "):
            # Current facts store superseded info without the prefix
            sub_lines.append(f"  - superseded: {fact.reason[len('superseded: '):]}")
    if fact.source:
        sub_lines.append(f"  - source: {fact.source}")

    if sub_lines:
        return line + "\n" + "\n".join(sub_lines)
    return line


def _format_conflict_entry(conflict: dict) -> str:
    """
    Format a conflict dict as Markdown lines.

    - ⚠️ subject | predicate | object_a: "val_a" vs "val_b"
      - fact_a: mem_xxx
      - fact_b: mem_yyy
      - detected: 2026-03-05
      - status: unresolved
    """
    line = f"- ⚠️ {conflict['description']}"
    sub_lines = [
        f"  - fact_a: {conflict['fact_a_source']}",
        f"  - fact_b: {conflict['fact_b_source']}",
        f"  - detected: {conflict['detected']}",
        f"  - status: {conflict['status']}",
    ]
    return line + "\n" + "\n".join(sub_lines)


# ── Parsing ─────────────────────────────────────────────────────────────────


def _parse_fact_line(line: str) -> Optional[Fact]:
    """
    Parse a single fact line back into a Fact object.

    Handles:
    - "- subject | predicate | object | since: YYYY-MM | confidence: 0.95"
    - "- subject | predicate | object | YYYY-MM to YYYY-MM"
    """
    # Strip the leading "- "
    text = line.strip()
    if not text.startswith("- "):
        return None
    text = text[2:]

    parts = [p.strip() for p in text.split("|")]
    if len(parts) < 3:
        return None

    subject = parts[0]
    predicate = parts[1]
    object_val = parts[2]

    since = ""
    confidence = 1.0
    expired_at = ""

    for extra in parts[3:]:
        extra = extra.strip()
        # since: YYYY-MM
        m_since = re.match(r"^since:\s*(.+)$", extra)
        if m_since:
            since = m_since.group(1).strip()
            continue
        # confidence: 0.95
        m_conf = re.match(r"^confidence:\s*(.+)$", extra)
        if m_conf:
            try:
                confidence = float(m_conf.group(1).strip())
            except ValueError:
                pass
            continue
        # YYYY-MM to YYYY-MM  (expired range)
        m_range = re.match(r"^(\S+)\s+to\s+(\S+)$", extra)
        if m_range:
            since = m_range.group(1).strip()
            expired_at = m_range.group(2).strip()
            continue

    return Fact(
        subject=subject,
        predicate=predicate,
        object=object_val,
        since=since,
        confidence=confidence,
        expired_at=expired_at,
    )


def _parse_sub_lines(lines: List[str], start: int) -> dict:
    """
    Parse indented sub-lines (starting at index `start`) that belong to a
    preceding fact or conflict entry.

    Returns a dict of key -> value for lines matching "  - key: value".
    Stops at the first non-indented line or end of list.
    """
    result: dict = {}
    i = start
    while i < len(lines):
        ln = lines[i]
        # Sub-lines are indented with at least 2 spaces + "- "
        m = re.match(r"^\s{2,}-\s+(\S+?):\s*(.*)$", ln)
        if m:
            result[m.group(1)] = m.group(2).strip()
            i += 1
        else:
            break
    return result


def _parse_conflict_entry(primary_line: str, sub_data: dict) -> dict:
    """
    Parse a conflict entry from its primary line and sub-line data.

    Primary: - ⚠️ description text
    Sub-data keys: fact_a, fact_b, detected, status
    """
    text = primary_line.strip()
    # Remove leading "- ⚠️ " or "- ⚠️"
    text = re.sub(r"^-\s*⚠️\s*", "", text)

    return {
        "description": text,
        "fact_a_source": sub_data.get("fact_a", ""),
        "fact_b_source": sub_data.get("fact_b", ""),
        "detected": sub_data.get("detected", ""),
        "status": sub_data.get("status", "unresolved"),
    }


def parse_facts_file(project: str, facts_dir: Optional[str | Path] = None) -> dict:
    """
    Parse a project's facts file.

    Returns:
        {
            "current": [Fact, ...],
            "expired": [Fact, ...],
            "conflicts": [
                {
                    "description": str,
                    "fact_a_source": str,
                    "fact_b_source": str,
                    "detected": str,
                    "status": str,
                },
                ...
            ],
        }

    If the file doesn't exist, returns empty lists.
    """
    path = _facts_path(project, facts_dir)
    if not path.exists():
        return {"current": [], "expired": [], "conflicts": []}

    text = path.read_text(encoding="utf-8")

    # ── Strip YAML front matter ─────────────────────────────────────────
    body = text
    fm_match = re.match(r"^---\s*\n(.*?\n)---\s*\n", text, re.DOTALL)
    if fm_match:
        body = text[fm_match.end():]

    lines = body.split("\n")

    current_facts: List[Fact] = []
    expired_facts: List[Fact] = []
    conflicts: List[dict] = []

    section: Optional[str] = None  # "current", "expired", "conflicts"
    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()

        # Detect section headers
        if stripped.startswith("## Current"):
            section = "current"
            i += 1
            continue
        elif stripped.startswith("## Expired"):
            section = "expired"
            i += 1
            continue
        elif stripped.startswith("## Conflicts"):
            section = "conflicts"
            i += 1
            continue
        elif stripped.startswith("# "):
            # Top-level heading (e.g. "# Facts: project") — skip
            i += 1
            continue

        # Skip blank lines
        if not stripped:
            i += 1
            continue

        # Process entries based on current section
        if section in ("current", "expired") and stripped.startswith("- "):
            # Only process non-indented lines (sub-lines start with spaces)
            if not ln[0].isspace():
                fact = _parse_fact_line(stripped)
                if fact:
                    # Parse sub-lines
                    sub = _parse_sub_lines(lines, i + 1)
                    if "source" in sub:
                        fact.source = sub["source"]
                    if "reason" in sub:
                        fact.reason = sub["reason"]
                    if "superseded" in sub:
                        # "superseded" is informational, stored in reason
                        # for current facts
                        if not fact.reason:
                            fact.reason = f"superseded: {sub['superseded']}"

                    if section == "current":
                        current_facts.append(fact)
                    else:
                        expired_facts.append(fact)

                    # Advance past sub-lines
                    i += 1
                    while i < len(lines):
                        m = re.match(r"^\s{2,}-\s+", lines[i])
                        if m:
                            i += 1
                        else:
                            break
                    continue

        elif section == "conflicts" and stripped.startswith("- ⚠️"):
            # Conflict entry
            sub = _parse_sub_lines(lines, i + 1)
            conflict = _parse_conflict_entry(stripped, sub)
            conflicts.append(conflict)

            # Advance past sub-lines
            i += 1
            while i < len(lines):
                m = re.match(r"^\s{2,}-\s+", lines[i])
                if m:
                    i += 1
                else:
                    break
            continue

        i += 1

    return {"current": current_facts, "expired": expired_facts, "conflicts": conflicts}


# ── Writing ─────────────────────────────────────────────────────────────────


def write_facts_file(project: str, data: dict, facts_dir: Optional[str | Path] = None) -> Path:
    """
    Write (overwrite) a project's facts file.

    Args:
        project: Project ID
        data: Dict with "current", "expired", "conflicts" keys
              (same format as parse_facts_file output)
        facts_dir: Override directory

    Returns:
        Path to written file
    """
    path = _facts_path(project, facts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    current: List[Fact] = data.get("current", [])
    expired: List[Fact] = data.get("expired", [])
    conflicts: List[dict] = data.get("conflicts", [])

    today = datetime.now().strftime("%Y-%m-%d")

    # ── Build the file ──────────────────────────────────────────────────
    lines: List[str] = []

    # YAML front matter
    lines.append("---")
    lines.append(f"project: {project}")
    lines.append(f"last_updated: {today}")
    lines.append("---")
    lines.append("")

    # Title
    lines.append(f"# Facts: {project}")
    lines.append("")

    # ## Current
    lines.append("## Current")
    lines.append("")
    for fact in current:
        lines.append(_format_fact_line(fact, expired=False))
    lines.append("")

    # ## Expired
    lines.append("## Expired")
    lines.append("")
    for fact in expired:
        lines.append(_format_fact_line(fact, expired=True))
    lines.append("")

    # ## Conflicts
    lines.append("## Conflicts")
    lines.append("")
    for conflict in conflicts:
        lines.append(_format_conflict_entry(conflict))
    lines.append("")

    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")
    return path


# ── Public API ──────────────────────────────────────────────────────────────


def add_fact(
    project: str,
    subject: str,
    predicate: str,
    object_val: str,
    confidence: float = 1.0,
    source_memory_id: str = "",
    since: str = "",
    source_text: str = "",
    facts_dir: Optional[str | Path] = None,
    exclusive_predicates: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Add a new fact to a project's facts file.
    Automatically checks for conflicts and resolves when possible.

    Args:
        project: Project ID
        subject: Fact subject (entity name)
        predicate: Fact predicate (relationship)
        object_val: Fact object (value)
        confidence: 0.0-1.0
        source_memory_id: ID of the memory that produced this fact
        since: ISO date string for when this fact became true
        source_text: Original text (used for conflict classification)
        facts_dir: Override directory
        exclusive_predicates: Override predicate rules

    Returns:
        {
            "added": bool,
            "conflict": Conflict or None,
            "resolution": dict or None,
        }
    """
    data = parse_facts_file(project, facts_dir)
    preds = exclusive_predicates if exclusive_predicates is not None else DEFAULT_EXCLUSIVE_PREDICATES

    new_fact = Fact(
        subject=subject,
        predicate=predicate,
        object=object_val,
        confidence=confidence,
        source=source_memory_id,
        since=since or datetime.now().strftime("%Y-%m"),
    )

    # Check for duplicate — same subject/predicate/object already current
    for existing in data["current"]:
        if (
            existing.subject.lower().strip() == subject.lower().strip()
            and existing.predicate.lower().strip() == predicate.lower().strip()
            and existing.object.lower().strip() == object_val.lower().strip()
        ):
            # Exact duplicate — update confidence/source if higher, but no conflict
            if confidence > existing.confidence:
                existing.confidence = confidence
            if source_memory_id and not existing.source:
                existing.source = source_memory_id
            write_facts_file(project, data, facts_dir)
            return {"added": False, "conflict": None, "resolution": None}

    # Check for conflicts against current facts
    # Pass source_text as the fact's source for conflict classification
    fact_for_check = Fact(
        subject=subject,
        predicate=predicate,
        object=object_val,
        confidence=confidence,
        source=source_text,  # classify_conflict reads source from fact
        since=since or datetime.now().strftime("%Y-%m"),
    )
    conflict = check_conflict(fact_for_check, data["current"], preds)

    if conflict is None:
        # No conflict — just add the fact
        data["current"].append(new_fact)
        write_facts_file(project, data, facts_dir)
        return {"added": True, "conflict": None, "resolution": None}

    # Conflict detected — attempt resolution
    resolution = resolve_conflict(conflict)

    if resolution["resolved"]:
        # Apply resolution: expire old fact, add new fact
        old = conflict.old_fact

        # Apply old_fact_updates
        old_updates = resolution.get("old_fact_updates", {})
        if "expired_at" in old_updates:
            old.expired_at = old_updates["expired_at"]
        if "reason" in old_updates:
            old.reason = old_updates["reason"]
        if "confidence" in old_updates:
            old.confidence = old_updates["confidence"]

        # Apply new_fact_updates
        new_updates = resolution.get("new_fact_updates", {})
        if "confidence" in new_updates:
            new_fact.confidence = new_updates["confidence"]
        if "reason" in new_updates:
            new_fact.reason = new_updates["reason"]

        # Move old fact from current to expired
        data["current"] = [f for f in data["current"] if f is not old]
        data["expired"].append(old)

        # Add new fact to current
        data["current"].append(new_fact)

        write_facts_file(project, data, facts_dir)
        return {"added": True, "conflict": conflict, "resolution": resolution}
    else:
        # Hard contradiction — defer to user
        # Apply confidence downgrades
        old = conflict.old_fact
        old_updates = resolution.get("old_fact_updates", {})
        if "confidence" in old_updates:
            old.confidence = old_updates["confidence"]

        new_updates = resolution.get("new_fact_updates", {})
        if "confidence" in new_updates:
            new_fact.confidence = new_updates["confidence"]

        # Add new fact to current (both stay current, both at 0.5)
        data["current"].append(new_fact)

        # Add conflict entry
        detected = conflict.detected_at[:10] if conflict.detected_at else datetime.now().strftime("%Y-%m-%d")
        desc = (
            f"{conflict.old_fact.subject} | {conflict.old_fact.predicate} | "
            f'"{conflict.old_fact.object}" vs "{conflict.new_fact.object}"'
        )
        conflict_entry = {
            "description": desc,
            "fact_a_source": conflict.old_fact.source or "",
            "fact_b_source": source_memory_id,
            "detected": detected,
            "status": "unresolved",
        }
        data["conflicts"].append(conflict_entry)

        write_facts_file(project, data, facts_dir)
        return {"added": True, "conflict": conflict, "resolution": resolution}


def expire_fact(
    project: str,
    subject: str,
    predicate: str,
    object_val: str,
    reason: str = "",
    superseded_by: str = "",
    facts_dir: Optional[str | Path] = None,
) -> bool:
    """
    Manually expire a fact. Moves it from Current to Expired.

    Returns: True if fact was found and expired, False if not found
    """
    data = parse_facts_file(project, facts_dir)
    found: Optional[Fact] = None

    for fact in data["current"]:
        if (
            fact.subject.lower().strip() == subject.lower().strip()
            and fact.predicate.lower().strip() == predicate.lower().strip()
            and fact.object.lower().strip() == object_val.lower().strip()
        ):
            found = fact
            break

    if found is None:
        return False

    now = datetime.now().isoformat()
    found.expired_at = now
    if reason:
        found.reason = reason
    elif superseded_by:
        found.reason = f"superseded by {superseded_by}"

    data["current"] = [f for f in data["current"] if f is not found]
    data["expired"].append(found)

    write_facts_file(project, data, facts_dir)
    return True


def get_active_facts(project: str, facts_dir: Optional[str | Path] = None) -> List[Fact]:
    """Get all current (non-expired) facts for a project."""
    data = parse_facts_file(project, facts_dir)
    return data["current"]


def get_facts_for_entity(
    entity: str,
    project: Optional[str] = None,
    facts_dir: Optional[str | Path] = None,
) -> List[Fact]:
    """
    Get all active facts where the subject matches the entity.
    If project is None, scan ALL facts files.
    """
    d = Path(facts_dir) if facts_dir else _default_facts_dir()
    entity_lower = entity.lower().strip()

    if project is not None:
        data = parse_facts_file(project, facts_dir)
        return [f for f in data["current"] if f.subject.lower().strip() == entity_lower]

    # Scan all facts files
    results: List[Fact] = []
    if not d.exists():
        return results

    for md_file in sorted(d.glob("*.md")):
        proj = md_file.stem
        try:
            data = parse_facts_file(proj, facts_dir)
        except Exception:
            continue
        for fact in data["current"]:
            if fact.subject.lower().strip() == entity_lower:
                results.append(fact)

    return results


def get_unresolved_conflicts(
    project: Optional[str] = None,
    facts_dir: Optional[str | Path] = None,
) -> List[dict]:
    """
    Get all unresolved conflicts.
    If project is None, scan ALL facts files.
    """
    d = Path(facts_dir) if facts_dir else _default_facts_dir()

    if project is not None:
        data = parse_facts_file(project, facts_dir)
        return [c for c in data["conflicts"] if c.get("status") == "unresolved"]

    # Scan all facts files
    results: List[dict] = []
    if not d.exists():
        return results

    for md_file in sorted(d.glob("*.md")):
        proj = md_file.stem
        try:
            data = parse_facts_file(proj, facts_dir)
        except Exception:
            continue
        for c in data["conflicts"]:
            if c.get("status") == "unresolved":
                c_with_project = dict(c)
                c_with_project["project"] = proj
                results.append(c_with_project)

    return results


def resolve_conflict_manual(
    project: str,
    conflict_index: int,
    winner: str,  # "a" or "b"
    facts_dir: Optional[str | Path] = None,
) -> bool:
    """
    Manually resolve a conflict.

    Args:
        conflict_index: Index into the conflicts list
        winner: "a" keeps fact_a, "b" keeps fact_b

    Returns: True if resolved
    """
    data = parse_facts_file(project, facts_dir)

    if conflict_index < 0 or conflict_index >= len(data["conflicts"]):
        return False

    conflict = data["conflicts"][conflict_index]
    if conflict.get("status") != "unresolved":
        return False

    # Parse the conflict description to find the two competing values
    # Format: subject | predicate | "val_a" vs "val_b"
    desc = conflict["description"]
    m = re.match(r'^(.+?)\s*\|\s*(.+?)\s*\|\s*"(.+?)"\s+vs\s+"(.+?)"$', desc)

    if not m:
        # Can't parse — just mark as resolved
        conflict["status"] = f"resolved ({winner})"
        write_facts_file(project, data, facts_dir)
        return True

    subject = m.group(1).strip()
    predicate = m.group(2).strip()
    val_a = m.group(3).strip()
    val_b = m.group(4).strip()

    loser_val = val_b if winner == "a" else val_a
    winner_val = val_a if winner == "a" else val_b

    # Expire the loser from current facts
    now = datetime.now().isoformat()
    new_current: List[Fact] = []
    for fact in data["current"]:
        if (
            fact.subject.lower().strip() == subject.lower().strip()
            and fact.predicate.lower().strip() == predicate.lower().strip()
            and fact.object.lower().strip() == loser_val.lower().strip()
        ):
            fact.expired_at = now
            fact.reason = f"manually resolved: {winner_val} chosen over {loser_val}"
            data["expired"].append(fact)
        else:
            new_current.append(fact)
    data["current"] = new_current

    # Restore confidence on the winner
    for fact in data["current"]:
        if (
            fact.subject.lower().strip() == subject.lower().strip()
            and fact.predicate.lower().strip() == predicate.lower().strip()
            and fact.object.lower().strip() == winner_val.lower().strip()
        ):
            fact.confidence = 1.0

    # Mark conflict as resolved
    conflict["status"] = f"resolved ({winner})"

    write_facts_file(project, data, facts_dir)
    return True
