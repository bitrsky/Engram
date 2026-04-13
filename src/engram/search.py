"""
search.py — Semantic search for Engram.

Combines vector search (ChromaDB) with facts and conflict awareness.
Search results include related facts and flag any unresolved conflicts.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .config import EngramConfig
from .index import IndexManager, SearchHit
from .facts import get_facts_for_entity, get_unresolved_conflicts


@dataclass
class EnrichedHit:
    """Search result enriched with facts and conflict info."""
    id: str
    content: str
    similarity: float
    project: str
    topics: list
    memory_type: str
    importance: float
    created: str
    file_path: str
    related_facts: list = field(default_factory=list)   # List of Fact objects
    conflicts: list = field(default_factory=list)        # List of conflict dicts


@dataclass
class SearchResults:
    """Container for search results with metadata."""
    hits: List[EnrichedHit]
    query: str
    project: str = None
    total_facts_found: int = 0
    unresolved_conflicts: int = 0


def search(
    query: str,
    index_manager: IndexManager,
    project: str = None,
    topics: List[str] = None,
    memory_type: str = None,
    n: int = 5,
    include_facts: bool = True,
    include_conflicts: bool = True,
    config: EngramConfig = None,
    think_fn=None,
) -> SearchResults:
    """
    Perform semantic search with fact and conflict enrichment.

    Flow:
    1. Query rewrite (optional, via think_fn)
    2. Vector search via IndexManager (with optional reranking)
    3. Update access stats for each hit
    4. Temporal reasoning (optional, via think_fn)
    5. For each hit, find related facts (by extracting entities from content)
    6. Check for unresolved conflicts
    7. Assemble enriched results

    Args:
        query: Search query text
        index_manager: IndexManager instance
        project: Filter by project
        topics: Filter by topics (ANY match)
        memory_type: Filter by type
        n: Max results
        include_facts: Whether to enrich with related facts
        include_conflicts: Whether to check for conflicts
        config: Configuration
        think_fn: Optional agent thinking function (see engram.llm.ThinkFn)

    Returns:
        SearchResults with enriched hits
    """
    config = config or EngramConfig()

    # Step 0: Query rewrite (optional)
    original_query = query
    if think_fn is not None and config.query_rewrite_enabled:
        try:
            from .llm import rewrite_query
            query = rewrite_query(query, think_fn)
        except Exception:
            pass  # Use original query

    # Step 1: Vector search (with optional reranking)
    if config.rerank_enabled:
        raw_hits = index_manager.vector_search_reranked(
            query=query,
            config=config,
            project=project,
            topics=topics,
            memory_type=memory_type,
            n=n,
            think_fn=think_fn,
        )
    else:
        raw_hits = index_manager.vector_search(
            query=query,
            project=project,
            topics=topics,
            memory_type=memory_type,
            n=n,
        )

    # Step 2: Update access stats
    for hit in raw_hits:
        try:
            index_manager.update_access_stats(hit.id)
        except Exception:
            pass  # Non-critical

    # Step 2.5: Temporal reasoning (optional)
    temporal_answer = None
    if think_fn is not None and config.temporal_reasoning_enabled and raw_hits:
        try:
            from .llm import answer_temporal
            temporal_answer = answer_temporal(original_query, raw_hits, think_fn)
        except Exception:
            pass

    # Step 3: Enrich with facts
    enriched = []
    total_facts = 0

    for hit in raw_hits:
        enriched_hit = EnrichedHit(
            id=hit.id,
            content=hit.content,
            similarity=hit.similarity,
            project=hit.project,
            topics=hit.topics,
            memory_type=hit.memory_type,
            importance=hit.importance,
            created=hit.created,
            file_path=hit.file_path,
        )

        if include_facts and hit.project:
            # Find facts related to entities mentioned in this memory
            related = _find_related_facts(hit.content, hit.project, config)
            enriched_hit.related_facts = related
            total_facts += len(related)

        enriched.append(enriched_hit)

    # Attach temporal reasoning answer to first hit (if available)
    if temporal_answer and enriched:
        enriched[0].content = (
            enriched[0].content.rstrip()
            + f"\n\n[Temporal Reasoning]\n{temporal_answer}"
        )

    # Step 4: Check conflicts
    total_conflicts = 0
    if include_conflicts:
        conflicts = get_unresolved_conflicts(
            project=project,
            facts_dir=config.facts_dir,
        )
        total_conflicts = len(conflicts)

        # Attach relevant conflicts to hits
        for hit in enriched:
            for c in conflicts:
                # Check if this conflict's entities are mentioned in the hit content
                if _conflict_relates_to(c, hit.content):
                    hit.conflicts.append(c)

    return SearchResults(
        hits=enriched,
        query=query,
        project=project,
        total_facts_found=total_facts,
        unresolved_conflicts=total_conflicts,
    )


def _find_related_facts(content: str, project: str, config: EngramConfig) -> list:
    """
    Extract entity names from content and find their facts.

    Simple approach:
    - Extract capitalized words/phrases as potential entities
    - Also try the project name itself
    - Look up each in the facts file
    """
    related = []
    seen_keys = set()  # (subject, predicate, object) to deduplicate

    entities = _extract_potential_entities(content)
    # Always include the project name as an entity to check
    entities.append(project)

    for entity in entities:
        try:
            facts = get_facts_for_entity(
                entity=entity,
                project=project,
                facts_dir=config.facts_dir,
            )
            for fact in facts:
                key = (
                    fact.subject.lower().strip(),
                    fact.predicate.lower().strip(),
                    fact.object.lower().strip(),
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    related.append(fact)
        except Exception:
            # Defensive: if facts file is malformed or missing, skip
            continue

    return related


def _conflict_relates_to(conflict: dict, content: str) -> bool:
    """
    Check if a conflict's entities are mentioned in content.
    Simple case-insensitive substring matching.
    """
    content_lower = content.lower()
    description = conflict.get("description", "")

    # Parse the conflict description to extract entity names.
    # Format: "subject | predicate | \"val_a\" vs \"val_b\""
    parts = [p.strip() for p in description.split("|")]

    for part in parts:
        # Clean up quotes and "vs" fragments
        cleaned = part.strip().strip('"').strip("'")
        # Skip very short tokens (predicates like "db") to avoid false positives
        if len(cleaned) < 2:
            continue
        # For "val_a" vs "val_b" parts, check each value separately
        if " vs " in cleaned:
            vs_parts = cleaned.split(" vs ")
            for vp in vs_parts:
                vp_clean = vp.strip().strip('"').strip("'")
                if len(vp_clean) >= 2 and vp_clean.lower() in content_lower:
                    return True
        elif cleaned.lower() in content_lower:
            return True

    return False


def _extract_potential_entities(text: str) -> List[str]:
    """
    Extract potential entity names from text.

    Heuristics:
    - Capitalized words not at sentence start (e.g., "MongoDB", "Clerk")
    - CamelCase words (e.g., "FastAPI", "TypeScript")
    - Quoted strings
    - Known tech terms
    """
    entities = []
    seen = set()

    def _add(name: str):
        key = name.lower().strip()
        if key and key not in seen and len(key) >= 2:
            seen.add(key)
            entities.append(name.strip())

    # 1. Quoted strings (single or double quotes)
    for m in re.finditer(r'["\']([A-Za-z][A-Za-z0-9 _-]{0,40})["\']', text):
        _add(m.group(1))

    # 2. CamelCase / PascalCase words (e.g., FastAPI, TypeScript, MongoDB)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b', text):
        _add(m.group(1))

    # 3. ALL-CAPS abbreviations followed by lowercase or at word boundary
    #    (e.g., "API", "SQL", "AWS") — at least 2 chars
    for m in re.finditer(r'\b([A-Z]{2,8})\b', text):
        _add(m.group(1))

    # 4. Capitalized words that appear mid-sentence (not after ". " or at start)
    #    Split into sentences, then look for capitalised words that aren't the
    #    first word of a sentence.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        words = sentence.split()
        for word in words[1:]:  # skip first word of sentence
            # Strip trailing punctuation
            clean = re.sub(r'[^A-Za-z0-9]', '', word)
            if clean and clean[0].isupper() and len(clean) >= 2:
                _add(clean)

    # 5. Tech-style names with mixed case/numbers at start (e.g., "Auth0", "S3")
    for m in re.finditer(r'\b([A-Z][a-zA-Z]*\d+[a-zA-Z]*)\b', text):
        _add(m.group(1))

    return entities


def format_search_results(results: SearchResults, max_width: int = 80) -> str:
    """
    Format search results as a human-readable string.

    Example:

    🔍 Search: "auth provider" (project: saas-app)

    1. [0.94] Auth Provider Decision (2026-01-15)
       We decided to use Clerk instead of Auth0...
       📌 Facts: saas-app→auth→Clerk (since 2026-01)

    2. [0.87] Auth Migration Notes (2026-02-20)
       Maya started the Clerk migration...

    ⚠️ 1 unresolved conflict in this project
    """
    lines = []

    # Header
    project_part = f" (project: {results.project})" if results.project else ""
    lines.append(f'🔍 Search: "{results.query}"{project_part}')
    lines.append("")

    if not results.hits:
        lines.append("  No results found.")
        return "\n".join(lines)

    for i, hit in enumerate(results.hits, 1):
        # Title line: index, similarity, id (used as title), date
        date_part = f" ({hit.created})" if hit.created else ""
        # Use the memory id as the display title (replace underscores/hyphens)
        title = hit.id.replace("_", " ").replace("-", " ")
        lines.append(f"{i}. [{hit.similarity:.2f}] {title}{date_part}")

        # Truncated content preview
        content_preview = hit.content.replace("\n", " ").strip()
        if len(content_preview) > 100:
            content_preview = content_preview[:100] + "..."
        lines.append(f"   {content_preview}")

        # Related facts
        if hit.related_facts:
            fact_strs = []
            for fact in hit.related_facts:
                since_part = f" (since {fact.since})" if fact.since else ""
                fact_strs.append(
                    f"{fact.subject}→{fact.predicate}→{fact.object}{since_part}"
                )
            # Show up to 3 facts inline, rest as count
            shown = fact_strs[:3]
            remaining = len(fact_strs) - 3
            facts_line = "; ".join(shown)
            if remaining > 0:
                facts_line += f" (+{remaining} more)"
            lines.append(f"   📌 Facts: {facts_line}")

        # Conflicts attached to this hit
        if hit.conflicts:
            for c in hit.conflicts:
                lines.append(f"   ⚠️ Conflict: {c.get('description', '?')}")

        lines.append("")

    # Summary footer
    if results.total_facts_found > 0:
        lines.append(f"📊 {results.total_facts_found} related fact(s) found")

    if results.unresolved_conflicts > 0:
        project_scope = f" in project '{results.project}'" if results.project else ""
        lines.append(
            f"⚠️ {results.unresolved_conflicts} unresolved conflict(s){project_scope}"
        )

    return "\n".join(lines)
