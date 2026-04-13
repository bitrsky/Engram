"""
index.py — Dual index manager for Engram (ChromaDB + SQLite).

The index is a DERIVED ARTIFACT from Markdown files.
It can be deleted and rebuilt at any time via rebuild().
Source of truth is always the .md files in memories/.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

from .store import parse_frontmatter


@dataclass
class SearchHit:
    """A single search result."""
    id: str
    content: str
    similarity: float  # 0.0 to 1.0 (1.0 = most similar)
    project: str
    topics: list
    memory_type: str
    importance: float
    created: str
    file_path: str


SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_index (
    id TEXT PRIMARY KEY,
    project TEXT,
    topics TEXT,
    memory_type TEXT,
    importance REAL,
    created TEXT,
    file_path TEXT,
    content_hash TEXT,
    access_count INTEGER DEFAULT 0,
    last_accessed TEXT,
    indexed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_mi_project ON memory_index(project);
CREATE INDEX IF NOT EXISTS idx_mi_created ON memory_index(created DESC);
CREATE INDEX IF NOT EXISTS idx_mi_importance ON memory_index(importance DESC);
CREATE INDEX IF NOT EXISTS idx_mi_content_hash ON memory_index(content_hash);

CREATE TABLE IF NOT EXISTS index_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

CHROMA_COLLECTION_NAME = "engram_memories"


def _content_hash(text: str) -> str:
    """Compute sha256 hex digest of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IndexManager:
    """
    Manages ChromaDB vector index + SQLite metadata index.

    Both indexes are derived from Markdown files and can be rebuilt.
    ChromaDB stores: id, document text, embedding, metadata (project, topics, memory_type, importance, created, file_path)
    SQLite stores: structured metadata for non-semantic queries (filter by project, sort by date, etc.)
    """

    def __init__(
        self,
        index_dir: str | Path,
        memories_dir: Optional[str | Path] = None,
        facts_dir: Optional[str | Path] = None,
        projects_dir: Optional[str | Path] = None,
    ):
        """
        Args:
            index_dir: Path to .index/ directory (contains vectors.chroma/ and meta.sqlite3)
            memories_dir: Path to memories/ directory (for rebuild)
            facts_dir: Path to facts/ directory (for rebuild)
            projects_dir: Path to projects/ directory (for rebuild)
        """
        self._index_dir = Path(index_dir)
        self._memories_dir = Path(memories_dir) if memories_dir else None
        self._facts_dir = Path(facts_dir) if facts_dir else None
        self._projects_dir = Path(projects_dir) if projects_dir else None
        self._index_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        chroma_path = str(self._index_dir / "vectors.chroma")
        self._chroma_client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize SQLite
        self._db_path = self._index_dir / "meta.sqlite3"
        self._conn: sqlite3.Connection | None = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SQLITE_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _db(self) -> sqlite3.Connection:
        """Return the active SQLite connection (raises if closed)."""
        assert self._conn is not None, "IndexManager already closed"
        return self._conn

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close database connections."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None  # type: ignore[assignment]

    def __del__(self):
        """Safety net — close connections on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Single-file indexing
    # ------------------------------------------------------------------

    def index_memory(self, filepath: str | Path) -> str:
        """
        Index a single memory file.
        Parses frontmatter + body, writes to ChromaDB + SQLite.

        Returns: memory id
        """
        filepath = Path(filepath)
        meta, body = parse_frontmatter(filepath)
        body = body.strip()

        memory_id = meta.get("id", filepath.stem)
        project = meta.get("project") or ""
        topics = meta.get("topics", []) or []
        memory_type = meta.get("memory_type", "note") or "note"
        importance = float(meta.get("importance", 3.0) or 3.0)
        created = str(meta.get("created", "")) or ""
        file_path_str = str(filepath.resolve())
        chash = _content_hash(body)
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # --- ChromaDB ---
        # Metadata values must be str/int/float/bool — store topics as JSON.
        chroma_meta = {
            "project": project,
            "topics": json.dumps(topics),
            "memory_type": memory_type,
            "importance": importance,
            "created": created,
            "file_path": file_path_str,
        }

        # Upsert handles both insert and update.
        self._collection.upsert(
            ids=[memory_id],
            documents=[body],
            metadatas=[chroma_meta],
        )

        # --- SQLite ---
        self._db.execute(
            """
            INSERT INTO memory_index
                (id, project, topics, memory_type, importance, created,
                 file_path, content_hash, access_count, last_accessed, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)
            ON CONFLICT(id) DO UPDATE SET
                project      = excluded.project,
                topics       = excluded.topics,
                memory_type  = excluded.memory_type,
                importance   = excluded.importance,
                created      = excluded.created,
                file_path    = excluded.file_path,
                content_hash = excluded.content_hash,
                indexed_at   = excluded.indexed_at
            """,
            (
                memory_id,
                project,
                json.dumps(topics),
                memory_type,
                importance,
                created,
                file_path_str,
                chash,
                now_iso,
            ),
        )
        self._db.commit()

        return memory_id

    def index_facts_file(self, filepath: str | Path) -> int:
        """
        Index all current facts from a single facts file.

        Parses ``facts/{project}.md``, deletes old fact entries for that project,
        then indexes each current fact as a separate document.

        Returns: number of facts indexed.
        """
        from .facts import parse_facts_file  # avoid circular import

        filepath = Path(filepath)
        if not filepath.exists():
            return 0

        project = filepath.stem  # facts/saas-app.md → "saas-app"
        data = parse_facts_file(project, facts_dir=filepath.parent)
        current_facts = data.get("current", [])

        # ── Remove old fact entries for this project ──────────────────────
        old_rows = self._db.execute(
            "SELECT id FROM memory_index WHERE project = ? AND memory_type = 'fact'",
            (project,),
        ).fetchall()
        for row in old_rows:
            try:
                self._collection.delete(ids=[row["id"]])
            except Exception:
                pass
            self._db.execute(
                "DELETE FROM memory_index WHERE id = ?", (row["id"],)
            )
        self._db.commit()

        # ── Index each current fact ───────────────────────────────────────
        file_path_str = str(filepath.resolve())
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        count = 0

        for fact in current_facts:
            # Build a stable ID from the triple
            triple_str = f"{fact.subject}|{fact.predicate}|{fact.object}"
            fact_hash = hashlib.md5(triple_str.encode()).hexdigest()[:12]
            fact_id = f"fact_{project}_{fact_hash}"

            # Natural-language document for vector search
            # Humanize the predicate: "uses_database" → "uses database"
            pred_text = fact.predicate.replace("_", " ")
            document = f"{fact.subject} {pred_text} {fact.object}"

            since = fact.since or ""
            source = fact.source or ""

            chroma_meta = {
                "project": project,
                "topics": json.dumps([]),
                "memory_type": "fact",
                "importance": 5.0,
                "created": since,
                "file_path": file_path_str,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "source": source,
            }

            self._collection.upsert(
                ids=[fact_id],
                documents=[document],
                metadatas=[chroma_meta],
            )

            chash = _content_hash(document)
            self._db.execute(
                """
                INSERT INTO memory_index
                    (id, project, topics, memory_type, importance, created,
                     file_path, content_hash, access_count, last_accessed, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)
                ON CONFLICT(id) DO UPDATE SET
                    project      = excluded.project,
                    topics       = excluded.topics,
                    memory_type  = excluded.memory_type,
                    importance   = excluded.importance,
                    created      = excluded.created,
                    file_path    = excluded.file_path,
                    content_hash = excluded.content_hash,
                    indexed_at   = excluded.indexed_at
                """,
                (
                    fact_id,
                    project,
                    json.dumps([]),
                    "fact",
                    5.0,
                    since,
                    file_path_str,
                    chash,
                    now_iso,
                ),
            )
            count += 1

        self._db.commit()
        return count

    def index_project_file(self, filepath: str | Path) -> str:
        """
        Index a single project file.

        Parses ``projects/{id}.md`` frontmatter and indexes the project
        metadata as a searchable document.

        Returns: project id.
        """
        filepath = Path(filepath)
        meta, body = parse_frontmatter(filepath)

        project_id = meta.get("id", filepath.stem)
        display_name = meta.get("display_name", project_id)
        status = meta.get("status", "active")
        description = meta.get("description", "")
        aliases = meta.get("aliases", []) or []
        tags = meta.get("tags", []) or []
        created = str(meta.get("created", "")) or ""
        file_path_str = str(filepath.resolve())

        doc_id = f"project_{project_id}"

        # Build a searchable document from all project metadata
        parts = [f"Project: {display_name}"]
        if description:
            parts.append(f"Description: {description}")
        parts.append(f"Status: {status}")
        if aliases:
            parts.append(f"Aliases: {', '.join(aliases)}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if body.strip():
            parts.append(body.strip())
        document = ". ".join(parts)

        chroma_meta = {
            "project": project_id,
            "topics": json.dumps(tags),
            "memory_type": "project",
            "importance": 4.0,
            "created": created,
            "file_path": file_path_str,
        }

        self._collection.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[chroma_meta],
        )

        chash = _content_hash(document)
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self._db.execute(
            """
            INSERT INTO memory_index
                (id, project, topics, memory_type, importance, created,
                 file_path, content_hash, access_count, last_accessed, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)
            ON CONFLICT(id) DO UPDATE SET
                project      = excluded.project,
                topics       = excluded.topics,
                memory_type  = excluded.memory_type,
                importance   = excluded.importance,
                created      = excluded.created,
                file_path    = excluded.file_path,
                content_hash = excluded.content_hash,
                indexed_at   = excluded.indexed_at
            """,
            (
                doc_id,
                project_id,
                json.dumps(tags),
                "project",
                4.0,
                created,
                file_path_str,
                chash,
                now_iso,
            ),
        )
        self._db.commit()

        return project_id

    def remove_from_index(self, memory_id: str) -> bool:
        """Remove a memory from both indexes. Returns True if found."""
        # Check existence in SQLite first.
        row = self._db.execute(
            "SELECT id FROM memory_index WHERE id = ?", (memory_id,)
        ).fetchone()

        found = row is not None

        # Remove from ChromaDB (silently ignores missing ids).
        try:
            self._collection.delete(ids=[memory_id])
        except Exception:
            pass

        # Remove from SQLite.
        self._db.execute(
            "DELETE FROM memory_index WHERE id = ?", (memory_id,)
        )
        self._db.commit()

        return found

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def rebuild(
        self,
        memories_dir: Optional[str | Path] = None,
        facts_dir: Optional[str | Path] = None,
        projects_dir: Optional[str | Path] = None,
    ) -> int:
        """
        Full rebuild: clear both indexes, scan all memories/*.md, facts/*.md,
        and projects/*.md, re-index everything.

        Returns: total number of entries indexed
        """
        mem_dir = Path(memories_dir) if memories_dir else self._memories_dir
        if mem_dir is None:
            raise ValueError(
                "memories_dir must be provided either in constructor or rebuild()"
            )
        f_dir = Path(facts_dir) if facts_dir else self._facts_dir
        p_dir = Path(projects_dir) if projects_dir else self._projects_dir

        # Clear ChromaDB collection.
        self._chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
        self._collection = self._chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Clear SQLite tables.
        self._db.execute("DELETE FROM memory_index")
        self._db.execute("DELETE FROM index_meta")
        self._db.commit()

        count = 0

        # ── Memories ──────────────────────────────────────────────────────
        if mem_dir and mem_dir.is_dir():
            for md_file in sorted(mem_dir.glob("*.md")):
                try:
                    meta, _ = parse_frontmatter(md_file)
                    if "id" not in meta:
                        continue
                    self.index_memory(md_file)
                    count += 1
                except Exception:
                    continue

        # ── Facts ─────────────────────────────────────────────────────────
        if f_dir and f_dir.is_dir():
            for md_file in sorted(f_dir.glob("*.md")):
                try:
                    count += self.index_facts_file(md_file)
                except Exception:
                    continue

        # ── Projects ──────────────────────────────────────────────────────
        if p_dir and p_dir.is_dir():
            for md_file in sorted(p_dir.glob("*.md")):
                try:
                    meta, _ = parse_frontmatter(md_file)
                    if "id" not in meta:
                        continue
                    self.index_project_file(md_file)
                    count += 1
                except Exception:
                    continue

        # Record rebuild timestamp.
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self._db.execute(
            """
            INSERT INTO index_meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            ("last_rebuild_time", now_iso),
        )
        self._db.commit()

        return count

    def incremental_update(
        self,
        memories_dir: Optional[str | Path] = None,
        facts_dir: Optional[str | Path] = None,
        projects_dir: Optional[str | Path] = None,
    ) -> int:
        """
        Incremental update: only process files modified since last index time.
        Also removes index entries for files that no longer exist.
        Scans memories/, facts/, and projects/ directories.

        Returns: number of entries updated
        """
        mem_dir = Path(memories_dir) if memories_dir else self._memories_dir
        if mem_dir is None:
            raise ValueError(
                "memories_dir must be provided either in constructor or incremental_update()"
            )
        f_dir = Path(facts_dir) if facts_dir else self._facts_dir
        p_dir = Path(projects_dir) if projects_dir else self._projects_dir

        # Get the last rebuild/update timestamp.
        row = self._db.execute(
            "SELECT value FROM index_meta WHERE key = ?",
            ("last_rebuild_time",),
        ).fetchone()

        last_rebuild_iso = row["value"] if row else None
        if last_rebuild_iso:
            try:
                last_rebuild_dt = datetime.fromisoformat(last_rebuild_iso)
                last_rebuild_ts = last_rebuild_dt.timestamp()
            except (ValueError, TypeError):
                last_rebuild_ts = 0.0
        else:
            last_rebuild_ts = 0.0

        # --- Remove stale entries (files that no longer exist) ---
        existing_rows = self._db.execute(
            "SELECT id, file_path FROM memory_index"
        ).fetchall()

        stale_ids = []
        for r in existing_rows:
            if not Path(r["file_path"]).exists():
                stale_ids.append(r["id"])

        for sid in stale_ids:
            self.remove_from_index(sid)

        # --- Build indexed file map for mtime comparison ---
        indexed_rows = self._db.execute(
            "SELECT file_path, indexed_at FROM memory_index"
        ).fetchall()
        indexed_map: Dict[str, str] = {
            r["file_path"]: (r["indexed_at"] or "") for r in indexed_rows
        }

        count = 0

        def _needs_reindex(md_file: Path) -> bool:
            """Return True if the file is new or modified since last index."""
            resolved = str(md_file.resolve())
            file_mtime = md_file.stat().st_mtime
            if resolved in indexed_map:
                indexed_at_str = indexed_map[resolved]
                if indexed_at_str:
                    try:
                        indexed_at_ts = datetime.fromisoformat(
                            indexed_at_str
                        ).timestamp()
                    except (ValueError, TypeError):
                        indexed_at_ts = 0.0
                else:
                    indexed_at_ts = 0.0
                if file_mtime <= indexed_at_ts:
                    return False
            return True

        # --- Memories ---
        if mem_dir.is_dir():
            for md_file in sorted(mem_dir.glob("*.md")):
                try:
                    if not _needs_reindex(md_file):
                        continue
                    meta, _ = parse_frontmatter(md_file)
                    if "id" not in meta:
                        continue
                    self.index_memory(md_file)
                    count += 1
                except Exception:
                    continue

        # --- Facts ---
        if f_dir and f_dir.is_dir():
            for md_file in sorted(f_dir.glob("*.md")):
                try:
                    if not _needs_reindex(md_file):
                        continue
                    count += self.index_facts_file(md_file)
                except Exception:
                    continue

        # --- Projects ---
        if p_dir and p_dir.is_dir():
            for md_file in sorted(p_dir.glob("*.md")):
                try:
                    if not _needs_reindex(md_file):
                        continue
                    meta, _ = parse_frontmatter(md_file)
                    if "id" not in meta:
                        continue
                    self.index_project_file(md_file)
                    count += 1
                except Exception:
                    continue

        # Update the last rebuild time.
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self._db.execute(
            """
            INSERT INTO index_meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            ("last_rebuild_time", now_iso),
        )
        self._db.commit()

        return count + len(stale_ids)

    # ------------------------------------------------------------------
    # Search — semantic (ChromaDB)
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query: str,
        project: Optional[str] = None,
        topics: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        n: int = 5,
    ) -> List[SearchHit]:
        """
        Semantic search using ChromaDB.

        Filters:
        - project: exact match on project metadata
        - topics: if provided, post-filter results that have ANY of the specified topics
        - memory_type: exact match

        Returns: list of SearchHit, sorted by similarity DESC
        Similarity is computed as 1 - distance (cosine distance → similarity).
        """
        # Build ChromaDB where filter for exact-match fields.
        where_clauses: List[dict] = []
        if project is not None:
            where_clauses.append({"project": {"$eq": project}})
        if memory_type is not None:
            where_clauses.append({"memory_type": {"$eq": memory_type}})

        where: Optional[dict] = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        # If topics filter requested, fetch more results for post-filtering.
        fetch_n = (n * 2) if topics else n

        # Guard against querying an empty collection.
        collection_count = self._collection.count()
        if collection_count == 0:
            return []

        # Don't request more results than exist.
        fetch_n = min(fetch_n, collection_count)

        query_kwargs = {
            "query_texts": [query],
            "n_results": fetch_n,
        }
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        # Unpack ChromaDB results (they come as lists-of-lists).
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        hits: List[SearchHit] = []
        for i, mid in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            document = documents[i] if i < len(documents) else ""

            # Cosine distance → similarity: similarity = 1.0 - distance.
            similarity = max(0.0, min(1.0, 1.0 - distance))

            # Parse topics from JSON string.
            raw_topics = meta.get("topics", "[]")
            try:
                parsed_topics = json.loads(raw_topics) if isinstance(raw_topics, str) else raw_topics
                hit_topics: list = parsed_topics if isinstance(parsed_topics, list) else []
            except (json.JSONDecodeError, TypeError):
                hit_topics = []

            # Post-filter by topics: keep if ANY of the requested topics
            # appear in the memory's topics list.
            if topics:
                topics_lower = {t.lower() for t in topics}
                hit_topics_lower = {t.lower() for t in hit_topics}
                if not topics_lower & hit_topics_lower:
                    continue

            hits.append(
                SearchHit(
                    id=mid,
                    content=document,
                    similarity=similarity,
                    project=str(meta.get("project", "")),
                    topics=hit_topics,
                    memory_type=str(meta.get("memory_type", "")),
                    importance=float(meta.get("importance", 3.0)),
                    created=str(meta.get("created", "")),
                    file_path=str(meta.get("file_path", "")),
                )
            )

        # Sort by similarity descending, trim to requested n.
        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits[:n]

    def vector_search_reranked(
        self,
        query: str,
        config,
        project: Optional[str] = None,
        topics: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        n: int = 5,
        candidates: int = 0,
        think_fn=None,
    ) -> List[SearchHit]:
        """
        Two-stage search: vector retrieve top candidates, then LLM rerank to top n.

        Falls back to plain vector_search() if LLM is unavailable or reranking fails.

        Args:
            query: Search query text
            config: EngramConfig instance (for LLM settings)
            project: Filter by project
            topics: Filter by topics
            memory_type: Filter by type
            n: Final number of results after reranking
            candidates: Number of candidates to fetch (0 = use config default)
            think_fn: Optional agent thinking function (see engram.llm.ThinkFn)
        """
        from .rerank import rerank

        if not config.rerank_enabled:
            return self.vector_search(
                query=query, project=project, topics=topics,
                memory_type=memory_type, n=n,
            )

        n_candidates = candidates if candidates > 0 else config.rerank_candidates
        # Ensure we fetch at least n candidates
        n_candidates = max(n_candidates, n)

        # Stage 1: vector search for broad candidates
        raw_hits = self.vector_search(
            query=query, project=project, topics=topics,
            memory_type=memory_type, n=n_candidates,
        )

        if len(raw_hits) <= n:
            # Not enough candidates to rerank — return as-is
            return raw_hits

        # Stage 2: LLM rerank
        return rerank(
            query=query,
            candidates=raw_hits,
            config=config,
            top_k=n,
            think_fn=think_fn,
        )

    # ------------------------------------------------------------------
    # Search — structured (SQLite)
    # ------------------------------------------------------------------

    def metadata_query(
        self,
        project: Optional[str] = None,
        memory_type: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        order_by: str = "created",
        limit: int = 10,
    ) -> List[dict]:
        """
        Query SQLite for structured metadata queries (no semantic search).

        Returns: list of dicts from memory_index table
        """
        # Whitelist columns allowed for ORDER BY to prevent injection.
        allowed_order = {"created", "importance", "access_count", "indexed_at"}
        if order_by not in allowed_order:
            order_by = "created"

        clauses: List[str] = []
        params: List = []

        if project is not None:
            clauses.append("project = ?")
            params.append(project)
        if memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        if since is not None:
            clauses.append("created >= ?")
            params.append(since)
        if until is not None:
            clauses.append("created <= ?")
            params.append(until)

        sql = "SELECT * FROM memory_index"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += f" ORDER BY {order_by} DESC LIMIT ?"
        params.append(limit)

        rows = self._db.execute(sql, params).fetchall()

        results: List[dict] = []
        for row in rows:
            d = dict(row)
            # Deserialize topics from JSON string for convenience.
            try:
                d["topics"] = json.loads(d.get("topics", "[]") or "[]")
            except (json.JSONDecodeError, TypeError):
                d["topics"] = []
            results.append(d)

        return results

    # ------------------------------------------------------------------
    # Access tracking
    # ------------------------------------------------------------------

    def update_access_stats(self, memory_id: str) -> None:
        """Increment access_count and update last_accessed in SQLite."""
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self._db.execute(
            """
            UPDATE memory_index
            SET access_count = access_count + 1,
                last_accessed = ?
            WHERE id = ?
            """,
            (now_iso, memory_id),
        )
        self._db.commit()

    # ------------------------------------------------------------------
    # Content hash lookup (for dedup)
    # ------------------------------------------------------------------

    def get_content_hash(self, content_hash: str) -> Optional[str]:
        """Look up a memory id by content hash. Returns id or None."""
        row = self._db.execute(
            "SELECT id FROM memory_index WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        return row["id"] if row else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """
        Return index statistics.
        {
            "total_memories": int,
            "projects": list of distinct projects,
            "last_rebuild": str or None,
            "chroma_count": int,
            "sqlite_count": int,
        }
        """
        # SQLite count.
        sqlite_row = self._db.execute(
            "SELECT COUNT(*) AS cnt FROM memory_index"
        ).fetchone()
        sqlite_count = sqlite_row["cnt"] if sqlite_row else 0

        # Distinct projects.
        project_rows = self._db.execute(
            "SELECT DISTINCT project FROM memory_index WHERE project != '' AND project IS NOT NULL"
        ).fetchall()
        projects = [r["project"] for r in project_rows]

        # Last rebuild time.
        meta_row = self._db.execute(
            "SELECT value FROM index_meta WHERE key = ?",
            ("last_rebuild_time",),
        ).fetchone()
        last_rebuild = meta_row["value"] if meta_row else None

        # ChromaDB count.
        chroma_count = self._collection.count()

        return {
            "total_memories": sqlite_count,
            "projects": sorted(projects),
            "last_rebuild": last_rebuild,
            "chroma_count": chroma_count,
            "sqlite_count": sqlite_count,
        }
