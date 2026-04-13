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

    def __init__(self, index_dir: str | Path, memories_dir: str | Path = None):
        """
        Args:
            index_dir: Path to .index/ directory (contains vectors.chroma/ and meta.sqlite3)
            memories_dir: Path to memories/ directory (for rebuild)
        """
        self._index_dir = Path(index_dir)
        self._memories_dir = Path(memories_dir) if memories_dir else None
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
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SQLITE_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close database connections."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

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
        self._conn.execute(
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
        self._conn.commit()

        return memory_id

    def remove_from_index(self, memory_id: str) -> bool:
        """Remove a memory from both indexes. Returns True if found."""
        # Check existence in SQLite first.
        row = self._conn.execute(
            "SELECT id FROM memory_index WHERE id = ?", (memory_id,)
        ).fetchone()

        found = row is not None

        # Remove from ChromaDB (silently ignores missing ids).
        try:
            self._collection.delete(ids=[memory_id])
        except Exception:
            pass

        # Remove from SQLite.
        self._conn.execute(
            "DELETE FROM memory_index WHERE id = ?", (memory_id,)
        )
        self._conn.commit()

        return found

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def rebuild(self, memories_dir: str | Path = None) -> int:
        """
        Full rebuild: clear both indexes, scan all memories/*.md, re-index everything.

        Returns: number of memories indexed
        """
        mem_dir = Path(memories_dir) if memories_dir else self._memories_dir
        if mem_dir is None:
            raise ValueError(
                "memories_dir must be provided either in constructor or rebuild()"
            )

        # Clear ChromaDB collection.
        self._chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
        self._collection = self._chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Clear SQLite tables.
        self._conn.execute("DELETE FROM memory_index")
        self._conn.execute("DELETE FROM index_meta")
        self._conn.commit()

        # Scan and index every .md file.
        count = 0
        if mem_dir.is_dir():
            for md_file in sorted(mem_dir.glob("*.md")):
                try:
                    meta, _ = parse_frontmatter(md_file)
                    if "id" not in meta:
                        # Not a memory file — skip.
                        continue
                    self.index_memory(md_file)
                    count += 1
                except Exception:
                    # Skip files that can't be parsed/indexed.
                    continue

        # Record rebuild timestamp.
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self._conn.execute(
            """
            INSERT INTO index_meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            ("last_rebuild_time", now_iso),
        )
        self._conn.commit()

        return count

    def incremental_update(self, memories_dir: str | Path = None) -> int:
        """
        Incremental update: only process files modified since last index time.
        Also removes index entries for files that no longer exist.

        Returns: number of memories updated
        """
        mem_dir = Path(memories_dir) if memories_dir else self._memories_dir
        if mem_dir is None:
            raise ValueError(
                "memories_dir must be provided either in constructor or incremental_update()"
            )

        # Get the last rebuild/update timestamp.
        row = self._conn.execute(
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
        existing_rows = self._conn.execute(
            "SELECT id, file_path FROM memory_index"
        ).fetchall()

        stale_ids = []
        for r in existing_rows:
            if not Path(r["file_path"]).exists():
                stale_ids.append(r["id"])

        for sid in stale_ids:
            self.remove_from_index(sid)

        # --- Index new/modified files ---
        count = 0
        if mem_dir.is_dir():
            # Build a set of already-indexed file paths for quick lookup.
            indexed_rows = self._conn.execute(
                "SELECT file_path, indexed_at FROM memory_index"
            ).fetchall()
            indexed_map: Dict[str, str] = {
                r["file_path"]: (r["indexed_at"] or "") for r in indexed_rows
            }

            for md_file in sorted(mem_dir.glob("*.md")):
                try:
                    resolved = str(md_file.resolve())
                    file_mtime = md_file.stat().st_mtime

                    # Decide whether to (re-)index this file.
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
                            # File hasn't been modified since last indexing.
                            continue

                    # Parse to verify it's a memory file.
                    meta, _ = parse_frontmatter(md_file)
                    if "id" not in meta:
                        continue

                    self.index_memory(md_file)
                    count += 1
                except Exception:
                    continue

        # Update the last rebuild time.
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self._conn.execute(
            """
            INSERT INTO index_meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            ("last_rebuild_time", now_iso),
        )
        self._conn.commit()

        return count + len(stale_ids)

    # ------------------------------------------------------------------
    # Search — semantic (ChromaDB)
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query: str,
        project: str = None,
        topics: List[str] = None,
        memory_type: str = None,
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
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

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
                hit_topics = json.loads(raw_topics) if isinstance(raw_topics, str) else raw_topics
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
                    project=meta.get("project", ""),
                    topics=hit_topics,
                    memory_type=meta.get("memory_type", ""),
                    importance=float(meta.get("importance", 3.0)),
                    created=meta.get("created", ""),
                    file_path=meta.get("file_path", ""),
                )
            )

        # Sort by similarity descending, trim to requested n.
        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits[:n]

    def vector_search_reranked(
        self,
        query: str,
        config,
        project: str = None,
        topics: List[str] = None,
        memory_type: str = None,
        n: int = 5,
        candidates: int = 0,
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
        )

    # ------------------------------------------------------------------
    # Search — structured (SQLite)
    # ------------------------------------------------------------------

    def metadata_query(
        self,
        project: str = None,
        memory_type: str = None,
        since: str = None,
        until: str = None,
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

        rows = self._conn.execute(sql, params).fetchall()

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
        self._conn.execute(
            """
            UPDATE memory_index
            SET access_count = access_count + 1,
                last_accessed = ?
            WHERE id = ?
            """,
            (now_iso, memory_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Content hash lookup (for dedup)
    # ------------------------------------------------------------------

    def get_content_hash(self, content_hash: str) -> Optional[str]:
        """Look up a memory id by content hash. Returns id or None."""
        row = self._conn.execute(
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
        sqlite_row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM memory_index"
        ).fetchone()
        sqlite_count = sqlite_row["cnt"] if sqlite_row else 0

        # Distinct projects.
        project_rows = self._conn.execute(
            "SELECT DISTINCT project FROM memory_index WHERE project != '' AND project IS NOT NULL"
        ).fetchall()
        projects = [r["project"] for r in project_rows]

        # Last rebuild time.
        meta_row = self._conn.execute(
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
