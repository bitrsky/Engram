"""
ingest.py — Multi-format ingestion for Engram.

Ingests content from various sources into the memory system:
- Markdown files
- Plain text files
- Code files (with intelligent chunking)
- Conversation logs
- Directories (recursive)

Each ingested chunk goes through the remember() pipeline.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import EngramConfig
from .remember import remember, remember_batch, RememberResult
from .index import IndexManager

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

MAX_TEXT_CHUNK = 800      # chars per chunk for prose
MAX_CODE_CHUNK = 2000     # chars per chunk for code
MIN_CHUNK_SIZE = 50       # skip tiny fragments
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@dataclass
class IngestResult:
    """Result of an ingestion operation."""
    total_files: int = 0
    total_chunks: int = 0
    memories_created: int = 0
    duplicates_skipped: int = 0
    quality_rejected: int = 0
    errors: int = 0
    error_files: list = field(default_factory=list)


def _merge_ingest_result(target: IngestResult, source: IngestResult) -> None:
    """Accumulate *source* counters into *target* in-place."""
    target.total_files += source.total_files
    target.total_chunks += source.total_chunks
    target.memories_created += source.memories_created
    target.duplicates_skipped += source.duplicates_skipped
    target.quality_rejected += source.quality_rejected
    target.errors += source.errors
    target.error_files.extend(source.error_files)


# ──────────────────────────────────────────────
# File type detection
# ──────────────────────────────────────────────

EXTENSION_MAP = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".tsx": "code",
    ".jsx": "code",
    ".go": "code",
    ".rs": "code",
    ".rb": "code",
    ".java": "code",
    ".c": "code",
    ".cpp": "code",
    ".h": "code",
    ".json": "data",
    ".yaml": "data",
    ".yml": "data",
    ".toml": "data",
    ".log": "text",
    ".csv": "data",
}

LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico",
    ".mp3", ".wav", ".mp4", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".pyo", ".class",
    ".woff", ".woff2", ".ttf", ".eot",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".env", ".mypy_cache", ".pytest_cache", ".tox",
    "dist", "build", ".next", ".nuxt",
}

# Words that add no topic value
_NOISE_PATH_PARTS = {
    "src", "lib", "app", "main", "index", "utils", "util", "helpers",
    "helper", "common", "shared", "internal", "pkg", "cmd", "test",
    "tests", "spec", "specs", "__init__", "mod", "core",
}


def _detect_file_type(filepath: Path) -> str:
    """Returns: 'markdown' | 'text' | 'code' | 'data' | 'conversation' | 'binary' | 'unknown'"""
    suffix = filepath.suffix.lower()

    if suffix in BINARY_EXTENSIONS:
        return "binary"

    # Check for conversation logs by name heuristic
    name_lower = filepath.stem.lower()
    if any(kw in name_lower for kw in ("conversation", "chat", "transcript", "dialog")):
        return "conversation"

    return EXTENSION_MAP.get(suffix, "unknown")


def _is_excluded(filepath: Path, exclude_patterns: Optional[List[str]] = None) -> bool:
    """Check if file should be excluded from ingestion."""
    # Binary check
    if filepath.suffix.lower() in BINARY_EXTENSIONS:
        return True

    # Check path parts against default excluded dirs
    parts = filepath.parts
    for part in parts:
        if part in DEFAULT_EXCLUDE_DIRS:
            return True

    # Size check
    try:
        if filepath.stat().st_size > MAX_FILE_SIZE:
            return True
    except OSError:
        return True

    # Symlinks
    if filepath.is_symlink():
        return True

    # Custom exclude patterns (glob-style matching against filename and relative path)
    if exclude_patterns:
        name = filepath.name
        path_str = str(filepath)
        for pattern in exclude_patterns:
            # fnmatch-style matching
            import fnmatch
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(path_str, pattern):
                return True

    return False


def _derive_topics(filepath: Path) -> List[str]:
    """
    Derive topic tags from file path and extension.

    Extracts meaningful words from path components, filtering out noise
    like 'src', 'lib', 'index', etc.

    Example: "src/auth/middleware.py" → ["auth", "middleware"]
    """
    topics = []

    # Get meaningful parts from path (directories + stem, skip root/drive)
    parts = list(filepath.parts)
    # Include the file stem (without extension)
    stem = filepath.stem.lower()

    # Collect directory names and stem
    candidates = []
    for part in parts[:-1]:  # directories only
        candidates.append(part.lower())
    candidates.append(stem)

    for candidate in candidates:
        # Skip noise words, hidden dirs, single chars, pure numbers
        if candidate in _NOISE_PATH_PARTS:
            continue
        if candidate.startswith("."):
            continue
        if len(candidate) <= 1:
            continue
        if candidate.isdigit():
            continue
        # Skip drive letters like "c:" or root-like entries
        if re.match(r"^[a-z]:?$", candidate):
            continue
        # Split camelCase and snake_case
        words = re.split(r"[_\-.]", candidate)
        # Further split camelCase
        expanded = []
        for word in words:
            expanded.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)", word))
        # If splitting produced nothing useful, keep original
        if expanded:
            for w in expanded:
                w_lower = w.lower()
                if w_lower not in _NOISE_PATH_PARTS and len(w_lower) > 1:
                    topics.append(w_lower)
        else:
            if candidate not in _NOISE_PATH_PARTS:
                topics.append(candidate)

    # Add language as topic for code files
    suffix = filepath.suffix.lower()
    lang = LANGUAGE_MAP.get(suffix)
    if lang:
        topics.append(lang)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique


# ──────────────────────────────────────────────
# Chunking strategies
# ──────────────────────────────────────────────

def chunk_markdown(content: str) -> List[dict]:
    """
    Split Markdown by ## headers.

    Each section becomes a chunk with the header prepended for context.
    If no headers found, treat as single chunk.

    Returns list of:
        {
            "content": "## Header\\n\\nsection text",
            "memory_type": "note" (or "decision" if contains decision markers),
            "topic": "section-title-slugified"
        }
    """
    content = content.strip()
    if not content:
        return []

    # Decision markers
    decision_markers = [
        "decision:", "decided to", "we decided", "we chose", "we picked",
        "going with", "settled on", "chose to", "selected",
    ]

    # Split by ## headers (keep the header line with each section)
    # Also handles # (h1) as top-level splits
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(content))

    if not matches:
        # No headers — treat entire content as one chunk
        memory_type = "note"
        content_lower = content.lower()
        if any(marker in content_lower for marker in decision_markers):
            memory_type = "decision"
        return [{"content": content, "memory_type": memory_type, "topic": ""}]

    chunks = []

    # Content before the first header (preamble)
    preamble = content[:matches[0].start()].strip()
    if preamble and len(preamble) >= MIN_CHUNK_SIZE:
        memory_type = "note"
        if any(m in preamble.lower() for m in decision_markers):
            memory_type = "decision"
        chunks.append({
            "content": preamble,
            "memory_type": memory_type,
            "topic": "",
        })

    # Each header section
    for i, match in enumerate(matches):
        header_line = match.group(0)
        header_text = match.group(2).strip()
        section_start = match.start()

        # Section ends at next header or end of content
        if i + 1 < len(matches):
            section_end = matches[i + 1].start()
        else:
            section_end = len(content)

        section_content = content[section_start:section_end].strip()

        if len(section_content) < MIN_CHUNK_SIZE:
            continue

        # Slugify the header for topic
        topic = re.sub(r"[^a-z0-9]+", "-", header_text.lower()).strip("-")

        # Detect memory type
        memory_type = "note"
        if any(m in section_content.lower() for m in decision_markers):
            memory_type = "decision"

        chunks.append({
            "content": section_content,
            "memory_type": memory_type,
            "topic": topic,
        })

    return chunks if chunks else [{"content": content, "memory_type": "note", "topic": ""}]


def chunk_text(content: str, max_size: int = MAX_TEXT_CHUNK) -> List[dict]:
    """
    Split plain text by paragraphs (double newlines).

    Merge small paragraphs to reach ~max_size chars per chunk.
    Don't split mid-paragraph.
    """
    content = content.strip()
    if not content:
        return []

    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r"\n\s*\n", content)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    chunks = []
    current_parts: List[str] = []
    current_size = 0

    for para in paragraphs:
        para_len = len(para)

        # If a single paragraph exceeds max_size, emit current buffer first,
        # then emit the large paragraph as its own chunk
        if para_len > max_size:
            if current_parts:
                merged = "\n\n".join(current_parts)
                if len(merged) >= MIN_CHUNK_SIZE:
                    chunks.append({"content": merged, "memory_type": "note", "topic": ""})
                current_parts = []
                current_size = 0
            # Emit the large paragraph directly
            chunks.append({"content": para, "memory_type": "note", "topic": ""})
            continue

        # Would adding this paragraph exceed max_size?
        new_size = current_size + para_len + (2 if current_parts else 0)  # +2 for \n\n
        if new_size > max_size and current_parts:
            # Flush current buffer
            merged = "\n\n".join(current_parts)
            if len(merged) >= MIN_CHUNK_SIZE:
                chunks.append({"content": merged, "memory_type": "note", "topic": ""})
            current_parts = [para]
            current_size = para_len
        else:
            current_parts.append(para)
            current_size = new_size

    # Flush remaining
    if current_parts:
        merged = "\n\n".join(current_parts)
        if len(merged) >= MIN_CHUNK_SIZE:
            chunks.append({"content": merged, "memory_type": "note", "topic": ""})

    return chunks


def chunk_code(content: str, language: str) -> List[dict]:
    """
    Split code by function/class definitions.

    Supported languages:
    - Python: split on `def ` and `class `
    - JavaScript/TypeScript: split on `function `, `class `, `const X = (`
    - Go: split on `func `
    - Rust: split on `fn `, `impl `

    Only keep chunks with substantial content (>5 lines).
    Each chunk includes the preceding comment/docstring if present.

    Falls back to chunk_text if regex-based splitting fails or produces
    no useful chunks.
    """
    content = content.strip()
    if not content:
        return []

    # Language-specific split patterns (match at the start of a line)
    patterns = {
        "python": re.compile(
            r"^(?=(?:def |class |async\s+def ))", re.MULTILINE
        ),
        "javascript": re.compile(
            r"^(?=(?:function\s|class\s|const\s+\w+\s*=\s*(?:\(|async)|export\s+(?:default\s+)?(?:function|class)))",
            re.MULTILINE,
        ),
        "typescript": re.compile(
            r"^(?=(?:function\s|class\s|const\s+\w+\s*=\s*(?:\(|async)|export\s+(?:default\s+)?(?:function|class)|interface\s|type\s+\w+\s*=))",
            re.MULTILINE,
        ),
        "go": re.compile(
            r"^(?=func\s)", re.MULTILINE
        ),
        "rust": re.compile(
            r"^(?=(?:(?:pub\s+)?fn\s|impl\s|(?:pub\s+)?struct\s|(?:pub\s+)?enum\s))",
            re.MULTILINE,
        ),
        "ruby": re.compile(
            r"^(?=(?:def\s|class\s|module\s))", re.MULTILINE
        ),
        "java": re.compile(
            r"^(?=(?:(?:public|private|protected|static|\s)*(?:class|interface|enum|void|int|String|boolean|long|double|float|char|byte|short)\s))",
            re.MULTILINE,
        ),
    }

    pattern = patterns.get(language)
    if pattern is None:
        # Unknown language — fall back to text chunking
        return chunk_text(content, max_size=MAX_CODE_CHUNK)

    # Find all split points
    split_points = [m.start() for m in pattern.finditer(content)]

    if len(split_points) < 2:
        # Not enough definitions to split meaningfully
        # If the whole file is small enough, keep it as one chunk
        if len(content) <= MAX_CODE_CHUNK:
            lines = content.split("\n")
            if len(lines) > 5:
                return [{"content": content, "memory_type": "note", "topic": ""}]
        # Otherwise fall back to text chunking
        return chunk_text(content, max_size=MAX_CODE_CHUNK)

    # Extend each chunk backwards to grab preceding comments/docstrings
    lines = content.split("\n")
    line_offsets = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line) + 1  # +1 for \n

    def _offset_to_line(char_offset: int) -> int:
        """Convert a character offset to a 0-based line number."""
        for i, lo in enumerate(line_offsets):
            if i + 1 < len(line_offsets) and line_offsets[i + 1] > char_offset:
                return i
            if i + 1 == len(line_offsets) and lo <= char_offset:
                return i
        return 0

    def _find_comment_start(def_line: int) -> int:
        """Walk backwards from def_line to include preceding comments/decorators."""
        start = def_line
        for ln in range(def_line - 1, -1, -1):
            stripped = lines[ln].strip()
            if not stripped:
                # Allow one blank line between comment block and definition
                if ln > 0 and lines[ln - 1].strip().startswith(("#", "//", "/*", "*", "@", "///")):
                    continue
                break
            if stripped.startswith(("#", "//", "/*", "*", "*/", "@", "///", "##")):
                start = ln
            elif stripped.startswith('"""') or stripped.startswith("'''"):
                start = ln
            else:
                break
        return start

    chunks = []
    adjusted_starts = []

    for sp in split_points:
        def_line = _offset_to_line(sp)
        comment_line = _find_comment_start(def_line)
        adjusted_start = line_offsets[comment_line] if comment_line < len(line_offsets) else sp
        adjusted_starts.append(adjusted_start)

    # Preamble: content before first definition (imports, module docstring, etc.)
    preamble = content[:adjusted_starts[0]].strip()
    if preamble and len(preamble) >= MIN_CHUNK_SIZE and preamble.count("\n") >= 5:
        chunks.append({"content": preamble, "memory_type": "note", "topic": "imports"})

    # Each definition block
    for i, start in enumerate(adjusted_starts):
        if i + 1 < len(adjusted_starts):
            end = adjusted_starts[i + 1]
        else:
            end = len(content)

        block = content[start:end].strip()

        # Skip tiny blocks (< 5 lines)
        if block.count("\n") < 5:
            continue

        # If block exceeds MAX_CODE_CHUNK, keep it anyway (functions can be long)
        # but cap at a reasonable size
        if len(block) > MAX_CODE_CHUNK * 2:
            block = block[:MAX_CODE_CHUNK * 2]

        # Try to extract a topic from the definition line
        topic = ""
        first_line = block.split("\n")[0].strip() if block else ""
        # Extract function/class name
        name_match = re.search(
            r"(?:def|class|func|fn|function|impl|const|interface|type|struct|enum|module)\s+(\w+)",
            first_line,
        )
        if not name_match:
            # Try from the actual split point line
            def_line_idx = _offset_to_line(split_points[i])
            if def_line_idx < len(lines):
                name_match = re.search(
                    r"(?:def|class|func|fn|function|impl|const|interface|type|struct|enum|module)\s+(\w+)",
                    lines[def_line_idx],
                )
        if name_match:
            topic = name_match.group(1).lower()

        chunks.append({"content": block, "memory_type": "note", "topic": topic})

    # If we got no meaningful chunks, fall back to text chunking
    if not chunks:
        return chunk_text(content, max_size=MAX_CODE_CHUNK)

    return chunks


def chunk_conversation(content: str) -> List[dict]:
    """
    Split conversation logs by speaker turns.

    Detects patterns like:
    - "User: ..." / "Assistant: ..."
    - "**User**: ..."
    - "> ..."

    Groups consecutive turns into exchange pairs for context.
    """
    content = content.strip()
    if not content:
        return []

    lines = content.split("\n")

    # Detect conversation format
    # Pattern 1: "> user message" followed by assistant response
    has_quote_markers = sum(1 for l in lines if l.strip().startswith(">")) >= 2
    # Pattern 2: "User: ..." / "Assistant: ..."
    role_pattern = re.compile(
        r"^(?:\*\*)?(?:User|Human|Assistant|AI|Bot|System|Claude|GPT|ChatGPT)(?:\*\*)?:\s*",
        re.IGNORECASE,
    )
    has_role_labels = sum(1 for l in lines if role_pattern.match(l.strip())) >= 2

    if not has_quote_markers and not has_role_labels:
        # Not a recognisable conversation format — fall back to text chunking
        return chunk_text(content, max_size=MAX_TEXT_CHUNK)

    # Parse turns
    turns: List[dict] = []  # {"role": "user"|"assistant", "text": str}

    if has_quote_markers:
        # "> text" = user, non-">" blocks = assistant
        current_role = None
        current_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(">"):
                # User turn
                user_text = stripped.lstrip(">").strip()
                if current_role == "assistant" and current_lines:
                    turns.append({"role": "assistant", "text": "\n".join(current_lines).strip()})
                    current_lines = []
                current_role = "user"
                current_lines.append(user_text)
            elif stripped == "" and current_role == "user" and current_lines:
                # End of user turn
                turns.append({"role": "user", "text": "\n".join(current_lines).strip()})
                current_lines = []
                current_role = "assistant"
            elif current_role == "assistant" or (current_role is None and stripped):
                current_role = "assistant"
                current_lines.append(line)
            elif current_role == "user":
                # Continuation of user turn? Only if starts with >
                current_lines.append(stripped)

        if current_lines and current_role:
            turns.append({"role": current_role, "text": "\n".join(current_lines).strip()})

    elif has_role_labels:
        current_role = None
        current_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            match = role_pattern.match(stripped)
            if match:
                # Flush previous turn
                if current_role and current_lines:
                    turns.append({"role": current_role, "text": "\n".join(current_lines).strip()})
                    current_lines = []

                label = stripped[:match.end()].strip().rstrip(":").strip("*").lower()
                if label in ("user", "human"):
                    current_role = "user"
                else:
                    current_role = "assistant"
                rest = stripped[match.end():].strip()
                if rest:
                    current_lines.append(rest)
            else:
                current_lines.append(line)

        if current_role and current_lines:
            turns.append({"role": current_role, "text": "\n".join(current_lines).strip()})

    if not turns:
        return chunk_text(content, max_size=MAX_TEXT_CHUNK)

    # Group turns into exchange pairs (user + assistant = 1 chunk)
    chunks = []
    i = 0
    while i < len(turns):
        exchange_parts = []
        exchange_parts.append(f"[{turns[i]['role'].upper()}]\n{turns[i]['text']}")
        # If next turn is a different role, pair them
        if (
            i + 1 < len(turns)
            and turns[i + 1]["role"] != turns[i]["role"]
        ):
            exchange_parts.append(f"[{turns[i + 1]['role'].upper()}]\n{turns[i + 1]['text']}")
            i += 2
        else:
            i += 1

        exchange_text = "\n\n".join(exchange_parts)
        if len(exchange_text) >= MIN_CHUNK_SIZE:
            chunks.append({
                "content": exchange_text,
                "memory_type": "note",
                "topic": "conversation",
            })

    return chunks if chunks else [{"content": content, "memory_type": "note", "topic": "conversation"}]


# ──────────────────────────────────────────────
# Ingestion functions
# ──────────────────────────────────────────────

def ingest_file(
    filepath: str | Path,
    project: Optional[str] = None,
    topics: Optional[List[str]] = None,
    config: Optional[EngramConfig] = None,
    index_manager: Optional[IndexManager] = None,
) -> IngestResult:
    """
    Ingest a single file into memory.

    Detects file type and applies appropriate chunking:
    - .md: Split by ## headers (each section = one memory)
    - .txt: Split by double newlines (paragraphs)
    - .py/.js/.ts/.go/.rs: Split by function/class definitions
    - .json/.yaml/.toml: Treat as single chunk
    - Other: Treat as plain text

    Args:
        filepath: Path to file
        project: Project tag
        topics: Topic tags (if None, derived from file extension)
        config: Configuration
        index_manager: Shared index manager

    Returns:
        IngestResult
    """
    filepath = Path(filepath)
    config = config or EngramConfig()
    result = IngestResult(total_files=1)

    # Validate file
    if not filepath.is_file():
        result.errors = 1
        result.error_files.append(str(filepath))
        logger.warning("File not found: %s", filepath)
        return result

    if _is_excluded(filepath):
        result.errors = 1
        result.error_files.append(str(filepath))
        logger.debug("File excluded: %s", filepath)
        return result

    # Read file content
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        result.errors = 1
        result.error_files.append(str(filepath))
        logger.warning("Cannot read %s: %s", filepath, e)
        return result

    content = content.strip()
    if not content or len(content) < MIN_CHUNK_SIZE:
        return result

    # Detect file type and chunk accordingly
    file_type = _detect_file_type(filepath)

    if file_type == "binary":
        return result

    if file_type == "markdown":
        chunks = chunk_markdown(content)
    elif file_type == "code":
        language = LANGUAGE_MAP.get(filepath.suffix.lower(), "")
        chunks = chunk_code(content, language)
    elif file_type == "conversation":
        chunks = chunk_conversation(content)
    elif file_type == "data":
        # Data files: treat as a single chunk
        chunks = [{"content": content, "memory_type": "note", "topic": ""}]
    else:
        # text, unknown — paragraph-based chunking
        chunks = chunk_text(content)

    if not chunks:
        return result

    result.total_chunks = len(chunks)

    # Derive topics from filepath if not provided
    file_topics = topics if topics else _derive_topics(filepath)

    # Build batch items for remember_batch
    items = []
    for chunk_info in chunks:
        chunk_content = chunk_info["content"]
        chunk_type = chunk_info.get("memory_type", "note")
        chunk_topic = chunk_info.get("topic", "")

        # Merge chunk-specific topic with file-level topics
        combined_topics = list(file_topics)
        if chunk_topic and chunk_topic not in combined_topics:
            combined_topics.append(chunk_topic)

        items.append({
            "content": chunk_content,
            "project": project,
            "topics": combined_topics,
            "memory_type": chunk_type,
            "source": "ingest",
            "skip_quality_check": False,
            "skip_dedup": False,
            "skip_facts": True,  # bulk ingestion skips fact extraction for speed
        })

    # Run through remember pipeline
    results = remember_batch(items, config=config, index_manager=index_manager)

    for r in results:
        if r.success:
            result.memories_created += 1
        elif "duplicate" in r.rejected_reason:
            result.duplicates_skipped += 1
        elif "low_quality" in r.rejected_reason:
            result.quality_rejected += 1
        else:
            result.errors += 1

    return result


def ingest_directory(
    dirpath: str | Path,
    project: Optional[str] = None,
    topics: Optional[List[str]] = None,
    recursive: bool = True,
    exclude_patterns: Optional[List[str]] = None,
    config: Optional[EngramConfig] = None,
) -> IngestResult:
    """
    Ingest all files in a directory.

    Default exclude patterns:
    - .git/, node_modules/, __pycache__/, .venv/, .env
    - *.pyc, *.pyo, *.so, *.dll, *.exe
    - Binary files (images, videos, archives)

    Args:
        dirpath: Directory to ingest
        project: Project tag
        topics: Topic tags
        recursive: Whether to recurse into subdirectories
        exclude_patterns: Additional glob patterns to exclude
        config: Configuration

    Returns:
        IngestResult (accumulated from all files)
    """
    dirpath = Path(dirpath)
    config = config or EngramConfig()
    overall = IngestResult()

    if not dirpath.is_dir():
        logger.warning("Directory not found: %s", dirpath)
        overall.errors = 1
        overall.error_files.append(str(dirpath))
        return overall

    # Create one IndexManager shared across all files for efficiency
    index_manager = IndexManager(
        index_dir=config.index_dir,
        memories_dir=config.memories_dir,
    )

    try:
        if recursive:
            walker = os.walk(dirpath)
        else:
            # Only the top-level directory
            try:
                entries = list(dirpath.iterdir())
                dirs = [e.name for e in entries if e.is_dir()]
                files = [e.name for e in entries if e.is_file()]
                walker = [(str(dirpath), dirs, files)]
            except OSError as e:
                logger.warning("Cannot list directory %s: %s", dirpath, e)
                overall.errors = 1
                overall.error_files.append(str(dirpath))
                return overall

        for root, dirs, filenames in walker:
            root_path = Path(root)

            # Prune excluded directories in-place (affects os.walk recursion)
            dirs[:] = [
                d for d in dirs
                if d not in DEFAULT_EXCLUDE_DIRS and not d.startswith(".")
            ]
            # Also prune based on custom exclude patterns
            if exclude_patterns:
                import fnmatch
                dirs[:] = [
                    d for d in dirs
                    if not any(fnmatch.fnmatch(d, p) for p in exclude_patterns)
                ]

            for filename in sorted(filenames):
                filepath = root_path / filename

                if _is_excluded(filepath, exclude_patterns):
                    continue

                # Check extension is in our known readable set
                suffix = filepath.suffix.lower()
                if suffix not in EXTENSION_MAP and suffix not in BINARY_EXTENSIONS:
                    # Unknown extension — skip unless it looks like text
                    # (we're conservative: skip by default)
                    continue

                if suffix in BINARY_EXTENSIONS:
                    continue

                file_result = ingest_file(
                    filepath=filepath,
                    project=project,
                    topics=topics,
                    config=config,
                    index_manager=index_manager,
                )
                _merge_ingest_result(overall, file_result)

    finally:
        index_manager.close()

    return overall


def ingest_text(
    text: str,
    project: Optional[str] = None,
    topics: Optional[List[str]] = None,
    source: str = "ingest",
    config: Optional[EngramConfig] = None,
    index_manager: Optional[IndexManager] = None,
) -> IngestResult:
    """
    Ingest raw text content.

    Splits into chunks and runs through remember pipeline.
    """
    config = config or EngramConfig()
    result = IngestResult(total_files=0)

    text = text.strip()
    if not text or len(text) < MIN_CHUNK_SIZE:
        return result

    # Decide chunking strategy based on content heuristics
    lines = text.split("\n")
    has_headers = any(l.strip().startswith("#") for l in lines)
    has_quotes = sum(1 for l in lines if l.strip().startswith(">")) >= 2
    has_role_labels = bool(
        re.search(
            r"^(?:\*\*)?(?:User|Human|Assistant|AI|Bot)(?:\*\*)?:\s*",
            text,
            re.MULTILINE | re.IGNORECASE,
        )
    )

    if has_quotes or has_role_labels:
        chunks = chunk_conversation(text)
    elif has_headers:
        chunks = chunk_markdown(text)
    else:
        chunks = chunk_text(text)

    if not chunks:
        return result

    result.total_chunks = len(chunks)

    items = []
    for chunk_info in chunks:
        chunk_content = chunk_info["content"]
        chunk_type = chunk_info.get("memory_type", "note")
        chunk_topic = chunk_info.get("topic", "")

        combined_topics = list(topics) if topics else []
        if chunk_topic and chunk_topic not in combined_topics:
            combined_topics.append(chunk_topic)

        items.append({
            "content": chunk_content,
            "project": project,
            "topics": combined_topics,
            "memory_type": chunk_type,
            "source": source,
            "skip_quality_check": False,
            "skip_dedup": False,
            "skip_facts": True,
        })

    results = remember_batch(items, config=config, index_manager=index_manager)

    for r in results:
        if r.success:
            result.memories_created += 1
        elif "duplicate" in r.rejected_reason:
            result.duplicates_skipped += 1
        elif "low_quality" in r.rejected_reason:
            result.quality_rejected += 1
        else:
            result.errors += 1

    return result
