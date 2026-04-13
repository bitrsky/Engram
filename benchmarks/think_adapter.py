"""
think_adapter.py — Create a ThinkFn backed by EchoCodeClient for benchmarks.

Provides:
- create_think_fn()       — sync ThinkFn for simple LLM calls
- async_think()           — async single LLM call
- deep_search_agent()     — async agent that searches memory files autonomously
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def create_think_fn(model: str = "anthropic/claude-sonnet-4-20250514"):
    """Create a sync ThinkFn backed by EchoCodeClient.

    Each call creates a fresh, stateless EchoCodeClient.
    """
    from echo_code import EchoCodeClient

    def think_fn(prompt: str, system: str = "", **kw) -> Optional[str]:
        client = EchoCodeClient(model=model, system_prompt=system, headless=True)
        coro = client.prompt_collect(prompt)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=kw.get("timeout", 120))
        else:
            return asyncio.run(coro)

    return think_fn


async def async_think(prompt: str, model: str, system: str = "", **kw) -> Optional[str]:
    """Single async LLM call — create a fresh EchoCodeClient, call, return."""
    from echo_code import EchoCodeClient
    client = EchoCodeClient(model=model, system_prompt=system, headless=True)
    try:
        return await client.prompt_collect(prompt)
    finally:
        await client.aclose()


# ═══════════════════════════════════════════════════════════════════════════
# Deep Search Agent — headless EchoCodeClient with built-in tools (ls, read)
# ═══════════════════════════════════════════════════════════════════════════

_DEEP_SEARCH_SYSTEM = """\
You are a memory search assistant. You have access to a personal knowledge base \
stored on disk. The knowledge base has three layers:

## Directory Structure

  {base_dir}/
  ├── identity.md              # User identity and preferences
  ├── memories/                # Conversation memories (markdown)
  │   ├── 2024-01-15_auth-decision.md
  │   └── ...
  ├── facts/                   # Knowledge graph (per-project)
  │   ├── project-name.md      # Facts as subject|predicate|object triples
  │   └── ...
  └── projects/                # Project registry
      ├── project-name.md      # Project metadata, status, aliases
      └── ...

## File Formats

### memories/*.md
YAML frontmatter with id, project, topics, memory_type, importance, created.
Body contains the raw conversation or note text.
Example frontmatter:
  ---
  id: session_5
  project: my-project
  topics: [auth, infrastructure]
  memory_type: decision
  importance: 4.0
  created: 2024-01-15T14:23:00
  ---
  We decided to use Clerk for authentication because...

### facts/*.md
Knowledge graph stored as structured lines:
  - subject | predicate | object | since: YYYY-MM | confidence: 0.95
    - source: mem_xxx
Sections: ## Current, ## Expired, ## Conflicts
Use this to find factual relationships (who uses what, preferences, tech stack).

### projects/*.md
Project metadata: display_name, status (active/paused/archived), aliases.
Use this to understand what projects exist and their current state.

## Search Strategy

1. Check the vector search hints first — they may already contain the answer.
2. For factual questions (what does X use, who prefers Y), check facts/ first.
3. For event/conversation questions, grep or read memories/.
4. Use grep to search across files efficiently, then read specific files.

## Response Format (MUST follow exactly)

ANSWER: {{your answer}}
EVIDENCE: {{comma-separated memory IDs (e.g. session_5, session_12) or fact file names}}

If not found:
ANSWER: NOT_FOUND
EVIDENCE: none
"""


async def deep_search_agent(
    query: str,
    model: str,
    base_dir: Path,
    vector_hits: list,
    project: Optional[str] = None,
) -> Optional[str]:
    """Run an agentic deep search — LLM uses built-in ls/read/grep to find answers.

    Creates a headless EchoCodeClient. The system prompt describes the full
    engram directory structure (memories, facts, projects). The LLM uses
    its built-in tools to autonomously search.

    Args:
        query: The user's question.
        model: LLM model name.
        base_dir: Path to the engram base directory (e.g. ~/.engram).
        vector_hits: Vector search top-K results (SearchHit objects).
        project: Optional project filter.

    Returns:
        The raw LLM response (containing ANSWER: and EVIDENCE:), or None.
    """
    from echo_code import EchoCodeClient

    base_dir = Path(base_dir).resolve()

    system = _DEEP_SEARCH_SYSTEM.format(base_dir=base_dir)

    # Build the user prompt with vector search hints
    hint_lines = []
    for i, hit in enumerate(vector_hits[:10]):
        content = hit.content.strip()
        if len(content) > 500:
            content = content[:500] + "..."
        created = getattr(hit, "created", "") or ""
        hit_id = getattr(hit, "id", f"hit_{i}")
        hint_lines.append(f"[{i+1}] ({hit_id}, {created})")
        hint_lines.append(content)
        hint_lines.append("")

    hints_block = "\n".join(hint_lines) if hint_lines else "(no vector results)"

    user_prompt = (
        f"Question: {query}\n\n"
        f"=== Vector search hints (may or may not contain the answer) ===\n"
        f"{hints_block}\n\n"
        f"If the hints above answer the question, respond immediately with "
        f"the memory IDs from the hints.\n"
        f"Otherwise, search the knowledge base at {base_dir}. "
        f"Use grep to find keywords in memories/, check facts/ for factual "
        f"questions, then read matching files.\n"
        f"EVIDENCE must list memory IDs (like session_5), not filenames."
    )

    client = EchoCodeClient(
        model=model,
        system_prompt=system,
        headless=True,
    )
    try:
        result = await client.prompt_collect(user_prompt)
        return result
    except Exception as exc:
        logger.debug("Deep search agent failed: %s", exc)
        return None
    finally:
        await client.aclose()
