# Engram — Markdown-First AI Memory System

> Structured recall with semantic search, fact extraction, and conflict resolution.

## Core Principles

1. **Markdown is Source of Truth** — all data lives in `.md` files you can read, edit, and version-control
2. **Project ≠ Directory** — projects are logical tags in frontmatter, not filesystem paths
3. **Index is Disposable** — ChromaDB + SQLite indexes are derived; delete `.index/` and rebuild anytime
4. **One AI Touch Point** — only fact extraction uses LLM (optional); everything else is deterministic

## Install

```bash
# with uv (recommended)
uv pip install -e .

# or from PyPI (when published)
pip install engram
```

## Quick Start

```bash
# Initialize memory storage
engram init

# Create a project
engram project create saas-app "My SaaS App"

# Remember something
engram remember "We chose Clerk over Auth0 for auth. Clerk pricing is better." --project saas-app

# Search memories
engram search "auth provider" --project saas-app

# Session startup (L0 identity + L1 working set)
engram wake-up --project saas-app

# View extracted facts
engram facts saas-app

# Check conflicts
engram conflicts
```

## Architecture

```
~/.engram/
├── memories/          # One .md file per memory (with YAML frontmatter)
├── projects/          # One .md file per project (registry)
├── facts/             # One .md file per project (structured knowledge graph)
├── identity.md        # L0 — who am I?
├── config.yaml        # Configuration
└── .index/            # Derived indexes (deletable, rebuildable)
    ├── vectors.chroma/    # ChromaDB semantic search
    └── meta.sqlite3       # SQLite structured queries
```

## 4-Layer Retrieval Stack

| Layer | Name | Trigger | Content |
|-------|------|---------|---------|
| L0 | Identity | Always | `identity.md` — who you are |
| L1 | Working Set | `wake-up` | Project overview + active facts + recent memories (≤500 tokens) |
| L2 | Contextual | Auto | Semantic search triggered by message content |
| L3 | Deep Search | Explicit | Full cross-project semantic search |

## Write Pipeline

```
Content → Quality Gate → Dedup → Write .md → Update Index → Extract Facts → Conflict Detection
                                    ↑                              ↑
                              Source of Truth               LLM (optional, $3/year)
```

## MCP Integration

For AI assistant integration via [Model Context Protocol](https://modelcontextprotocol.io/):

```bash
pip install engram[mcp]
```

12 tools available: `engram_status`, `engram_search`, `engram_recall`, `engram_facts`, `engram_timeline`, `engram_conflicts`, `engram_list_projects`, `engram_wake_up`, `engram_remember`, `engram_learn_fact`, `engram_forget_fact`, `engram_resolve_conflict`.

## Benchmarks

Engram ships with a 4-layer internal benchmark suite **plus** adapters for two academic memory benchmarks:

| Benchmark | Questions | What it tests |
|---|---|---|
| **Internal** (synthetic) | 100 | Retrieval recall, pipeline quality, conflict detection, end-to-end |
| **[LongMemEval](https://github.com/xiaowu0162/LongMemEval)** | 500 | 5 memory abilities across 40–500 session haystacks |
| **[LoCoMo](https://github.com/snap-research/locomo)** | ~200 | Long-term conversational memory (10 convos, ~300 turns each) |

External datasets are **downloaded at runtime** (cached in `.cache/`, never committed to repo).

```bash
cd packages/engram

# Internal benchmarks (pytest-based)
python -m benchmarks.run_benchmark                     # all layers
python -m benchmarks.run_benchmark --layer retrieval   # retrieval recall only

# Academic benchmarks (standalone scripts)
python -m benchmarks.bench_longmemeval                 # LongMemEval_S (auto-downloads)
python -m benchmarks.bench_locomo                      # LoCoMo (auto-downloads)
```

See [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md) for detailed results and methodology.

## LLM Configuration (Optional)

Engram works without any LLM. To enable AI-powered fact extraction:

```yaml
# ~/.engram/config.yaml
llm:
  provider: ollama          # ollama | openai | anthropic | none
  model: llama3.2           # model name
  # api_key: sk-...         # for openai/anthropic
```

## License

MIT
