# Benchmark Datasets

## Built-in Synthetic Dataset

The benchmark framework ships with a synthetic dataset that requires no external downloads:

- **`synthetic_conversations.json`** — 50 conversation sessions simulating a multi-month SaaS project
- **`synthetic_questions.json`** — 100 questions with ground-truth answers (80 test / 20 dev)

### Categories

| Category | Sessions | Questions (test/dev) | Description |
|---|---|---|---|
| single_session | s01–s10 | 33 (27/6) | Single fact per session |
| multi_session | s11–s20 | 14 (12/2) | Answer spans 2+ sessions |
| temporal | s21–s30 | 18 (15/3) | Chronological progression |
| preference | s31–s40 | 21 (18/3) | Personal preferences/opinions |
| adversarial | s41–s45 | 8 (7/1) | Red herrings, similar names |
| knowledge_update | s46–s50 | 6 (5/1) | Facts that changed over time |

### Train/Test Split Protocol

- **Dev split (20 questions):** Use for tuning thresholds and debugging
- **Test split (80 questions):** Never look at during development. Report as headline numbers.
- Both splits are marked in the JSON with `"split": "dev"` or `"split": "test"`

## External Datasets (Auto-Downloaded)

External academic datasets are **downloaded at runtime** and cached in `benchmarks/.cache/` (gitignored).
No manual setup needed — run the benchmark and it fetches what it needs.

### LongMemEval

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (Wu et al., 2024) — 500 questions testing 5 long-term memory abilities.

```bash
# Auto-downloads from HuggingFace on first run:
cd packages/engram
python -m benchmarks.bench_longmemeval                    # LongMemEval_S (~40 sessions)
python -m benchmarks.bench_longmemeval --variant oracle   # Oracle (evidence only)
python -m benchmarks.bench_longmemeval --limit 50         # First 50 questions
```

### LoCoMo

[LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., 2024) — 10 conversations, ~300 turns each, QA pairs in 5 categories.

```bash
# Auto-clones from GitHub on first run:
cd packages/engram
python -m benchmarks.bench_locomo                         # All 10 conversations
python -m benchmarks.bench_locomo --limit 3               # First 3 conversations
```

### Download Only (no evaluation)

```bash
python -m benchmarks.download_datasets              # Download all
python -m benchmarks.download_datasets longmemeval   # Just LongMemEval
python -m benchmarks.download_datasets locomo         # Just LoCoMo
```

## Creating Custom Datasets

Follow the JSON schema used by the synthetic dataset:

### Conversations Schema
```json
{
  "session_id": "s01",
  "timestamp": "2025-01-15T10:00:00",
  "project": "my-project",
  "topics": ["topic1"],
  "turns": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### Questions Schema
```json
{
  "id": "q01",
  "question": "What is X?",
  "category": "single_session",
  "evidence_session_ids": ["s01"],
  "answer": "Y",
  "answer_aliases": ["y", "Y thing"],
  "difficulty": "easy",
  "split": "test"
}
```
