# Engram Benchmark Results

> Run benchmarks: `cd packages/engram && python -m benchmarks.run_benchmark`

## Datasets

### Internal (Synthetic)

- **50 conversations, 100 questions** (80 test / 20 dev)
- Simulated multi-month SaaS development project
- 6 categories: single-session, multi-session, temporal, preference, adversarial, knowledge-update
- Committed to repo in `benchmarks/datasets/`

### External (Academic)

Downloaded at runtime, cached in `benchmarks/.cache/` (gitignored).

| Dataset | Source | Questions | Sessions | Categories |
|---|---|---|---|---|
| **LongMemEval** | [HuggingFace](https://huggingface.co/datasets/xiaowu0162/longmemeval) | 500 | 40–500/question | info-extraction, multi-session, temporal, knowledge-update, abstention |
| **LoCoMo** | [GitHub](https://github.com/snap-research/locomo) | ~200 (10 convos) | up to 35/convo | single-hop, multi-hop, temporal, commonsense, adversarial |

```bash
# Download only:
python -m benchmarks.download_datasets

# Run directly:
python -m benchmarks.bench_longmemeval            # LongMemEval_S (auto-downloads)
python -m benchmarks.bench_longmemeval --variant oracle --limit 50
python -m benchmarks.bench_locomo                  # LoCoMo (auto-downloads)
```

## Retrieval Recall (test split, n=80)

| Mode | R@3 | R@5 | R@10 | NDCG@5 | MRR |
|---|---|---|---|---|---|
| Raw vector search | —% | —% | —% | —.— | —.— |
| Project filtered | —% | —% | —% | —.— | —.— |

> Run `python -m benchmarks.run_benchmark --layer retrieval` to fill in these numbers.

## Per-Category Breakdown (raw mode, R@5)

| Category | Count | R@5 | Failed |
|---|---|---|---|
| single_session | — | —% | — |
| multi_session | — | —% | — |
| temporal | — | —% | — |
| preference | — | —% | — |
| adversarial | — | —% | — |
| knowledge_update | — | —% | — |

## Pipeline Quality

| Component | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Quality gate | —% | —% | —% | —% |
| Fact extraction (heuristic) | — | —% | — | — |

> Run `python -m benchmarks.run_benchmark --layer pipeline` to fill in these numbers.

## Conflict Detection

| Type | Accuracy | Count |
|---|---|---|
| temporal_succession | —% | 5 |
| implicit_supersede | —% | 7 |
| opinion_change | —% | 3 |
| hard_contradiction | —% | 5 |
| Overall | —% | 20 |

> Run `python -m benchmarks.run_benchmark --layer conflicts` to fill in these numbers.

## End-to-End

| Metric | Value |
|---|---|
| Session retrieval R@5 | —% |
| Session retrieval R@10 | —% |
| Answer in top-5 content | —% |

> Run `python -m benchmarks.run_benchmark --layer e2e` to fill in these numbers.

## External: LongMemEval

| Metric | LongMemEval_S | LongMemEval_Oracle |
|---|---|---|
| R@3 | —% | —% |
| R@5 | —% | —% |
| R@10 | —% | —% |
| NDCG@5 | —.— | —.— |
| MRR | —.— | —.— |

**Per-Category (LongMemEval_S, R@10):**

| Category | Count | R@10 | MRR |
|---|---|---|---|
| info-extraction | — | —% | —.— |
| multi-session | — | —% | —.— |
| temporal-reasoning | — | —% | —.— |
| knowledge-update | — | —% | —.— |

> Run `python -m benchmarks.bench_longmemeval` to fill in these numbers.
> Compare with MemPalace: 96.6% R@5 (LongMemEval_S, hybrid_v4)

## External: LoCoMo

| Metric | Value |
|---|---|
| R@3 | —% |
| R@5 | —% |
| R@10 | —% |
| MRR | —.— |

**Per-Category (R@10):**

| Category | Count | R@10 | MRR |
|---|---|---|---|
| single-hop | — | —% | —.— |
| multi-hop | — | —% | —.— |
| temporal | — | —% | —.— |
| commonsense | — | —% | —.— |

> Run `python -m benchmarks.bench_locomo` to fill in these numbers.

## Reproducibility

- **Embedding model:** all-MiniLM-L6-v2 (ChromaDB default)
- **LLM mode:** none (heuristic only for benchmarks)
- **Run command:** `python -m benchmarks.run_benchmark`
- **Raw results:** `benchmarks/results/`

## Methodology Notes

### Train/Test Split (lessons from MemPalace-CN)

We follow strict evaluation protocol:
1. **20 dev questions** — used for tuning thresholds during development
2. **80 test questions** — never examined during development, reported as headline numbers
3. Both scores are reported; only test scores matter for comparisons
4. Raw JSONL results saved for audit

### Why This Matters

MemPalace-CN's hybrid_v4 achieved "100% on LongMemEval" by examining and fixing the exact 3 failing questions — classic teaching-to-the-test. Their honest held-out score was 98.4%. Our framework avoids this pitfall by design.
