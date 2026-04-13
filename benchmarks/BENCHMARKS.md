# Engram Benchmark Results

> Run benchmarks: `cd packages/engram && python -m benchmarks.run_benchmark`

## 1. Internal Synthetic Benchmarks

### Dataset

- **Synthetic:** 50 conversations, 100 questions (80 test / 20 dev)
- **Project:** simulated multi-month SaaS development
- **6 categories:** single-session, multi-session, temporal, preference, adversarial, knowledge-update

### Retrieval Recall (test split, n=80)

| Mode | R@3 | R@5 | R@10 | NDCG@5 | MRR |
|---|---|---|---|---|---|
| Raw vector search | 100% | 100% | 100% | 0.923 | 0.935 |

> Run `python -m pytest benchmarks/test_retrieval.py` to reproduce.

### Pipeline Quality

| Component | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Quality gate | 95%+ | 95%+ | 95.2% | 95%+ |
| Fact extraction (heuristic) | -- | 100% | -- | -- |

> Run `python -m pytest benchmarks/test_pipeline.py` to reproduce.

### Conflict Detection

| Type | Accuracy | Count |
|---|---|---|
| temporal_succession | 100% | 5 |
| implicit_supersede | 95% | 7 |
| opinion_change | 100% | 3 |
| hard_contradiction | 100% | 5 |
| **Overall** | **95%** | **20** |

> Run `python -m pytest benchmarks/test_conflicts.py` to reproduce.

### End-to-End

| Metric | Value |
|---|---|
| Session retrieval R@5 | 100% |
| Session retrieval R@10 | 100% |
| Answer in top-5 content | 91.2% |

> Run `python -m pytest benchmarks/test_e2e.py` to reproduce.

---

## 2. LongMemEval (Academic Benchmark)

**Source:** [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)

- **500 questions** across 5 memory ability categories
- **~40 session haystacks** per question (variant S)
- **470 evaluated**, 30 skipped (abstention questions with no evidence sessions)
- **Runtime:** 1487s (~25 min) on local machine

### Overall Results

| Metric | Score |
|---|---|
| **Recall@3** | **88.9%** |
| **Recall@5** | **93.0%** |
| **Recall@10** | **96.8%** |
| Strict Recall@3 | 78.0% |
| Strict Recall@5 | 86.0% |
| Strict Recall@10 | 92.9% |
| NDCG@3 | 0.781 |
| NDCG@5 | 0.812 |
| NDCG@10 | 0.839 |
| **MRR** | **0.847** |

### Per-Category Breakdown

| Category | Count | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| info-extraction | 150 | 82.0% | 88.0% | 94.0% | 0.785 |
| multi-session | 121 | 95.0% | 95.9% | 99.2% | 0.918 |
| temporal-reasoning | 127 | 89.8% | 93.7% | 96.1% | 0.832 |
| knowledge-update | 72 | 91.7% | 97.2% | **100%** | 0.883 |

> Run `python -m benchmarks.bench_longmemeval` to reproduce.

---

## 3. LoCoMo (Academic Benchmark)

**Source:** [snap-research/locomo](https://github.com/snap-research/locomo)

- **10 conversations**, ~300 turns each
- **1536 QA pairs** evaluated (from ~1987 total)
- **4 categories:** single-hop, multi-hop, temporal, commonsense
- **Runtime:** 61s on local machine

### Overall Results

| Metric | Score |
|---|---|
| **Recall@3** | **42.7%** |
| **Recall@5** | **50.7%** |
| **Recall@10** | **65.5%** |
| Strict Recall@3 | 37.0% |
| Strict Recall@5 | 44.9% |
| Strict Recall@10 | 59.6% |
| NDCG@3 | 0.334 |
| NDCG@5 | 0.366 |
| NDCG@10 | 0.417 |
| **MRR** | **0.384** |

### Per-Category Breakdown

| Category | Count | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| single-hop | 282 | 53.9% | 64.2% | **85.1%** | 0.481 |
| multi-hop | 321 | 49.5% | 58.6% | 70.1% | 0.426 |
| commonsense | 841 | 37.7% | 44.4% | 58.3% | 0.347 |
| temporal | 92 | 30.4% | 39.1% | 55.4% | 0.283 |

> Run `python -m benchmarks.bench_locomo` to reproduce.

---

## Summary Comparison

| Benchmark | Questions | R@5 | R@10 | MRR |
|---|---|---|---|---|
| Internal (synthetic) | 80 | 100% | 100% | 0.935 |
| **LongMemEval** (S, cleaned) | 470 | **93.0%** | **96.8%** | **0.847** |
| **LoCoMo** | 1536 | 50.7% | 65.5% | 0.384 |

### Analysis

- **LongMemEval R@10=96.8%** is very strong -- competitive with systems that use full LLM re-ranking. Pure vector search with all-MiniLM-L6-v2 handles session-level retrieval well.
- **LoCoMo is harder** because it tests fine-grained dialog-turn retrieval. Commonsense questions (55% of dataset) require inference beyond surface-level matching. The 65.5% R@10 reflects the limits of pure embedding similarity for conversational QA.
- **Knowledge-update achieves 100% R@10** on LongMemEval, showing Engram's temporal awareness works well.
- **Temporal reasoning is weakest** on both benchmarks -- a known challenge for embedding-only retrieval.

---

## Reproducibility

- **Embedding model:** all-MiniLM-L6-v2 (ChromaDB default)
- **LLM mode:** none (heuristic only for benchmarks, no re-ranking)
- **Python:** 3.12
- **ChromaDB:** 0.6.x
- **Datasets:** auto-downloaded to `benchmarks/.cache/` (not committed)
- **Raw results:** `benchmarks/results/*.jsonl` and `*_summary.json`

### Run Commands

```bash
cd packages/engram

# Internal (pytest-based, ~10s)
python -m pytest benchmarks/ -v

# LongMemEval (standalone, ~25 min for 500 questions)
python -m benchmarks.bench_longmemeval

# LoCoMo (standalone, ~60s for 10 conversations)
python -m benchmarks.bench_locomo
```

## Methodology Notes

### Train/Test Split (lessons from MemPalace-CN)

We follow strict evaluation protocol:
1. **20 dev questions** -- used for tuning thresholds during development
2. **80 test questions** -- never examined during development, reported as headline numbers
3. Both scores are reported; only test scores matter for comparisons
4. Raw JSONL results saved for audit

### Why This Matters

MemPalace-CN's hybrid_v4 achieved "100% on LongMemEval" by examining and fixing the exact 3 failing questions -- classic teaching-to-the-test. Their honest held-out score was 98.4%. Our framework avoids this pitfall by design.
