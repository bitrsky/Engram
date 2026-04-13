#!/usr/bin/env python3
"""
bench_longmemeval.py -- Evaluate Engram retrieval on the LongMemEval benchmark.

LongMemEval (Wu et al., 2024) tests five core memory abilities:
  1. Information Extraction   -- single-session-user / preference / assistant
  2. Multi-Session Reasoning  -- multi-session
  3. Knowledge Updates        -- knowledge-update
  4. Temporal Reasoning       -- temporal-reasoning
  5. Abstention               -- *_abs question IDs

This script:
  1. Downloads LongMemEval from HuggingFace (cached in .cache/)
  2. Ingests each question's haystack sessions into a fresh Engram index
  3. Runs vector_search for the question
  4. Computes Recall@K, NDCG@K, MRR at session level
  5. Breaks down by question_type
  6. Saves results to benchmarks/results/

Usage:
    python -m benchmarks.bench_longmemeval                    # run on _s variant
    python -m benchmarks.bench_longmemeval --variant oracle   # oracle (easy mode)
    python -m benchmarks.bench_longmemeval --limit 50         # first 50 questions
    python -m benchmarks.bench_longmemeval --k 5 10 20        # custom K values

Requires:  pip install engram[dev]
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Local imports
from .download_datasets import load_longmemeval
from .metrics import (
    RetrievalResult,
    aggregate_retrieval_results,
    format_category_table,
    format_retrieval_table,
    mean_reciprocal_rank,
    ndcg_at_k,
    per_category_breakdown,
    rank_of_first_hit,
    recall_at_k,
    save_results_jsonl,
    strict_recall_at_k,
)


# ===========================================================================
# Adapters: LongMemEval -> Engram
# ===========================================================================

# Map LongMemEval internal question_type names -> standardised category names
QUESTION_TYPE_MAP = {
    "single_hop": "info-extraction",
    "single-session-user": "info-extraction",
    "implicit_preference_v2": "info-extraction",
    "single-session-preference": "info-extraction",
    "assistant_previnfo": "info-extraction",
    "single-session-assistant": "info-extraction",
    "two_hop": "multi-session",
    "multi_session_synthesis": "multi-session",
    "multi-session": "multi-session",
    "temp_reasoning_implicit": "temporal-reasoning",
    "temp_reasoning_explicit": "temporal-reasoning",
    "temporal-reasoning": "temporal-reasoning",
    "knowledge_update": "knowledge-update",
    "knowledge-update": "knowledge-update",
}


def _normalise_category(raw_type: str, question_id: str) -> str:
    """Map LongMemEval question type to our standard categories."""
    # Abstention questions end with _abs
    if question_id.endswith("_abs"):
        return "abstention"
    return QUESTION_TYPE_MAP.get(raw_type, raw_type)


def _sessions_to_documents(entry: dict) -> List[Tuple[str, str]]:
    """
    Convert a LongMemEval entry's haystack_sessions into (session_id, text) pairs.

    Each session becomes one document. We concatenate user+assistant turns
    into a single text block, prefixed with the session date if available.
    """
    session_ids = entry.get("haystack_session_ids", [])
    sessions = entry.get("haystack_sessions", [])
    dates = entry.get("haystack_dates", [])

    documents = []
    for i, session_turns in enumerate(sessions):
        sid = session_ids[i] if i < len(session_ids) else f"session_{i}"
        date_str = dates[i] if i < len(dates) else ""

        lines = []
        if date_str:
            lines.append(f"[Session date: {date_str}]")

        for turn in session_turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")

        documents.append((sid, "\n".join(lines)))

    return documents


# ===========================================================================
# Core benchmark runner
# ===========================================================================

def evaluate_single_question(
    entry: dict,
    k_values: List[int],
    use_rerank: bool = False,
) -> Optional[dict]:
    """
    Evaluate Engram retrieval on a single LongMemEval question.

    Creates a temporary Engram index, ingests the haystack sessions,
    queries with the question, and computes metrics.

    Args:
        entry: LongMemEval question entry
        k_values: K values for recall metrics
        use_rerank: If True, use LLM reranking (requires LLM config)

    Returns a result dict or None if the question should be skipped.
    """
    # Lazy imports -- keep startup fast
    from engram.config import EngramConfig
    from engram.index import IndexManager
    from engram.store import write_memory

    question = entry.get("question", "")
    question_id = entry.get("question_id", "")
    question_type = entry.get("question_type", "")
    answer = entry.get("answer", "")
    answer_session_ids = set(entry.get("answer_session_ids", []))

    # Skip abstention questions for retrieval eval (no ground truth location)
    if question_id.endswith("_abs"):
        return None

    # Skip if no evidence sessions
    if not answer_session_ids:
        return None

    category = _normalise_category(question_type, question_id)
    documents = _sessions_to_documents(entry)

    if not documents:
        return None

    # --- Create temporary Engram instance ---
    tmpdir = tempfile.mkdtemp(prefix="engram_lme_")
    try:
        engram_dir = Path(tmpdir) / ".engram"
        engram_dir.mkdir()
        (engram_dir / "memories").mkdir()
        (engram_dir / "projects").mkdir()
        (engram_dir / "facts").mkdir()
        (engram_dir / ".index").mkdir()

        # Minimal config
        (engram_dir / "config.yaml").write_text(
            "llm:\n  provider: none\n", encoding="utf-8"
        )
        (engram_dir / "identity.md").write_text(
            "---\nname: LME Bench\n---\n", encoding="utf-8"
        )

        config = EngramConfig(base_dir=str(engram_dir))
        mgr = IndexManager(
            index_dir=config.index_dir,
            memories_dir=config.memories_dir,
        )

        # Ingest each session as a separate memory
        session_id_to_memory_id = {}
        for sid, text in documents:
            filepath = write_memory(
                content=text,
                project="longmemeval",
                topics=[],
                memory_type="conversation",
                source="longmemeval-benchmark",
                importance=3.0,
                memories_dir=config.memories_dir,
                memory_id=sid,
            )
            mid = mgr.index_memory(filepath)
            session_id_to_memory_id[sid] = mid

        # --- Query ---
        max_k = max(k_values)
        if use_rerank:
            hits = mgr.vector_search_reranked(
                query=question, config=config, n=max_k,
                candidates=config.rerank_candidates,
            )
        else:
            hits = mgr.vector_search(query=question, n=max_k)
        retrieved_ids = [h.id for h in hits]

        # Map memory IDs back to session IDs for comparison
        memory_to_session = {v: k for k, v in session_id_to_memory_id.items()}
        retrieved_session_ids = [
            memory_to_session.get(mid, mid) for mid in retrieved_ids
        ]

        # --- Metrics ---
        result = {
            "question_id": question_id,
            "question_type": question_type,
            "category": category,
            "question": question,
            "answer": answer,
            "n_sessions": len(documents),
            "n_evidence": len(answer_session_ids),
        }

        for k in k_values:
            result[f"recall@{k}"] = recall_at_k(
                retrieved_session_ids, answer_session_ids, k
            )
            result[f"strict_recall@{k}"] = strict_recall_at_k(
                retrieved_session_ids, answer_session_ids, k
            )
            result[f"ndcg@{k}"] = ndcg_at_k(
                retrieved_session_ids, answer_session_ids, k
            )

        result["mrr"] = mean_reciprocal_rank(
            retrieved_session_ids, answer_session_ids
        )
        result["first_hit_rank"] = rank_of_first_hit(
            retrieved_session_ids, answer_session_ids
        )
        result["top_retrieved"] = retrieved_session_ids[:5]

        mgr.close()
        return result

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_benchmark(
    variant: str = "s",
    k_values: List[int] = None,
    limit: int = 0,
    skip_types: List[str] = None,
    use_rerank: bool = False,
) -> Tuple[List[dict], dict]:
    """
    Run the full LongMemEval benchmark.

    Args:
        variant: "s", "m", or "oracle"
        k_values: List of K values for Recall@K (default: [3, 5, 10])
        limit: Max questions to evaluate (0 = all)
        skip_types: Question types to skip
        use_rerank: If True, use LLM reranking

    Returns:
        (results_list, summary_dict)
    """
    if k_values is None:
        k_values = [3, 5, 10]
    if skip_types is None:
        skip_types = []

    print(f"\n{'='*60}")
    print(f"LongMemEval Benchmark -- Engram Retrieval Evaluation")
    print(f"{'='*60}")
    print(f"Variant:    longmemeval_{variant}")
    print(f"K values:   {k_values}")
    print(f"Limit:      {limit or 'all'}")
    print(f"Rerank:     {'ON' if use_rerank else 'OFF'}")
    print(f"Time:       {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    # Load dataset
    print("Loading LongMemEval dataset...")
    data = load_longmemeval(variant)
    print(f"  Loaded {len(data)} questions")

    # Filter
    if limit > 0:
        data = data[:limit]

    if skip_types:
        data = [e for e in data if e.get("question_type") not in skip_types]
        print(f"  After skip filter: {len(data)} questions")

    # Run evaluation
    results = []
    skipped = 0
    start_time = time.time()

    for i, entry in enumerate(data):
        qid = entry.get("question_id", f"q{i}")
        qtype = entry.get("question_type", "unknown")

        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(data)}] {rate:.1f} q/s -- {qid} ({qtype})")

        result = evaluate_single_question(entry, k_values, use_rerank=use_rerank)
        if result is None:
            skipped += 1
            continue
        results.append(result)

    elapsed_total = time.time() - start_time

    # -- Aggregate --
    print(f"\nEvaluated {len(results)} questions, skipped {skipped}")
    print(f"Total time: {elapsed_total:.1f}s ({len(results)/elapsed_total:.1f} q/s)\n")

    # Per-category stats
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    summary = {
        "variant": variant,
        "total_questions": len(data),
        "evaluated": len(results),
        "skipped": skipped,
        "rerank": use_rerank,
        "elapsed_seconds": round(elapsed_total, 1),
        "timestamp": datetime.now().isoformat(),
        "overall": {},
        "by_category": {},
    }

    # Overall metrics
    for k in k_values:
        key = f"recall@{k}"
        vals = [r[key] for r in results]
        summary["overall"][key] = round(sum(vals) / len(vals), 4) if vals else 0

        key = f"strict_recall@{k}"
        vals = [r[key] for r in results]
        summary["overall"][key] = round(sum(vals) / len(vals), 4) if vals else 0

        key = f"ndcg@{k}"
        vals = [r[key] for r in results]
        summary["overall"][key] = round(sum(vals) / len(vals), 4) if vals else 0

    mrr_vals = [r["mrr"] for r in results]
    summary["overall"]["mrr"] = round(sum(mrr_vals) / len(mrr_vals), 4) if mrr_vals else 0

    # Per-category metrics
    for cat, cat_results in sorted(by_category.items()):
        cat_summary = {"count": len(cat_results)}
        for k in k_values:
            key = f"recall@{k}"
            vals = [r[key] for r in cat_results]
            cat_summary[key] = round(sum(vals) / len(vals), 4) if vals else 0
        mrr_vals = [r["mrr"] for r in cat_results]
        cat_summary["mrr"] = round(sum(mrr_vals) / len(mrr_vals), 4) if mrr_vals else 0
        summary["by_category"][cat] = cat_summary

    # -- Print report --
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    # Overall table
    print("Overall:")
    for metric, val in sorted(summary["overall"].items()):
        print(f"  {metric:25s} {val:.4f} ({val*100:.1f}%)")

    # Per-category table
    print(f"\nPer-Category Recall@{k_values[-1]}:")
    print(f"  {'Category':25s} {'Count':>6s} {'R@'+str(k_values[-1]):>8s} {'MRR':>8s}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8}")
    for cat, stats in sorted(summary["by_category"].items()):
        r_k = stats.get(f"recall@{k_values[-1]}", 0)
        mrr = stats.get("mrr", 0)
        print(f"  {cat:25s} {stats['count']:6d} {r_k:8.4f} {mrr:8.4f}")

    # -- Save results --
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"longmemeval_{variant}_{ts}.jsonl"
    summary_file = results_dir / f"longmemeval_{variant}_{ts}_summary.json"

    with open(results_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  {results_file}")
    print(f"  {summary_file}")

    return results, summary


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Engram retrieval on LongMemEval"
    )
    parser.add_argument(
        "--variant",
        default="s",
        choices=["s", "m", "oracle"],
        help="LongMemEval variant (default: s)",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        help="K values for Recall@K (default: 3 5 10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max questions to evaluate (0 = all)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable LLM reranking (requires LLM config)",
    )
    args = parser.parse_args()
    run_benchmark(
        variant=args.variant, k_values=args.k, limit=args.limit,
        use_rerank=args.rerank,
    )


if __name__ == "__main__":
    main()
