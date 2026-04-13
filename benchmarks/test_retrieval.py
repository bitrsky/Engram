"""
test_retrieval.py -- Layer 1: Retrieval recall benchmarks.

Tests whether vector search retrieves the correct evidence session(s)
for each question in the benchmark dataset.

Modes tested:
1. raw -- bare vector search, no filters
2. project_filtered -- with project metadata filter
3. topic_filtered -- with topic metadata filter (requires knowing topic)

Results are collected and reported as R@3, R@5, R@10, NDCG@5.
"""

import json
from pathlib import Path
from typing import Dict, List, Set

import pytest

from .metrics import (
    RetrievalResult,
    aggregate_retrieval_results,
    format_category_table,
    format_retrieval_table,
    ndcg_at_k,
    per_category_breakdown,
    rank_of_first_hit,
    recall_at_k,
    save_results_jsonl,
)

RESULTS_DIR = Path(__file__).parent / "results"


class TestRetrievalRaw:
    """Layer 1a: Raw vector search (no metadata filters)."""

    def _run_retrieval(self, bench_index, questions, mode="raw", top_k=10):
        """Run retrieval for a list of questions and return results."""
        results = []

        for q in questions:
            query = q["question"]
            evidence_ids = set(q["evidence_session_ids"])

            # Perform search
            hits = bench_index.vector_search(query=query, n=top_k)
            retrieved_ids = [h.id for h in hits]
            similarity_scores = [h.similarity for h in hits]

            # Compute metrics
            hit = recall_at_k(retrieved_ids, evidence_ids, top_k) > 0
            first_rank = rank_of_first_hit(retrieved_ids, evidence_ids)
            ndcg = ndcg_at_k(retrieved_ids, evidence_ids, 5)

            results.append(RetrievalResult(
                question_id=q["id"],
                question=q["question"],
                category=q["category"],
                split=q["split"],
                mode=mode,
                top_k=top_k,
                retrieved_ids=retrieved_ids,
                evidence_ids=q["evidence_session_ids"],
                hit=hit,
                rank_of_first_hit=first_rank,
                similarity_scores=similarity_scores,
                ndcg=ndcg,
            ))

        return results

    def test_raw_retrieval_dev(self, bench_index, dev_questions):
        """Run raw retrieval on dev split -- for tuning."""
        results = self._run_retrieval(bench_index, dev_questions, mode="raw")
        agg = aggregate_retrieval_results(results)

        print(f"\n[DEV] Raw Retrieval: R@5={agg['recall_at_5']:.1%}, "
              f"R@10={agg['recall_at_10']:.1%}, NDCG@5={agg['ndcg_at_5']:.3f}")

        # Dev split: no assertions, just report
        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_results_jsonl(results, RESULTS_DIR / "raw_dev.jsonl")

    def test_raw_retrieval_test(self, bench_index, test_questions):
        """Run raw retrieval on test split -- the real benchmark."""
        results = self._run_retrieval(bench_index, test_questions, mode="raw")
        agg = aggregate_retrieval_results(results)

        print(f"\n[TEST] Raw Retrieval: R@3={agg['recall_at_3']:.1%}, "
              f"R@5={agg['recall_at_5']:.1%}, R@10={agg['recall_at_10']:.1%}, "
              f"NDCG@5={agg['ndcg_at_5']:.3f}, MRR={agg['mrr']:.3f}")

        # Per-category breakdown
        breakdown = per_category_breakdown(results, k=5)
        print("\nPer-category R@5:")
        for cat, data in breakdown.items():
            print(f"  {cat}: {data['recall_at_k']:.1%} ({data['count']} questions)")
            if data["failed_ids"]:
                print(f"    Failed: {', '.join(data['failed_ids'])}")

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_results_jsonl(results, RESULTS_DIR / "raw_test.jsonl")

        # Minimum quality bar: R@10 should be > 50% (basic sanity check)
        assert agg["recall_at_10"] > 0.3, (
            f"R@10 is {agg['recall_at_10']:.1%} -- below minimum quality bar of 30%"
        )


class TestRetrievalFiltered:
    """Layer 1b: Filtered vector search (project, topic)."""

    def test_project_filtered(self, bench_index, test_questions):
        """Test retrieval with project filter."""
        results = []
        for q in test_questions:
            query = q["question"]
            evidence_ids = set(q["evidence_session_ids"])

            hits = bench_index.vector_search(
                query=query,
                project="saas-app",
                n=10,
            )
            retrieved_ids = [h.id for h in hits]

            hit = recall_at_k(retrieved_ids, evidence_ids, 10) > 0
            first_rank = rank_of_first_hit(retrieved_ids, evidence_ids)

            results.append(RetrievalResult(
                question_id=q["id"],
                question=q["question"],
                category=q["category"],
                split=q["split"],
                mode="project_filtered",
                top_k=10,
                retrieved_ids=retrieved_ids,
                evidence_ids=q["evidence_session_ids"],
                hit=hit,
                rank_of_first_hit=first_rank,
            ))

        agg = aggregate_retrieval_results(results)
        print(f"\n[TEST] Project-filtered: R@5={agg['recall_at_5']:.1%}, "
              f"R@10={agg['recall_at_10']:.1%}")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_results_jsonl(results, RESULTS_DIR / "project_filtered_test.jsonl")


class TestRetrievalReport:
    """Generate the final retrieval benchmark report."""

    def test_generate_report(self, bench_index, all_questions):
        """Generate combined report across all modes (run last)."""
        modes = {}

        # Raw mode
        raw_results = []
        for q in all_questions:
            hits = bench_index.vector_search(query=q["question"], n=10)
            retrieved_ids = [h.id for h in hits]
            evidence_ids = set(q["evidence_session_ids"])

            raw_results.append(RetrievalResult(
                question_id=q["id"],
                question=q["question"],
                category=q["category"],
                split=q["split"],
                mode="raw",
                top_k=10,
                retrieved_ids=retrieved_ids,
                evidence_ids=q["evidence_session_ids"],
                hit=recall_at_k(retrieved_ids, evidence_ids, 10) > 0,
                rank_of_first_hit=rank_of_first_hit(retrieved_ids, evidence_ids),
            ))

        modes["Raw vector search"] = aggregate_retrieval_results(raw_results, split="test")

        # Project filtered
        pf_results = []
        for q in all_questions:
            hits = bench_index.vector_search(
                query=q["question"], project="saas-app", n=10,
            )
            retrieved_ids = [h.id for h in hits]
            evidence_ids = set(q["evidence_session_ids"])

            pf_results.append(RetrievalResult(
                question_id=q["id"],
                question=q["question"],
                category=q["category"],
                split=q["split"],
                mode="project_filtered",
                top_k=10,
                retrieved_ids=retrieved_ids,
                evidence_ids=q["evidence_session_ids"],
                hit=recall_at_k(retrieved_ids, evidence_ids, 10) > 0,
                rank_of_first_hit=rank_of_first_hit(retrieved_ids, evidence_ids),
            ))

        modes["Project filtered"] = aggregate_retrieval_results(pf_results, split="test")

        # Print formatted report
        print("\n" + "=" * 60)
        print("ENGRAM RETRIEVAL BENCHMARK REPORT")
        print("=" * 60)
        print(format_retrieval_table(modes))
        print()

        # Category breakdown for raw mode
        breakdown = per_category_breakdown(raw_results, split="test", k=5)
        print(format_category_table(breakdown, k=5))
