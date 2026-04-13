"""
test_e2e.py — Layer 4: End-to-end benchmark.

Full pipeline: ingest conversations -> search -> check answer in results.
Tests the complete Engram system from ingestion to retrieval.
"""

import pytest

from .conftest import conversation_to_text, load_conversations, load_questions
from .metrics import (
    RetrievalResult,
    aggregate_retrieval_results,
    ndcg_at_k,
    per_category_breakdown,
    rank_of_first_hit,
    recall_at_k,
)


class TestEndToEnd:
    """Full pipeline integration benchmark."""

    def test_e2e_remember_and_search(self, bench_config, bench_index, test_questions):
        """
        Test the full pipeline: memories are already ingested via bench_index fixture.
        Now search and verify evidence sessions appear in results.
        """
        results = []
        answer_in_content = 0

        for q in test_questions:
            query = q["question"]
            evidence_ids = set(q["evidence_session_ids"])
            answer = q["answer"]
            aliases = q.get("answer_aliases", [])

            # Search
            hits = bench_index.vector_search(query=query, n=10)
            retrieved_ids = [h.id for h in hits]

            # Check if answer text appears in any retrieved content
            answer_found = False
            all_answers = [answer.lower()] + [a.lower() for a in aliases]
            for hit in hits[:5]:
                content_lower = hit.content.lower()
                if any(ans in content_lower for ans in all_answers):
                    answer_found = True
                    break

            if answer_found:
                answer_in_content += 1

            hit_at_10 = recall_at_k(retrieved_ids, evidence_ids, 10) > 0
            first_rank = rank_of_first_hit(retrieved_ids, evidence_ids)

            results.append(RetrievalResult(
                question_id=q["id"],
                question=query,
                category=q["category"],
                split=q["split"],
                mode="e2e",
                top_k=10,
                retrieved_ids=retrieved_ids,
                evidence_ids=q["evidence_session_ids"],
                hit=hit_at_10,
                rank_of_first_hit=first_rank,
            ))

        agg = aggregate_retrieval_results(results)
        answer_rate = answer_in_content / len(test_questions) if test_questions else 0

        print(f"\n[E2E] Results:")
        print(f"  Session retrieval R@5:  {agg['recall_at_5']:.1%}")
        print(f"  Session retrieval R@10: {agg['recall_at_10']:.1%}")
        print(f"  Answer in top-5 content: {answer_rate:.1%}")

        # Category breakdown
        breakdown = per_category_breakdown(results, k=5)
        print("\n  Per-category R@5:")
        for cat, data in breakdown.items():
            print(f"    {cat}: {data['recall_at_k']:.1%} ({data['count']})")

    def test_e2e_adversarial_questions(self, bench_index, all_questions):
        """
        Adversarial questions should still retrieve relevant context,
        even if the answer is "no" or requires disambiguation.
        """
        adversarial = [q for q in all_questions if q["category"] == "adversarial"]

        hits_found = 0
        for q in adversarial:
            evidence_ids = set(q["evidence_session_ids"])
            hits = bench_index.vector_search(query=q["question"], n=10)
            retrieved_ids = [h.id for h in hits]

            if recall_at_k(retrieved_ids, evidence_ids, 10) > 0:
                hits_found += 1

        rate = hits_found / len(adversarial) if adversarial else 0
        print(f"\n[E2E Adversarial] {hits_found}/{len(adversarial)} found evidence ({rate:.1%})")

    def test_e2e_temporal_questions(self, bench_index, all_questions):
        """
        Temporal questions often need multiple sessions.
        Check if at least one evidence session is found.
        """
        temporal = [q for q in all_questions if q["category"] == "temporal"]

        hits_found = 0
        for q in temporal:
            evidence_ids = set(q["evidence_session_ids"])
            hits = bench_index.vector_search(query=q["question"], n=10)
            retrieved_ids = [h.id for h in hits]

            if recall_at_k(retrieved_ids, evidence_ids, 10) > 0:
                hits_found += 1

        rate = hits_found / len(temporal) if temporal else 0
        print(f"\n[E2E Temporal] {hits_found}/{len(temporal)} found evidence ({rate:.1%})")
