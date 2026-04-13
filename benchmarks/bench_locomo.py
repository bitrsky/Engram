#!/usr/bin/env python3
"""
bench_locomo.py -- Evaluate Engram retrieval on the LoCoMo benchmark.

LoCoMo (Maharana et al., 2024) contains 10 long-term conversations with:
  - ~300 turns, ~9K tokens, up to 35 sessions per conversation
  - QA pairs in 5 categories: single-hop, multi-hop, temporal,
    commonsense/world-knowledge, adversarial
  - Evidence dialog IDs linking questions to ground-truth turns

This script:
  1. Downloads LoCoMo from GitHub (cached in .cache/)
  2. For each conversation: ingests all sessions into an Engram index
  3. For each QA pair: queries Engram and checks if the evidence turn's
     session appears in the top-K retrieved sessions
  4. Computes Recall@K, NDCG@K, MRR -- overall and per QA category
  5. Saves results to benchmarks/results/

Usage:
    python -m benchmarks.bench_locomo                   # vector search only
    python -m benchmarks.bench_locomo --limit 5         # first 5 conversations
    python -m benchmarks.bench_locomo --k 3 5 10 20     # custom K values
    python -m benchmarks.bench_locomo --deep --think MODEL  # deep search

Requires:  pip install engram[dev]
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Local imports
from .download_datasets import load_locomo
from .metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    rank_of_first_hit,
    recall_at_k,
    strict_recall_at_k,
)


# ===========================================================================
# Adapters: LoCoMo -> Engram
# ===========================================================================

# LoCoMo QA categories (1-indexed in the dataset)
LOCOMO_CATEGORIES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "commonsense",
    5: "adversarial",
    "single-hop": "single-hop",
    "multi-hop": "multi-hop",
    "temporal": "temporal",
    "open-ended": "commonsense",
    "adversarial": "adversarial",
}


def _normalise_category(raw_cat) -> str:
    """Map LoCoMo category (int or str) to standard name."""
    if isinstance(raw_cat, int):
        return LOCOMO_CATEGORIES.get(raw_cat, f"category-{raw_cat}")
    return LOCOMO_CATEGORIES.get(str(raw_cat).lower(), str(raw_cat))


def _extract_sessions(conversation: dict) -> List[Tuple[str, str, str]]:
    """
    Extract sessions from a LoCoMo conversation.

    Returns list of (session_key, date_str, text) tuples.
    Session keys are like "session_1", "session_2", etc.
    """
    sessions = []

    session_keys = sorted(
        [k for k in conversation.keys()
         if re.match(r"session_\d+$", k)],
        key=lambda x: int(x.split("_")[1])
    )

    for skey in session_keys:
        turns = conversation[skey]
        if not isinstance(turns, list):
            continue

        date_key = f"{skey}_date_time"
        date_str = conversation.get(date_key, "")

        lines = []
        if date_str:
            lines.append(f"[Session date: {date_str}]")

        for turn in turns:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            if text:
                lines.append(f"{speaker}: {text}")

        if lines:
            sessions.append((skey, date_str, "\n".join(lines)))

    return sessions


def _evidence_to_session_ids(evidence_ids: List[str]) -> Set[str]:
    """
    Convert LoCoMo evidence dialog IDs to session IDs.

    Evidence IDs are like "D1:3", "D5:2" where number after D = session number.
    We map "D1:3" -> "session_1", "D5:2" -> "session_5", etc.
    """
    session_ids = set()
    for eid in evidence_ids:
        eid = eid.strip()
        if eid.startswith("D"):
            parts = eid[1:].split(":")
            if parts and parts[0].isdigit():
                session_ids.add(f"session_{parts[0]}")
        else:
            parts = eid.replace(":", "_").split("_")
            if parts and parts[0].isdigit():
                session_ids.add(f"session_{parts[0]}")
    return session_ids


def _build_result(
    sample_id, question, answer, category, evidence_session_ids,
    hits, sessions, memory_to_session, k_values,
):
    """Build a metrics result dict from hits."""
    retrieved_ids = [h.id for h in hits]
    retrieved_session_ids = [
        memory_to_session.get(mid, mid) for mid in retrieved_ids
    ]

    result = {
        "sample_id": sample_id,
        "question": question,
        "answer": answer,
        "category": category,
        "n_sessions": len(sessions),
        "n_evidence": len(evidence_session_ids),
        "evidence_sessions": sorted(evidence_session_ids),
    }

    for k in k_values:
        result[f"recall@{k}"] = recall_at_k(
            retrieved_session_ids, evidence_session_ids, k
        )
        result[f"strict_recall@{k}"] = strict_recall_at_k(
            retrieved_session_ids, evidence_session_ids, k
        )
        result[f"ndcg@{k}"] = ndcg_at_k(
            retrieved_session_ids, evidence_session_ids, k
        )

    result["mrr"] = mean_reciprocal_rank(
        retrieved_session_ids, evidence_session_ids
    )
    result["first_hit_rank"] = rank_of_first_hit(
        retrieved_session_ids, evidence_session_ids
    )
    result["top_retrieved"] = retrieved_session_ids[:5]

    return result


# ===========================================================================
# Core benchmark runner
# ===========================================================================

def evaluate_conversation(
    sample: dict,
    k_values: List[int],
    use_deep_search: bool = False,
    think_model: str = "",
) -> List[dict]:
    """
    Evaluate Engram retrieval on all QA pairs from one LoCoMo conversation.

    Creates a temporary Engram index, ingests all sessions from the
    conversation, then evaluates each QA pair.

    Args:
        sample: LoCoMo conversation sample
        k_values: K values for recall metrics
        use_deep_search: If True, use LLM agent deep search (requires think_model)
        think_model: Model name for deep search (e.g. 'anthropic/claude-haiku-3-5-20241022')

    Returns list of per-question result dicts.
    """
    from engram.config import EngramConfig
    from engram.index import IndexManager
    from engram.store import write_memory

    conversation = sample.get("conversation", {})
    qa_pairs = sample.get("qa", [])
    sample_id = sample.get("sample_id", "unknown")

    if not qa_pairs:
        return []

    # Extract sessions
    sessions = _extract_sessions(conversation)
    if not sessions:
        print(f"    ! No sessions found in {sample_id}")
        return []

    # --- Create temporary Engram instance ---
    tmpdir = tempfile.mkdtemp(prefix="engram_locomo_")
    results = []

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
            "---\nname: LoCoMo Bench\n---\n", encoding="utf-8"
        )

        config = EngramConfig(base_dir=str(engram_dir))

        mgr = IndexManager(
            index_dir=config.index_dir,
            memories_dir=config.memories_dir,
            facts_dir=config.facts_dir,
            projects_dir=config.projects_dir,
        )

        # Ingest sessions
        session_key_to_memory_id = {}
        for skey, date_str, text in sessions:
            filepath = write_memory(
                content=text,
                project="locomo",
                topics=[],
                memory_type="conversation",
                source="locomo-benchmark",
                importance=3.0,
                memories_dir=config.memories_dir,
                memory_id=skey,
            )
            mid = mgr.index_memory(filepath)
            session_key_to_memory_id[skey] = mid

        memory_to_session = {v: k for k, v in session_key_to_memory_id.items()}
        max_k = max(k_values)

        # Prepare QA tasks (skip adversarial — no evidence)
        qa_tasks = []
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            raw_category = qa.get("category", "unknown")
            evidence_ids = qa.get("evidence", [])

            category = _normalise_category(raw_category)
            if category == "adversarial":
                continue

            evidence_session_ids = _evidence_to_session_ids(evidence_ids)
            if not evidence_session_ids:
                continue

            qa_tasks.append((question, answer, category, evidence_session_ids))

        # --- Vector search only (no LLM) ---
        if not use_deep_search:
            for task in qa_tasks:
                question, answer, category, evidence_session_ids = task
                hits = mgr.vector_search(query=question, n=max_k)
                results.append(_build_result(
                    sample_id, question, answer, category, evidence_session_ids,
                    hits, sessions, memory_to_session, k_values,
                ))

        # --- Deep search (LLM agent) ---
        else:
            import asyncio as _aio
            from engram.llm import build_deep_search_prompt, parse_deep_search_response
            from benchmarks.think_adapter import create_think_fn

            think_fn = create_think_fn(model=think_model)

            async def _eval_one_deep(task, sem):
                question, answer, category, evidence_session_ids = task

                # Vector search for hints
                raw_hits = mgr.vector_search(query=question, n=max_k)

                # Build prompt from llm.py
                system, user_prompt = build_deep_search_prompt(
                    query=question,
                    base_dir=str(Path(config.base_dir).resolve()),
                    vector_hits=raw_hits,
                )

                # Agentic deep search via ThinkFn
                async with sem:
                    deep_answer = await think_fn(user_prompt, system=system)

                parsed = parse_deep_search_response(deep_answer or "")

                if parsed["found"]:
                    # Extract session references from LLM response
                    found_sessions = set(re.findall(r'session_\d+', deep_answer or ""))
                    seen = set()
                    hits = []
                    for sk in sorted(found_sessions):
                        mid = session_key_to_memory_id.get(sk)
                        if mid:
                            matched = [h for h in raw_hits if h.id == mid]
                            if matched:
                                hits.append(matched[0])
                            else:
                                from engram.index import SearchHit
                                hits.append(SearchHit(
                                    id=mid, content="", similarity=1.0,
                                    project="locomo", topics=[], memory_type="conversation",
                                    importance=3.0, created="", file_path="",
                                ))
                            seen.add(mid)
                    # Fill remaining from vector hits
                    for h in raw_hits:
                        if h.id not in seen:
                            hits.append(h)
                            seen.add(h.id)
                    hits = hits[:max_k]
                else:
                    hits = raw_hits[:max_k]

                return _build_result(
                    sample_id, question, answer, category, evidence_session_ids,
                    hits, sessions, memory_to_session, k_values,
                )

            sem = _aio.Semaphore(10)

            async def _run_all():
                tasks = [_eval_one_deep(t, sem) for t in qa_tasks]
                return await _aio.gather(*tasks, return_exceptions=True)

            raw_results = _aio.run(_run_all())
            results = [r for r in raw_results if not isinstance(r, Exception)]
            errors = [r for r in raw_results if isinstance(r, Exception)]
            if errors:
                print(f"    ! {len(errors)} QA pairs failed: {errors[0]}")

        mgr.close()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return results


def run_benchmark(
    k_values: Optional[List[int]] = None,
    limit: int = 0,
    think_model: str = "",
    use_deep_search: bool = False,
) -> Tuple[List[dict], dict]:
    """
    Run the full LoCoMo benchmark.

    Args:
        k_values: K values for Recall@K (default: [3, 5, 10])
        limit: Max conversations to evaluate (0 = all)
        think_model: Model name for deep search. Empty = no LLM.
        use_deep_search: If True, use LLM agent deep search (requires think_model).

    Returns:
        (results_list, summary_dict)
    """
    if k_values is None:
        k_values = [3, 5, 10]

    print(f"\n{'='*60}")
    print(f"LoCoMo Benchmark -- Engram Retrieval Evaluation")
    print(f"{'='*60}")
    print(f"K values:       {k_values}")
    print(f"Limit:          {limit or 'all (10 conversations)'}")
    print(f"Mode:           {'Deep Search (LLM agent)' if use_deep_search else 'Vector Search Only'}")
    if use_deep_search:
        print(f"Model:          {think_model}")
    print(f"Time:           {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    # Load dataset
    print("Loading LoCoMo dataset...")
    data = load_locomo()
    print(f"  Loaded {len(data)} conversations")

    if limit > 0:
        data = data[:limit]

    # Run evaluation
    all_results = []
    start_time = time.time()

    for i, sample in enumerate(data):
        sample_id = sample.get("sample_id", f"conv_{i}")
        n_qa = len(sample.get("qa", []))
        print(f"  [{i+1}/{len(data)}] {sample_id} -- {n_qa} QA pairs")

        conv_results = evaluate_conversation(
            sample, k_values,
            use_deep_search=use_deep_search,
            think_model=think_model,
        )
        print(f"           -> {len(conv_results)} evaluated")
        all_results.extend(conv_results)

    elapsed_total = time.time() - start_time

    # -- Aggregate --
    print(f"\nTotal: {len(all_results)} QA pairs evaluated")
    print(f"Time:  {elapsed_total:.1f}s\n")

    # Per-category
    by_category = defaultdict(list)
    for r in all_results:
        by_category[r["category"]].append(r)

    summary = {
        "dataset": "locomo",
        "total_qa": len(all_results),
        "n_conversations": len(data),
        "deep_search": use_deep_search,
        "think_model": think_model or "(none)",
        "elapsed_seconds": round(elapsed_total, 1),
        "timestamp": datetime.now().isoformat(),
        "overall": {},
        "by_category": {},
    }

    # Overall
    for k in k_values:
        for metric_name in [f"recall@{k}", f"strict_recall@{k}", f"ndcg@{k}"]:
            vals = [r[metric_name] for r in all_results]
            summary["overall"][metric_name] = (
                round(sum(vals) / len(vals), 4) if vals else 0
            )

    mrr_vals = [r["mrr"] for r in all_results]
    summary["overall"]["mrr"] = (
        round(sum(mrr_vals) / len(mrr_vals), 4) if mrr_vals else 0
    )

    # Per-category
    for cat, cat_results in sorted(by_category.items()):
        cat_summary = {"count": len(cat_results)}
        for k in k_values:
            key = f"recall@{k}"
            vals = [r[key] for r in cat_results]
            cat_summary[key] = round(sum(vals) / len(vals), 4) if vals else 0.0
        mrr_vals = [r["mrr"] for r in cat_results]
        cat_summary["mrr"] = round(sum(mrr_vals) / len(mrr_vals), 4) if mrr_vals else 0.0
        summary["by_category"][cat] = cat_summary

    # -- Print report --
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    print("Overall:")
    for metric, val in sorted(summary["overall"].items()):
        print(f"  {metric:25s} {val:.4f} ({val*100:.1f}%)")

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
    results_file = results_dir / f"locomo_{ts}.jsonl"
    summary_file = results_dir / f"locomo_{ts}_summary.json"

    with open(results_file, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  {results_file}")
    print(f"  {summary_file}")

    return all_results, summary


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Engram retrieval on LoCoMo"
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
        help="Max conversations to evaluate (0 = all 10)",
    )
    parser.add_argument(
        "--think",
        type=str,
        default="",
        metavar="MODEL",
        help="Model for deep search (e.g. anthropic/claude-haiku-3-5-20241022)",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Enable LLM deep search (requires --think). LLM reads memory files directly.",
    )
    args = parser.parse_args()
    run_benchmark(
        k_values=args.k,
        limit=args.limit,
        think_model=args.think,
        use_deep_search=args.deep,
    )


if __name__ == "__main__":
    main()
