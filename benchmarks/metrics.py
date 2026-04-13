"""
metrics.py -- Benchmark metrics for Engram.

Core metrics:
- Recall@K: fraction of questions where >=1 evidence doc appears in top-K
- NDCG@K: Normalized Discounted Cumulative Gain
- Precision/Recall/F1 for classification tasks (quality gate, dedup, conflicts)
- Per-category breakdown with failed question tracking
"""

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ===========================================================================
# Retrieval Metrics
# ===========================================================================


def recall_at_k(retrieved_ids: List[str], evidence_ids: Set[str], k: int) -> float:
    """
    Recall@K: does at least one evidence document appear in the top-K results?

    Returns 1.0 if yes, 0.0 if no.  For single-evidence questions this is
    equivalent to Hit@K.  For multi-evidence questions, this is a soft recall
    (any hit counts).

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (best first)
        evidence_ids: Set of ground-truth evidence document IDs
        k: Cutoff

    Returns:
        1.0 if any evidence doc in top-K, else 0.0
    """
    if not evidence_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & evidence_ids else 0.0


def strict_recall_at_k(retrieved_ids: List[str], evidence_ids: Set[str], k: int) -> float:
    """
    Strict Recall@K: fraction of evidence documents found in top-K.

    For multi-session questions where ALL evidence sessions must be found.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        evidence_ids: Set of ground-truth evidence document IDs
        k: Cutoff

    Returns:
        Fraction of evidence_ids found in top-K (0.0 to 1.0)
    """
    if not evidence_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    found = top_k & evidence_ids
    return len(found) / len(evidence_ids)


def ndcg_at_k(retrieved_ids: List[str], evidence_ids: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    Binary relevance: 1 if id in evidence_ids, else 0.
    Ideal ranking has all evidence docs at positions 1..len(evidence_ids).

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        evidence_ids: Set of ground-truth evidence document IDs
        k: Cutoff

    Returns:
        NDCG score (0.0 to 1.0)
    """
    if not evidence_ids:
        return 0.0

    # DCG for actual ranking
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in evidence_ids:
            # Binary relevance: rel = 1
            dcg += 1.0 / math.log2(i + 2)  # i+2 because position is 1-indexed

    # Ideal DCG: all evidence docs at top positions
    n_relevant = min(len(evidence_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def mean_reciprocal_rank(retrieved_ids: List[str], evidence_ids: Set[str]) -> float:
    """
    Mean Reciprocal Rank: 1/rank of first relevant document.

    Returns 0.0 if no relevant document found.
    """
    if not evidence_ids:
        return 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in evidence_ids:
            return 1.0 / (i + 1)
    return 0.0


def rank_of_first_hit(retrieved_ids: List[str], evidence_ids: Set[str]) -> Optional[int]:
    """
    Return 1-indexed rank of first evidence document in results.

    Returns None if no evidence doc found.
    """
    if not evidence_ids:
        return None
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in evidence_ids:
            return i + 1
    return None


# ===========================================================================
# Classification Metrics (quality gate, dedup, conflicts)
# ===========================================================================


@dataclass
class ClassificationMetrics:
    """Precision, recall, F1 for a binary classification task."""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = (self.true_positives + self.true_negatives +
                 self.false_positives + self.false_negatives)
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0

    def add(self, predicted: bool, actual: bool):
        """Record a single prediction."""
        if predicted and actual:
            self.true_positives += 1
        elif predicted and not actual:
            self.false_positives += 1
        elif not predicted and actual:
            self.false_negatives += 1
        else:
            self.true_negatives += 1


def compute_classification_metrics(
    predictions: List[bool],
    labels: List[bool],
) -> ClassificationMetrics:
    """Compute classification metrics from parallel lists of predictions and labels."""
    metrics = ClassificationMetrics()
    for pred, label in zip(predictions, labels):
        metrics.add(pred, label)
    return metrics


def per_class_accuracy(
    predictions: List[str],
    labels: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class accuracy for multi-class classification.

    Returns:
        {
            "class_name": {"correct": N, "total": N, "accuracy": 0.X},
            ...
        }
    """
    class_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for pred, label in zip(predictions, labels):
        class_stats[label]["total"] += 1
        if pred == label:
            class_stats[label]["correct"] += 1

    result = {}
    for cls, stats in class_stats.items():
        result[cls] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
        }
    return result


# ===========================================================================
# Aggregation & Reporting
# ===========================================================================


@dataclass
class RetrievalResult:
    """Result of a single retrieval question."""
    question_id: str
    question: str
    category: str
    split: str  # "dev" or "test"
    mode: str  # "raw", "project_filtered", etc.
    top_k: int
    retrieved_ids: List[str]
    evidence_ids: List[str]  # stored as list for JSON serialization
    hit: bool  # recall@k == 1.0
    rank_of_first_hit: Optional[int]
    similarity_scores: List[float] = field(default_factory=list)
    ndcg: float = 0.0
    mrr: float = 0.0


def aggregate_retrieval_results(
    results: List[RetrievalResult],
    split: Optional[str] = None,
) -> Dict[str, float]:
    """
    Aggregate retrieval results into summary metrics.

    Args:
        results: List of per-question results
        split: If specified, filter to only "dev" or "test" split

    Returns:
        {
            "count": N,
            "recall_at_3": 0.X,
            "recall_at_5": 0.X,
            "recall_at_10": 0.X,
            "ndcg_at_5": 0.X,
            "mrr": 0.X,
            "mean_rank": X.X,
        }
    """
    if split:
        results = [r for r in results if r.split == split]

    if not results:
        return {"count": 0}

    n = len(results)

    # Recall@K from stored hit status
    r_at_3 = sum(1 for r in results
                 if recall_at_k(r.retrieved_ids, set(r.evidence_ids), 3) > 0) / n
    r_at_5 = sum(1 for r in results
                 if recall_at_k(r.retrieved_ids, set(r.evidence_ids), 5) > 0) / n
    r_at_10 = sum(1 for r in results
                  if recall_at_k(r.retrieved_ids, set(r.evidence_ids), 10) > 0) / n

    # NDCG@5
    ndcg_5 = sum(ndcg_at_k(r.retrieved_ids, set(r.evidence_ids), 5) for r in results) / n

    # MRR
    mrr_score = sum(mean_reciprocal_rank(r.retrieved_ids, set(r.evidence_ids))
                    for r in results) / n

    # Mean rank of first hit (only for questions that had a hit)
    ranks = [r.rank_of_first_hit for r in results if r.rank_of_first_hit is not None]
    mean_rank = sum(ranks) / len(ranks) if ranks else float("inf")

    return {
        "count": n,
        "recall_at_3": r_at_3,
        "recall_at_5": r_at_5,
        "recall_at_10": r_at_10,
        "ndcg_at_5": ndcg_5,
        "mrr": mrr_score,
        "mean_rank": mean_rank,
    }


def per_category_breakdown(
    results: List[RetrievalResult],
    split: Optional[str] = None,
    k: int = 5,
) -> Dict[str, Dict]:
    """
    Break down retrieval results by question category.

    Returns:
        {
            "single_session": {
                "count": N,
                "recall_at_k": 0.X,
                "failed_ids": ["q01", "q03"],
            },
            ...
        }
    """
    if split:
        results = [r for r in results if r.split == split]

    by_cat: Dict[str, List[RetrievalResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    breakdown = {}
    for cat, cat_results in sorted(by_cat.items()):
        n = len(cat_results)
        hits = sum(1 for r in cat_results
                   if recall_at_k(r.retrieved_ids, set(r.evidence_ids), k) > 0)
        failed = [r.question_id for r in cat_results
                  if recall_at_k(r.retrieved_ids, set(r.evidence_ids), k) == 0]

        breakdown[cat] = {
            "count": n,
            "recall_at_k": hits / n if n > 0 else 0.0,
            "failed_ids": failed,
        }

    return breakdown


# ===========================================================================
# Formatting
# ===========================================================================


def format_retrieval_table(
    mode_results: Dict[str, Dict[str, float]],
) -> str:
    """
    Format retrieval results as a Markdown table.

    Args:
        mode_results: {mode_name: aggregate_dict}

    Returns:
        Markdown table string
    """
    lines = []
    lines.append("| Mode | R@3 | R@5 | R@10 | NDCG@5 | MRR |")
    lines.append("|---|---|---|---|---|---|")

    for mode, agg in mode_results.items():
        r3 = f"{agg.get('recall_at_3', 0) * 100:.1f}%"
        r5 = f"{agg.get('recall_at_5', 0) * 100:.1f}%"
        r10 = f"{agg.get('recall_at_10', 0) * 100:.1f}%"
        ndcg = f"{agg.get('ndcg_at_5', 0):.3f}"
        mrr = f"{agg.get('mrr', 0):.3f}"
        lines.append(f"| {mode} | {r3} | {r5} | {r10} | {ndcg} | {mrr} |")

    return "\n".join(lines)


def format_category_table(
    breakdown: Dict[str, Dict],
    k: int = 5,
) -> str:
    """
    Format per-category breakdown as a Markdown table.

    Args:
        breakdown: Output of per_category_breakdown()
        k: The K value used

    Returns:
        Markdown table string
    """
    lines = []
    lines.append(f"| Category | Count | R@{k} | Failed |")
    lines.append("|---|---|---|---|")

    for cat, data in breakdown.items():
        count = data["count"]
        recall = f"{data['recall_at_k'] * 100:.1f}%"
        failed = ", ".join(data["failed_ids"][:5])  # Show up to 5
        if len(data["failed_ids"]) > 5:
            failed += f" (+{len(data['failed_ids']) - 5})"
        if not failed:
            failed = "--"
        lines.append(f"| {cat} | {count} | {recall} | {failed} |")

    return "\n".join(lines)


def format_classification_table(
    component_metrics: Dict[str, ClassificationMetrics],
) -> str:
    """
    Format classification results (quality gate, dedup, etc.) as a Markdown table.

    Args:
        component_metrics: {component_name: ClassificationMetrics}

    Returns:
        Markdown table string
    """
    lines = []
    lines.append("| Component | Precision | Recall | F1 | Accuracy |")
    lines.append("|---|---|---|---|---|")

    for name, m in component_metrics.items():
        lines.append(
            f"| {name} | {m.precision * 100:.1f}% | {m.recall * 100:.1f}% "
            f"| {m.f1 * 100:.1f}% | {m.accuracy * 100:.1f}% |"
        )

    return "\n".join(lines)


# ===========================================================================
# Result Persistence (JSONL)
# ===========================================================================


def save_results_jsonl(results: List[RetrievalResult], filepath: Path) -> None:
    """
    Save retrieval results to a JSONL file.

    Each line is a JSON object with all result fields.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in results:
            obj = {
                "question_id": r.question_id,
                "question": r.question,
                "category": r.category,
                "split": r.split,
                "mode": r.mode,
                "top_k": r.top_k,
                "retrieved_ids": r.retrieved_ids,
                "evidence_ids": r.evidence_ids,
                "hit": r.hit,
                "rank_of_first_hit": r.rank_of_first_hit,
                "similarity_scores": r.similarity_scores,
                "ndcg": r.ndcg,
                "mrr": r.mrr,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_results_jsonl(filepath: Path) -> List[RetrievalResult]:
    """Load retrieval results from a JSONL file."""
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            results.append(RetrievalResult(
                question_id=obj["question_id"],
                question=obj["question"],
                category=obj["category"],
                split=obj["split"],
                mode=obj["mode"],
                top_k=obj["top_k"],
                retrieved_ids=obj["retrieved_ids"],
                evidence_ids=obj["evidence_ids"],
                hit=obj["hit"],
                rank_of_first_hit=obj.get("rank_of_first_hit"),
                similarity_scores=obj.get("similarity_scores", []),
                ndcg=obj.get("ndcg", 0.0),
                mrr=obj.get("mrr", 0.0),
            ))
    return results


def generate_run_id() -> str:
    """Generate a unique run ID based on current timestamp."""
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")
