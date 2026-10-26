"""Phase 6: Evaluation metrics — Precision@K, Recall@K, NDCG@K, Hit Rate@K."""

from __future__ import annotations

import math
from collections.abc import Iterable


def precision_at_k(recommended: Iterable[str], ground_truth: set[str], k: int) -> float:
    recommended = list(recommended)[:k]
    if not recommended:
        return 0.0
    hits = len(set(recommended) & ground_truth)
    return hits / len(recommended)


def recall_at_k(recommended: Iterable[str], ground_truth: set[str], k: int) -> float:
    recommended = list(recommended)[:k]
    if not ground_truth:
        return 0.0
    hits = len(set(recommended) & ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(recommended: Iterable[str], ground_truth: set[str], k: int) -> float:
    recommended = list(recommended)[:k]
    if not ground_truth:
        return 0.0
    dcg = 0.0
    for i, cid in enumerate(recommended):
        rel = 1.0 if cid in ground_truth else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth), k)))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(recommended: Iterable[str], ground_truth: set[str], k: int) -> float:
    """Fraction of users with at least one relevant item in top-K."""
    recommended = list(recommended)[:k]
    if not ground_truth:
        return 0.0
    return 1.0 if len(set(recommended) & ground_truth) > 0 else 0.0


EVAL_KS = [5, 10, 20]


def evaluate_recommender(
    rec,
    test_users: list[str],
    ground_truth: dict[str, set[str]],
    ks: list[int] | None = None,
) -> dict:
    """Compute P@K, R@K, NDCG@K, HitRate@K for a recommender across K values."""
    if ks is None:
        ks = EVAL_KS

    results: dict[str, dict] = {}
    for k in ks:
        precisions = []
        recalls = []
        ndcgs = []
        hit_rates = []
        evaluated = 0
        for uid in test_users:
            gt = ground_truth.get(uid, set())
            if not gt:
                continue
            recommended = rec.recommend(uid, k)
            evaluated += 1
            hits = len(set(recommended) & gt)
            precisions.append(hits / len(recommended) if recommended else 0.0)
            recalls.append(hits / len(gt))
            ndcgs.append(ndcg_at_k(recommended, gt, k))
            hit_rates.append(1.0 if hits > 0 else 0.0)

        results[f"k{k}"] = {
            "precision": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
            "recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            "ndcg": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
            "hit_rate": round(sum(hit_rates) / len(hit_rates), 4) if hit_rates else 0.0,
            "users_evaluated": evaluated,
        }
    return results
