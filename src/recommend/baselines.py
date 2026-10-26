"""Phase 2: Baselines — popularity, random, user-based collaborative filtering."""

from __future__ import annotations

import json
import math
import os
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    INTERACTIONS_PATH,
    SPLITS_PATH,
    USERS_PATH,
    CATALOG_PATH,
    RECOMMENDATION_N,
    TRAIN_RATIO,
)
from src.data.loader import load_catalog, load_users, load_interactions, load_splits

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class BaseRecommender:
    name = "base"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.all_course_ids: list[str] = []
        self.completed_by_user: dict[str, set[str]] = {}
        self.train_user_ids: set[str] = set()
        self.fitted: bool = False

    def fit(self, interactions_df: pd.DataFrame, train_user_ids: set[str], catalog: list[dict]) -> None:
        self.train_user_ids = set(train_user_ids)
        self.all_course_ids = [c["id"] for c in catalog]
        self.completed_by_user = {}
        for uid in train_user_ids:
            self.completed_by_user[uid] = set()
        if interactions_df is not None and not interactions_df.empty:
            train_df = interactions_df[interactions_df["user_id"].isin(train_user_ids)]
            completed = train_df[train_df["action"] == "completed"]
            for uid, group in completed.groupby("user_id"):
                self.completed_by_user[uid] = set(group["course_id"].tolist())
        self.fitted = True

    def _exclude_completed(self, candidates: list[str], user_id: str, n: int) -> list[str]:
        completed = self.completed_by_user.get(user_id, set())
        filtered = [c for c in candidates if c not in completed]
        return filtered[:n]

    def recommend(self, user_id: str, n: int = RECOMMENDATION_N) -> list[str]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.__dict__.items() if k != "fitted"}
        state["fitted"] = self.fitted
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)
        self.fitted = state.get("fitted", False)


class RandomRecommender(BaseRecommender):
    name = "random"

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.rng: random.Random | None = None

    def fit(self, interactions_df, train_user_ids, catalog):
        super().fit(interactions_df, train_user_ids, catalog)
        self.rng = random.Random(self.seed)

    def recommend(self, user_id: str, n: int = RECOMMENDATION_N) -> list[str]:
        if not self.fitted:
            raise RuntimeError("Recommender not fitted. Call fit() first.")
        available = [c for c in self.all_course_ids if c not in self.completed_by_user.get(user_id, set())]
        self.rng.shuffle(available)
        return available[:n]


class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def __init__(self):
        super().__init__()
        self.popularity: Counter = Counter()

    def fit(self, interactions_df, train_user_ids, catalog):
        super().fit(interactions_df, train_user_ids, catalog)
        self.popularity = Counter()
        if interactions_df is not None and not interactions_df.empty:
            train_df = interactions_df[interactions_df["user_id"].isin(train_user_ids)]
            completed = train_df[train_df["action"] == "completed"]
            self.popularity = Counter(completed["course_id"].tolist())

    def recommend(self, user_id: str, n: int = RECOMMENDATION_N) -> list[str]:
        if not self.fitted:
            raise RuntimeError("Recommender not fitted. Call fit() first.")
        ranked = [cid for cid, _ in self.popularity.most_common()]
        ranked += [cid for cid in self.all_course_ids if cid not in self.popularity]
        return self._exclude_completed(ranked, user_id, n)


class UserCFRecommender(BaseRecommender):
    name = "user_cf"

    def __init__(self, k_neighbors: int = 20, seed: int = 42):
        super().__init__(seed)
        self.k_neighbors = k_neighbors
        self.user_index: dict[str, int] = {}
        self.course_index: dict[str, int] = {}
        self.interaction_matrix: np.ndarray | None = None
        self.similarity_matrix: np.ndarray | None = None
        self.popularity_fallback: PopularityRecommender | None = None

    def fit(self, interactions_df, train_user_ids, catalog):
        super().fit(interactions_df, train_user_ids, catalog)

        self.popularity_fallback = PopularityRecommender()
        self.popularity_fallback.fit(interactions_df, train_user_ids, catalog)

        users = sorted(train_user_ids)
        self.user_index = {uid: i for i, uid in enumerate(users)}
        self.course_index = {cid: i for i, cid in enumerate(self.all_course_ids)}

        n_users = len(users)
        n_courses = len(self.all_course_ids)
        matrix = np.zeros((n_users, n_courses), dtype=np.float64)

        if interactions_df is not None and not interactions_df.empty:
            train_df = interactions_df[interactions_df["user_id"].isin(train_user_ids)]
            for _, row in train_df.iterrows():
                uid = row["user_id"]
                cid = row["course_id"]
                action = row["action"]
                rating = row["rating"]
                if uid not in self.user_index or cid not in self.course_index:
                    continue
                if action == "rated" and str(rating).strip() != "":
                    value = float(rating)
                else:
                    value = 1.0
                matrix[self.user_index[uid], self.course_index[cid]] = value

        self.interaction_matrix = matrix

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms
        sim = normalized @ normalized.T
        np.fill_diagonal(sim, -1.0)
        self.similarity_matrix = sim

        self.fitted = True

    def recommend(self, user_id: str, n: int = RECOMMENDATION_N) -> list[str]:
        if not self.fitted:
            raise RuntimeError("Recommender not fitted. Call fit() first.")

        if user_id not in self.user_index:
            return self.popularity_fallback.recommend(user_id, n)

        u_idx = self.user_index[user_id]
        sim_row = self.similarity_matrix[u_idx]
        top_k_idx = np.argsort(sim_row)[::-1][: self.k_neighbors]
        top_k_sim = sim_row[top_k_idx]

        scores: dict[str, float] = defaultdict(float)
        for sim_idx, sim_val in zip(top_k_idx, top_k_sim):
            if sim_val <= 0:
                continue
            sim_uid = [uid for uid, idx in self.user_index.items() if idx == sim_idx][0]
            for cid in self.completed_by_user.get(sim_uid, set()):
                scores[cid] += sim_val

        ranked = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        if not ranked:
            return self.popularity_fallback.recommend(user_id, n)

        return self._exclude_completed(ranked, user_id, n)


def _ndcg_at_k(recommended: list[str], ground_truth: set[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    dcg = 0.0
    for i, cid in enumerate(recommended[:k]):
        rel = 1.0 if cid in ground_truth else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth), k)))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_baselines() -> dict:
    catalog = load_catalog(CATALOG_PATH)
    users = load_users(USERS_PATH)
    interactions = load_interactions(INTERACTIONS_PATH)
    splits = load_splits(SPLITS_PATH)

    train_ids = set(splits["train_interactions"])
    test_ids = set(splits["test_holdout"])
    challenge_users = set(splits["challenge_users"])

    df = pd.DataFrame(interactions)
    df["id"] = df["id"].astype(str)

    # Derive user sets from interaction splits
    train_df = df[df["id"].isin(train_ids)]
    test_df = df[df["id"].isin(test_ids)]
    train_users = set(train_df["user_id"].tolist())
    test_users = set(test_df["user_id"].tolist())

    recs = {
        "random": RandomRecommender(),
        "popularity": PopularityRecommender(),
        "user_cf": UserCFRecommender(),
    }

    for name, rec in recs.items():
        rec.fit(train_df, train_users, catalog)
        models_dir = MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)
        rec.save(models_dir / f"{name}.pkl")

    ground_truth: dict[str, set[str]] = {}
    # Ground truth includes completed + rated>=4; rated>=4 treated as strong implicit positive signal.
    completed_test = test_df[test_df["action"] == "completed"]
    rated_test = test_df[test_df["action"] == "rated"]
    highly_rated_test = rated_test[rated_test["rating"].astype(float) >= 4.0]
    positive_test = pd.concat([completed_test, highly_rated_test])
    for uid, group in positive_test.groupby("user_id"):
        ground_truth[uid] = set(group["course_id"].tolist())

    results: dict[str, dict] = {}
    for name, rec in recs.items():
        precisions = []
        recalls = []
        ndcgs = []
        for uid in test_users:
            recommended = rec.recommend(uid, RECOMMENDATION_N)
            gt = ground_truth.get(uid, set())
            if not gt:
                continue
            hit = len(set(recommended) & gt)
            precisions.append(hit / len(recommended) if recommended else 0.0)
            recalls.append(hit / len(gt))
            ndcgs.append(_ndcg_at_k(recommended, gt, RECOMMENDATION_N))
        results[name] = {
            "precision_at_10": round(np.mean(precisions), 4) if precisions else 0.0,
            "recall_at_10": round(np.mean(recalls), 4) if recalls else 0.0,
            "ndcg_at_10": round(np.mean(ndcgs), 4) if ndcgs else 0.0,
            "num_test_users_evaluated": len(precisions),
        }

    return results


def write_report(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = __import__("datetime").datetime.now().strftime("%Y%m%d")
    eval_dir = out_dir / f"eval_{date_str}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Baselines Evaluation", "", f"Date: {date_str}", "", "| Recommender | Precision@10 | Recall@10 | NDCG@10 | Users |", "|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['precision_at_10']} | {m['recall_at_10']} | {m['ndcg_at_10']} | {m['num_test_users_evaluated']} |")
    lines.extend(["", "Ground truth: completed courses + courses rated >= 4 by test users in held-out interactions.", "Fit only on train users and train interactions."])

    md_path = eval_dir / "baselines.md"
    json_path = eval_dir / "baselines.json"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Report written: {md_path}")
    print(f"Report written: {json_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    results = evaluate_baselines()
    write_report(results, Path(__file__).resolve().parents[2] / "reports")
