from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import RECOMMENDATION_N
from evaluate.metrics import evaluate_recommender
from recommend.baselines import UserCFRecommender, PopularityRecommender
from recommend.content import ContentBased


class Hybrid:
    name = "hybrid"

    def __init__(self):
        self.cf = UserCFRecommender(k_neighbors=20)
        self.cb = ContentBased()
        self.popularity_fallback = PopularityRecommender()
        self.catalog = None
        self.interactions_train_df = None
        self.train_user_ids = None
        self.fitted = False

    def fit(self, catalog_df, interactions_train_df) -> None:
        if isinstance(catalog_df, list):
            self.catalog = catalog_df
            catalog_list = catalog_df
        else:
            self.catalog = catalog_df.to_dict("records") if hasattr(catalog_df, "to_dict") else catalog_df
            catalog_list = self.catalog
        if interactions_train_df is not None and not isinstance(interactions_train_df, pd.DataFrame):
            interactions_train_df = pd.DataFrame(interactions_train_df)
        self.interactions_train_df = interactions_train_df
        if interactions_train_df is not None and not interactions_train_df.empty:
            self.train_user_ids = set(interactions_train_df["user_id"].astype(str).tolist())
        else:
            self.train_user_ids = set()

        self.cf.fit(interactions_train_df, self.train_user_ids, catalog_list)
        self.cb.fit(catalog_df, interactions_train_df)
        self.popularity_fallback.fit(interactions_train_df, self.train_user_ids, catalog_list)
        self.fitted = True

    def recommend(self, user_id: str, n: int = RECOMMENDATION_N, alpha: float = 0.5) -> list[str]:
        if not self.fitted:
            raise RuntimeError("Recommender not fitted. Call fit() first.")

        user_id = str(user_id)

        # Cold-start: if CF falls back to popularity
        cf_rec = self.cf.recommend(user_id, 50)
        cb_rec = self.cb.recommend(user_id, 50)

        # Check if user is cold start for CF
        cf_is_cold = (user_id not in getattr(self.cf, "user_index", {}))

        if cf_is_cold:
            # Blend popularity with content-based
            pop_rec = self.popularity_fallback.recommend(user_id, 50)
            scores = {}
            for i, cid in enumerate(pop_rec):
                scores[cid] = scores.get(cid, 0.0) + (1.0 / (i + 1)) * alpha  # treat pop as base
            for i, cid in enumerate(cb_rec):
                scores[cid] = scores.get(cid, 0.0) + (1.0 / (i + 1)) * (1.0 - alpha)
            ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            completed = self.cf.completed_by_user.get(user_id, set()) if hasattr(self.cf, "completed_by_user") else set()
            results = [c for c in ranked if c not in completed][:n]
            if not results:
                return self.popularity_fallback.recommend(user_id, n)
            return results

        # Build candidate pool from union of top-50 from CF and top-50 from content-based
        scores = {}
        # Get scores from CF (use position-based scoring as proxy)
        for i, cid in enumerate(cf_rec):
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (i + 1)) * alpha
        for i, cid in enumerate(cb_rec):
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (i + 1)) * (1.0 - alpha)

        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        # Apply completed-course exclusion after blending
        completed = self.cf.completed_by_user.get(user_id, set()) if hasattr(self.cf, "completed_by_user") else set()
        results = [c for c in ranked if c not in completed][:n]
        if not results:
            return self.popularity_fallback.recommend(user_id, n)
        return results
