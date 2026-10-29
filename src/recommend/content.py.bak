"""Phase 3: Content-based recommender using TF-IDF on tags and description."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import RECOMMENDATION_N
from src.evaluate.metrics import evaluate_recommender
from src.recommend.baselines import PopularityRecommender


class ContentBased:
    name = "content_based"

    # Signal weights for user profile (same as CF for consistency)
    SIGNAL_WEIGHTS = {
        "completed": 1.0,
        "rated_high": 0.8,  # rated >= 4
        "rated_low": 0.3,   # rated 1-3
        "viewed": 0.1,
    }

    def __init__(self):
        self.all_course_ids: list[str] = []
        self.course_index: dict[str, int] = {}
        self.course_vectors = None  # sparse matrix
        self.completed_by_user: dict[str, set[str]] = {}
        self.train_user_ids: set[str] = set()
        self.popularity_fallback: PopularityRecommender | None = None
        self.fitted: bool = False
        self.catalog_df = None
        self.user_interactions = {}

    def _signal_weight(self, action: str, rating: str | None) -> float:
        if action == "completed":
            return self.SIGNAL_WEIGHTS["completed"]
        if action == "rated" and rating is not None and str(rating).strip() != "":
            try:
                r = float(rating)
            except ValueError:
                return self.SIGNAL_WEIGHTS["rated_low"]
            if r >= 4.0:
                return self.SIGNAL_WEIGHTS["rated_high"]
            return self.SIGNAL_WEIGHTS["rated_low"]
        if action == "viewed":
            return self.SIGNAL_WEIGHTS["viewed"]
        return self.SIGNAL_WEIGHTS["viewed"]

    def fit(self, catalog_df, interactions_train_df) -> None:
        # Convert catalog to DataFrame if it's a list
        if isinstance(catalog_df, list):
            catalog_df = pd.DataFrame(catalog_df)
        self.catalog_df = catalog_df.copy()
        self.all_course_ids = catalog_df["id"].astype(str).tolist()
        self.course_index = {cid: i for i, cid in enumerate(self.all_course_ids)}

        # Ensure interactions_train_df is DataFrame
        if interactions_train_df is not None and not isinstance(interactions_train_df, pd.DataFrame):
            interactions_train_df = pd.DataFrame(interactions_train_df)

        # Track completed courses per user for exclusion
        self.train_user_ids = set(interactions_train_df["user_id"].astype(str).tolist()) if interactions_train_df is not None and not interactions_train_df.empty else set()
        self.completed_by_user = {uid: set() for uid in self.train_user_ids}
        if interactions_train_df is not None and not interactions_train_df.empty:
            completed = interactions_train_df[interactions_train_df["action"] == "completed"]
            for uid, group in completed.groupby("user_id"):
                self.completed_by_user[str(uid)] = set(group["course_id"].astype(str).tolist())

        # Store user interactions for profile building
        self.user_interactions = {}
        if interactions_train_df is not None and not interactions_train_df.empty:
            for uid in self.train_user_ids:
                self.user_interactions[str(uid)] = []
            for _, row in interactions_train_df.iterrows():
                uid = str(row["user_id"])
                cid = str(row["course_id"])
                action = row["action"]
                rating = row["rating"]
                weight = self._signal_weight(action, rating)
                if uid in self.user_interactions:
                    self.user_interactions[uid].append((cid, weight))

        # Build TF-IDF on tags (pipe-separated, treat each tag as a word)
        tags_text = []
        for _, row in catalog_df.iterrows():
            tags = str(row.get("tags", ""))
            # Replace pipes with spaces so each tag becomes a separate token
            tags_text.append(tags.replace("|", " "))

        # Build TF-IDF on description
        desc_text = [str(row.get("description", "")) for _, row in catalog_df.iterrows()]

        # Fit TF-IDF vectorizers only on catalog
        tfidf_tags = TfidfVectorizer()
        tfidf_desc = TfidfVectorizer()

        tags_matrix = tfidf_tags.fit_transform(tags_text)
        desc_matrix = tfidf_desc.fit_transform(desc_text)

        # Concatenate with weights: tags 0.7, description 0.3
        tags_weighted = tags_matrix * 0.7
        desc_weighted = desc_matrix * 0.3
        self.course_vectors = sparse.hstack([tags_weighted, desc_weighted], format="csr")

        # Build popularity fallback for cold start
        self.popularity_fallback = PopularityRecommender()
        catalog_list = catalog_df.to_dict("records")
        self.popularity_fallback.fit(interactions_train_df, self.train_user_ids, catalog_list)

        self.fitted = True

    def recommend(self, user_id: str, n: int = RECOMMENDATION_N) -> list[str]:
        if not self.fitted:
            raise RuntimeError("Recommender not fitted. Call fit() first.")

        user_id = str(user_id)

        # Cold-start: if user has zero interactions in training data
        if user_id not in self.train_user_ids or not self.user_interactions.get(user_id):
            if self.popularity_fallback is not None:
                return self.popularity_fallback.recommend(user_id, n)
            return []

        # Get user's interactions to build profile
        user_interactions = self.user_interactions.get(user_id, [])
        if not user_interactions:
            if self.popularity_fallback is not None:
                return self.popularity_fallback.recommend(user_id, n)
            return []

        # Build weighted average of interacted course vectors
        profile = None
        total_weight = 0.0
        for cid, weight in user_interactions:
            if cid not in self.course_index:
                continue
            idx = self.course_index[cid]
            vec = self.course_vectors[idx]
            if profile is None:
                profile = vec * weight
            else:
                profile = profile + vec * weight
            total_weight += weight

        if profile is None or total_weight == 0:
            if self.popularity_fallback is not None:
                return self.popularity_fallback.recommend(user_id, n)
            return []

        profile = profile / total_weight

        # Compute cosine similarity between user profile and all course vectors
        similarities = cosine_similarity(profile, self.course_vectors).flatten()

        # Get top-N by similarity, excluding train-completed courses
        completed = self.completed_by_user.get(user_id, set())
        scored = [(i, similarities[i]) for i in range(len(self.all_course_ids))]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored:
            cid = self.all_course_ids[idx]
            if cid in completed:
                continue
            results.append(cid)
            if len(results) >= n:
                break

        if not results and self.popularity_fallback is not None:
            return self.popularity_fallback.recommend(user_id, n)
        return results
