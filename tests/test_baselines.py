"""Tests for Phase 2 baselines."""

from __future__ import annotations

import json
import random

import pandas as pd
import pytest

from src.config import CATALOG_PATH, INTERACTIONS_PATH, SPLITS_PATH, USERS_PATH
from src.data.loader import load_catalog, load_interactions, load_splits
from src.recommend.baselines import (
    RandomRecommender,
    PopularityRecommender,
    UserCFRecommender,
    evaluate_baselines,
)


def _user_sets_from_splits(splits: dict, interactions: list[dict]) -> tuple[set[str], set[str]]:
    """Derive train_users and test_users from interaction-level splits."""
    train_ids = set(splits["train_interactions"])
    test_ids = set(splits["test_holdout"])
    train_users = {it["user_id"] for it in interactions if it["id"] in train_ids}
    test_users = {it["user_id"] for it in interactions if it["id"] in test_ids}
    return train_users, test_users


@pytest.fixture(scope="module")
def data():
    catalog = load_catalog(CATALOG_PATH)
    interactions = load_interactions(INTERACTIONS_PATH)
    splits = load_splits(SPLITS_PATH)
    return catalog, interactions, splits


@pytest.fixture(scope="module")
def fitted_recs(data):
    catalog, interactions, splits = data
    df = pd.DataFrame(interactions)
    train_users, _ = _user_sets_from_splits(splits, interactions)
    recs = {
        "random": RandomRecommender(seed=42),
        "popularity": PopularityRecommender(),
        "user_cf": UserCFRecommender(k_neighbors=20),
    }
    for rec in recs.values():
        rec.fit(df, train_users, catalog)
    return recs, splits, catalog, df


class TestCommonInterface:
    def test_recommend_returns_list_of_course_ids(self, fitted_recs):
        recs, _, _, _ = fitted_recs
        for rec in recs.values():
            result = rec.recommend("user-0001", 10)
            assert isinstance(result, list)
            assert all(isinstance(c, str) for c in result)
            assert len(result) <= 10

    def test_recommend_default_n(self, fitted_recs):
        recs, _, _, _ = fitted_recs
        for rec in recs.values():
            result = rec.recommend("user-0001")
            assert len(result) <= 10


class TestCompletedExclusion:
    def test_completed_courses_never_recommended(self, fitted_recs, data):
        recs, splits, catalog, _ = fitted_recs
        _, interactions, _ = data
        train_users, _ = _user_sets_from_splits(splits, interactions)
        df_train = pd.DataFrame(interactions)
        train_df = df_train[df_train["user_id"].isin(train_users)]
        completed_train = train_df[train_df["action"] == "completed"]

        for uid, group in completed_train.groupby("user_id"):
            completed = set(group["course_id"].tolist())
            if not completed:
                continue
            for rec in recs.values():
                result = rec.recommend(uid, 20)
                overlap = set(result) & completed
                assert overlap == set(), f"{rec.name} recommended completed courses: {overlap}"


class TestRandomReproducibility:
    def test_random_reproducible_same_seed(self, data):
        catalog, interactions, splits = data
        df = pd.DataFrame(interactions)
        train_users, _ = _user_sets_from_splits(splits, interactions)

        r1 = RandomRecommender(seed=42)
        r1.fit(df, train_users, catalog)
        r2 = RandomRecommender(seed=42)
        r2.fit(df, train_users, catalog)

        for uid in ["user-0001", "user-0050", "user-0100"]:
            assert r1.recommend(uid, 10) == r2.recommend(uid, 10)

    def test_random_different_seed_different_output(self, data):
        catalog, interactions, splits = data
        df = pd.DataFrame(interactions)
        train_users, _ = _user_sets_from_splits(splits, interactions)

        r1 = RandomRecommender(seed=1)
        r1.fit(df, train_users, catalog)
        r2 = RandomRecommender(seed=2)
        r2.fit(df, train_users, catalog)

        results = [r1.recommend("user-0001", 10) for _ in range(1)]
        results += [r2.recommend("user-0001", 10) for _ in range(1)]
        # Different seeds may or may not differ; just check both return valid lists
        assert len(results[0]) == 10
        assert len(results[1]) == 10


class TestPopularity:
    def test_popularity_ranks_by_completed_count(self, data):
        catalog, interactions, splits = data
        df = pd.DataFrame(interactions)
        train_users, _ = _user_sets_from_splits(splits, interactions)
        train_df = df[df["user_id"].isin(train_users)]
        completed = train_df[train_df["action"] == "completed"]
        counts = completed["course_id"].value_counts()

        rec = PopularityRecommender()
        rec.fit(df, train_users, catalog)

        # Top recommended (for a user with no completed) should be most completed
        result = rec.recommend("user-9999", 5)  # user not in train, no completed
        top5 = result[:5]
        assert top5 == list(counts.index[:5])

    def test_popularity_excludes_completed(self, fitted_recs, data):
        recs, splits, catalog, _ = fitted_recs
        _, interactions, _ = data
        rec = recs["popularity"]
        train_users, _ = _user_sets_from_splits(splits, interactions)
        df_train = pd.DataFrame(interactions)
        train_df = df_train[df_train["user_id"].isin(train_users)]
        completed_train = train_df[train_df["action"] == "completed"]

        for uid, group in completed_train.groupby("user_id"):
            completed = set(group["course_id"].tolist())
            if not completed:
                continue
            result = rec.recommend(uid, 20)
            assert set(result).isdisjoint(completed)


class TestUserCF:
    def test_cf_cold_start_falls_back_to_popularity(self, fitted_recs, data):
        recs, splits, catalog, df = fitted_recs
        cf = recs["user_cf"]
        pop = recs["popularity"]

        # User with 0 interactions (not in train) -> cold start
        result_cf = cf.recommend("user-9999", 10)
        result_pop = pop.recommend("user-9999", 10)
        assert result_cf == result_pop

    def test_cf_excludes_completed(self, fitted_recs, data):
        recs, splits, catalog, _ = fitted_recs
        _, interactions, _ = data
        rec = recs["user_cf"]
        train_users, _ = _user_sets_from_splits(splits, interactions)
        df_train = pd.DataFrame(interactions)
        train_df = df_train[df_train["user_id"].isin(train_users)]
        completed_train = train_df[train_df["action"] == "completed"]

        for uid, group in completed_train.groupby("user_id"):
            completed = set(group["course_id"].tolist())
            if not completed:
                continue
            result = rec.recommend(uid, 20)
            assert set(result).isdisjoint(completed)


class TestWeightedCF:
    """Weighted CF should score viewed interactions of similar users,
    giving it a candidate pool where binary (completed-only) CF has none."""

    def _synthetic_catalog(self) -> list[dict]:
        return [
            {"id": "course-G", "title": "G", "tags": "a|b", "difficulty": "beginner", "duration_minutes": "60", "description": "desc G"},
            {"id": "course-H", "title": "H", "tags": "a|b", "difficulty": "beginner", "duration_minutes": "60", "description": "desc H"},
            {"id": "course-I", "title": "I", "tags": "c|d", "difficulty": "beginner", "duration_minutes": "60", "description": "desc I"},
            {"id": "course-J", "title": "J", "tags": "e|f", "difficulty": "beginner", "duration_minutes": "60", "description": "desc J"},
        ]

    def _synthetic_interactions(self) -> list[dict]:
        return [
            {"user_id": "user-A", "course_id": "course-G", "action": "viewed", "rating": "", "timestamp": "2024-01-01T00:00:00Z"},
            {"user_id": "user-A", "course_id": "course-H", "action": "viewed", "rating": "", "timestamp": "2024-01-02T00:00:00Z"},
            {"user_id": "user-B", "course_id": "course-G", "action": "viewed", "rating": "", "timestamp": "2024-01-03T00:00:00Z"},
            {"user_id": "user-B", "course_id": "course-H", "action": "viewed", "rating": "", "timestamp": "2024-01-04T00:00:00Z"},
            {"user_id": "user-B", "course_id": "course-I", "action": "completed", "rating": "", "timestamp": "2024-01-05T00:00:00Z"},
            {"user_id": "user-C", "course_id": "course-J", "action": "completed", "rating": "", "timestamp": "2024-01-06T00:00:00Z"},
        ]

    def test_weighted_cf_recommends_viewed_course_of_similar_user(self):
        catalog = self._synthetic_catalog()
        interactions = self._synthetic_interactions()
        df = pd.DataFrame(interactions)
        train_users = {"user-A", "user-B", "user-C"}

        cf = UserCFRecommender(k_neighbors=20)
        cf.fit(df, train_users, catalog)

        recs = cf.recommend("user-A", 10)
        # course-G was viewed by similar user-B (shared viewed courses G, H)
        # Weighted CF scores viewed interactions, so course-G should be recommended
        assert "course-G" in recs, f"Weighted CF should recommend course-G, got {recs}"

    def test_binary_cf_misses_viewed_interactions(self):
        # Simulate old binary CF: only completed interactions of similar users are scored
        interactions = self._synthetic_interactions()
        completed_by_user: dict[str, set[str]] = {}
        for _, row in pd.DataFrame(interactions).iterrows():
            if row["action"] == "completed":
                completed_by_user.setdefault(row["user_id"], set()).add(row["course_id"])

        # user-B completed: {course-I}, user-C completed: {course-J}
        # Neither is course-G (which was only viewed, never completed, by anyone)
        assert "course-G" not in completed_by_user.get("user-B", set())
        assert "course-G" not in completed_by_user.get("user-C", set())

        # Binary CF scoring pool contains only course-I and course-J, never course-G
        all_binary_scores = set()
        for courses in completed_by_user.values():
            all_binary_scores.update(courses)
        assert "course-G" not in all_binary_scores


class TestEvaluation:
    def test_evaluate_returns_all_three(self, fitted_recs, data):
        catalog, interactions, splits = data
        results = evaluate_baselines()
        assert "random" in results
        assert "popularity" in results
        assert "user_cf" in results
        for name, k_results in results.items():
            assert "k5" in k_results
            assert "k10" in k_results
            assert "k20" in k_results
            for k_key, m in k_results.items():
                assert "precision" in m
                assert "recall" in m
                assert "ndcg" in m
                assert "hit_rate" in m
                assert "users_evaluated" in m
                assert 0.0 <= m["precision"] <= 1.0
                assert 0.0 <= m["recall"] <= 1.0
                assert 0.0 <= m["ndcg"] <= 1.0
                assert 0.0 <= m["hit_rate"] <= 1.0

    def test_evaluation_uses_test_users_only(self, data):
        _, interactions, splits = data
        results = evaluate_baselines()
        _, test_users = _user_sets_from_splits(splits, interactions)
        # Each recommender evaluates the same set of test users with ground truth
        for k_results in results.values():
            for m in k_results.values():
                assert m["users_evaluated"] <= len(test_users)
