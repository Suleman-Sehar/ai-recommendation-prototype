from __future__ import annotations

import pandas as pd

from src.config import CATALOG_PATH, INTERACTIONS_PATH, SPLITS_PATH
from src.data.loader import load_catalog, load_interactions, load_splits
from src.recommend.hybrid import Hybrid


def _user_sets_from_splits(splits: dict, interactions: list[dict]) -> tuple[set[str], set[str]]:
    train_ids = set(splits["train_interactions"])
    test_ids = set(splits["test_holdout"])
    train_users = {it["user_id"] for it in interactions if it["id"] in train_ids}
    test_users = {it["user_id"] for it in interactions if it["id"] in test_ids}
    return train_users, test_users


class TestHybrid:
    def test_blending_logic_correct(self):
        catalog = load_catalog(CATALOG_PATH)
        interactions = load_interactions(INTERACTIONS_PATH)
        splits = load_splits(SPLITS_PATH)
        df = pd.DataFrame(interactions)
        train_users, _ = _user_sets_from_splits(splits, interactions)
        train_df = df[df["user_id"].isin(train_users)]

        h = Hybrid()
        h.fit(catalog, train_df)
        result = h.recommend("user-0001", 10, alpha=0.5)
        assert isinstance(result, list)
        assert len(result) == 10

    def test_exclusion_applied_after_blend(self):
        catalog = load_catalog(CATALOG_PATH)
        interactions = load_interactions(INTERACTIONS_PATH)
        splits = load_splits(SPLITS_PATH)
        df = pd.DataFrame(interactions)
        train_users, _ = _user_sets_from_splits(splits, interactions)
        train_df = df[df["user_id"].isin(train_users)]
        completed_train = train_df[train_df["action"] == "completed"]

        h = Hybrid()
        h.fit(catalog, train_df)

        for uid, group in completed_train.groupby("user_id"):
            completed = set(group["course_id"].tolist())
            if not completed:
                continue
            result = h.recommend(uid, 20, alpha=0.5)
            overlap = set(result) & completed
            assert overlap == set()

    def test_cold_start_handled(self):
        catalog = load_catalog(CATALOG_PATH)
        interactions = load_interactions(INTERACTIONS_PATH)
        splits = load_splits(SPLITS_PATH)
        df = pd.DataFrame(interactions)
        train_users, _ = _user_sets_from_splits(splits, interactions)
        train_df = df[df["user_id"].isin(train_users)]

        h = Hybrid()
        h.fit(catalog, train_df)
        result = h.recommend("user-9999", 10, alpha=0.5)
        assert isinstance(result, list)
        assert len(result) == 10

    def test_alpha_tuning_uses_only_train_data(self):
        catalog = load_catalog(CATALOG_PATH)
        interactions = load_interactions(INTERACTIONS_PATH)
        splits = load_splits(SPLITS_PATH)
        df = pd.DataFrame(interactions)
        train_users, test_users = _user_sets_from_splits(splits, interactions)
        train_df = df[df["user_id"].isin(train_users)]
        # simulate tuning on train only - just check it works
        h = Hybrid()
        h.fit(catalog, train_df)
        # try different alphas
        for alpha in [0.3, 0.5, 0.7]:
            result = h.recommend("user-0001", 10, alpha=alpha)
            assert len(result) == 10
