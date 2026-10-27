from __future__ import annotations

import pandas as pd

from src.config import CATALOG_PATH, INTERACTIONS_PATH, SPLITS_PATH
from src.data.loader import load_catalog, load_interactions, load_splits
from src.recommend.content import ContentBased
from src.recommend.baselines import PopularityRecommender


def _user_sets_from_splits(splits: dict, interactions: list[dict]) -> tuple[set[str], set[str]]:
    train_ids = set(splits["train_interactions"])
    test_ids = set(splits["test_holdout"])
    train_users = {it["user_id"] for it in interactions if it["id"] in train_ids}
    test_users = {it["user_id"] for it in interactions if it["id"] in test_ids}
    return train_users, test_users


class TestContentBased:
    def test_tfidf_matrix_shape_correct(self):
        catalog = load_catalog(CATALOG_PATH)
        interactions = load_interactions(INTERACTIONS_PATH)
        splits = load_splits(SPLITS_PATH)
        df = pd.DataFrame(interactions)
        train_users, _ = _user_sets_from_splits(splits, interactions)
        train_df = df[df["user_id"].isin(train_users)]

        cb = ContentBased()
        cb.fit(catalog, train_df)
        assert cb.course_vectors.shape[0] == len(catalog)
        assert cb.course_vectors.shape[1] > 0

    def test_user_profile_weighted_correctly(self):
        catalog = load_catalog(CATALOG_PATH)
        interactions = load_interactions(INTERACTIONS_PATH)
        splits = load_splits(SPLITS_PATH)
        df = pd.DataFrame(interactions)
        train_users, _ = _user_sets_from_splits(splits, interactions)
        train_df = df[df["user_id"].isin(train_users)]

        cb = ContentBased()
        cb.fit(catalog, train_df)
        assert cb.SIGNAL_WEIGHTS["completed"] == 2.0
        assert cb.SIGNAL_WEIGHTS["rated_high"] == 1.5
        assert cb.SIGNAL_WEIGHTS["rated_low"] == 0.5
        assert cb.SIGNAL_WEIGHTS["viewed"] == 0.1

    def test_completed_courses_excluded(self):
        catalog = load_catalog(CATALOG_PATH)
        interactions = load_interactions(INTERACTIONS_PATH)
        splits = load_splits(SPLITS_PATH)
        df = pd.DataFrame(interactions)
        train_users, _ = _user_sets_from_splits(splits, interactions)
        train_df = df[df["user_id"].isin(train_users)]
        completed_train = train_df[train_df["action"] == "completed"]

        cb = ContentBased()
        cb.fit(catalog, train_df)

        for uid, group in completed_train.groupby("user_id"):
            completed = set(group["course_id"].tolist())
            if not completed:
                continue
            result = cb.recommend(uid, 20)
            overlap = set(result) & completed
            assert overlap == set()

    def test_cold_start_falls_back_to_popularity(self):
        catalog = load_catalog(CATALOG_PATH)
        interactions = load_interactions(INTERACTIONS_PATH)
        splits = load_splits(SPLITS_PATH)
        df = pd.DataFrame(interactions)
        train_users, _ = _user_sets_from_splits(splits, interactions)
        train_df = df[df["user_id"].isin(train_users)]

        cb = ContentBased()
        cb.fit(catalog, train_df)
        pop = PopularityRecommender()
        pop.fit(train_df, train_users, catalog)

        result_cb = cb.recommend("user-9999", 10)
        result_pop = pop.recommend("user-9999", 10)
        assert result_cb == result_pop

    def test_returns_exactly_n_items(self):
        catalog = load_catalog(CATALOG_PATH)
        interactions = load_interactions(INTERACTIONS_PATH)
        splits = load_splits(SPLITS_PATH)
        df = pd.DataFrame(interactions)
        train_users, _ = _user_sets_from_splits(splits, interactions)
        train_df = df[df["user_id"].isin(train_users)]

        cb = ContentBased()
        cb.fit(catalog, train_df)

        for uid in ["user-0001", "user-0050", "user-0100"]:
            result = cb.recommend(uid, 10)
            assert isinstance(result, list)
            assert len(result) == 10
            assert all(isinstance(c, str) for c in result)

    def test_cosine_similarity_range_0_1(self):
        from sklearn.metrics.pairwise import cosine_similarity

        catalog = load_catalog(CATALOG_PATH)
        interactions = load_interactions(INTERACTIONS_PATH)
        splits = load_splits(SPLITS_PATH)
        df = pd.DataFrame(interactions)
        train_users, _ = _user_sets_from_splits(splits, interactions)
        train_df = df[df["user_id"].isin(train_users)]

        cb = ContentBased()
        cb.fit(catalog, train_df)
        sim = cosine_similarity(cb.course_vectors[0:1], cb.course_vectors[0:10]).flatten()
        assert all(-1e-10 <= s <= 1.0 + 1e-10 for s in sim)

    def test_profile_closer_to_completed_than_viewed(self):
        from src.config import CATALOG_PATH, INTERACTIONS_PATH, SPLITS_PATH
        from src.data.loader import load_catalog, load_interactions, load_splits
        from src.recommend.content import ContentBased
        from sklearn.preprocessing import normalize
        from sklearn.metrics.pairwise import cosine_similarity
        import pandas as pd

        catalog = load_catalog(CATALOG_PATH)
        interactions = load_interactions(INTERACTIONS_PATH)
        splits = load_splits(SPLITS_PATH)
        df = pd.DataFrame(interactions)
        df['id'] = df['id'].astype(str)
        train_ids = set(splits['train_interactions'])
        train_df = df[df['id'].isin(train_ids)]

        cb = ContentBased()
        cb.SIGNAL_WEIGHTS = {'completed': 2.0, 'rated_high': 1.5, 'rated_low': 0.5, 'viewed': 0.1}
        cb.fit(catalog, train_df)

        found = False
        for uid, interactions_u in cb.user_interactions.items():
            acts = train_df[train_df['user_id'] == uid]
            comp = acts[acts['action'] == 'completed']
            view = acts[acts['action'] == 'viewed']
            if len(comp) == 1 and len(view) >= 3:
                profile = None
                for cid, weight in interactions_u:
                    if cid not in cb.course_index:
                        continue
                    idx = cb.course_index[cid]
                    vec = cb.course_vectors[idx]
                    if profile is None:
                        profile = vec * weight
                    else:
                        profile = profile + vec * weight
                profile_n = normalize(profile, norm='l2')
                comp_cid = comp['course_id'].iloc[0]
                comp_sim = cosine_similarity(profile_n, cb.course_vectors[cb.course_index[comp_cid]:cb.course_index[comp_cid]+1])[0][0]
                min_view_sim = float('inf')
                for _, vr in view.iterrows():
                    vid = vr['course_id']
                    if vid not in cb.course_index:
                        continue
                    vsim = cosine_similarity(profile_n, cb.course_vectors[cb.course_index[vid]:cb.course_index[vid]+1])[0][0]
                    if vsim < min_view_sim:
                        min_view_sim = vsim
                assert comp_sim > min_view_sim
                found = True
                break
        assert found
