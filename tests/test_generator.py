"""Tests for Phase 1 data generator."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.config import (
    ALL_TAGS,
    CATALOG_PATH,
    DATA_DIR,
    DIFFICULTIES,
    INTERACTIONS_PATH,
    NUM_COURSES,
    NUM_USERS,
    SEED,
    SPLITS_PATH,
    TAG_DOMAINS,
    TRAIN_RATIO,
    USERS_PATH,
    USER_TIERS,
    CHALLENGE_USER_COUNT,
)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class TestCatalog:
    def test_catalog_exists(self):
        assert CATALOG_PATH.exists()

    def test_catalog_row_count(self):
        rows = _read_csv(CATALOG_PATH)
        assert len(rows) == NUM_COURSES

    def test_catalog_schema(self):
        rows = _read_csv(CATALOG_PATH)
        expected = {"id", "title", "tags", "difficulty", "duration_minutes", "description"}
        assert set(rows[0].keys()) == expected

    def test_course_ids_unique(self):
        rows = _read_csv(CATALOG_PATH)
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("course-") for i in ids)

    def test_course_titles_nonempty(self):
        rows = _read_csv(CATALOG_PATH)
        assert all(r["title"].strip() for r in rows)

    def test_course_difficulty_valid(self):
        rows = _read_csv(CATALOG_PATH)
        assert all(r["difficulty"] in DIFFICULTIES for r in rows)

    def test_course_duration_range(self):
        rows = _read_csv(CATALOG_PATH)
        for r in rows:
            assert 10 <= int(r["duration_minutes"]) <= 480

    def test_course_description_min_len(self):
        rows = _read_csv(CATALOG_PATH)
        assert all(len(r["description"]) >= 50 for r in rows)

    def test_tags_pipe_separated(self):
        rows = _read_csv(CATALOG_PATH)
        for r in rows:
            tags = r["tags"].split("|")
            assert 2 <= len(tags) <= 4
            assert all(t.strip() for t in tags)

    def test_all_tags_used(self):
        rows = _read_csv(CATALOG_PATH)
        used = set()
        for r in rows:
            used.update(r["tags"].split("|"))
        expected_tags = set()
        for domain_tags in TAG_DOMAINS.values():
            expected_tags.update(domain_tags)
        assert used == expected_tags
        assert len(used) == 25

    def test_tag_usage_under_30_percent(self):
        rows = _read_csv(CATALOG_PATH)
        from collections import Counter
        counter = Counter()
        for r in rows:
            counter.update(r["tags"].split("|"))
        total = len(rows)
        for tag, count in counter.items():
            assert count / total <= 0.30, f"Tag {tag} at {count/total:.2%} > 30%"

    def test_tag_domains_five_each(self):
        # Verify config has 5 domains x 5 tags
        assert len(TAG_DOMAINS) == 5
        for domain, tags in TAG_DOMAINS.items():
            assert len(tags) == 5


class TestUsers:
    def test_users_exists(self):
        assert USERS_PATH.exists()

    def test_user_row_count(self):
        rows = _read_csv(USERS_PATH)
        assert len(rows) == NUM_USERS

    def test_user_schema(self):
        rows = _read_csv(USERS_PATH)
        assert set(rows[0].keys()) == {"user_id", "tier", "joined_days_ago"}

    def test_user_ids_unique(self):
        rows = _read_csv(USERS_PATH)
        ids = [r["user_id"] for r in rows]
        assert len(ids) == len(set(ids))
        assert all(i.startswith("user-") for i in ids)

    def test_user_tier_valid(self):
        rows = _read_csv(USERS_PATH)
        assert all(r["tier"] in USER_TIERS for r in rows)

    def test_joined_days_ago_nonneg(self):
        rows = _read_csv(USERS_PATH)
        assert all(int(r["joined_days_ago"]) >= 0 for r in rows)


class TestInteractions:
    def test_interactions_exists(self):
        assert INTERACTIONS_PATH.exists()

    def test_interaction_schema(self):
        rows = _read_csv(INTERACTIONS_PATH)
        assert set(rows[0].keys()) == {"id", "user_id", "course_id", "action", "rating", "timestamp"}

    def test_actions_valid(self):
        rows = _read_csv(INTERACTIONS_PATH)
        assert all(r["action"] in {"viewed", "completed", "rated"} for r in rows)

    def test_rating_only_when_rated(self):
        rows = _read_csv(INTERACTIONS_PATH)
        for r in rows:
            if r["action"] == "rated":
                assert r["rating"] != ""
                assert 1 <= int(r["rating"]) <= 5
            else:
                assert r["rating"] == ""

    def test_timestamps_iso8601(self):
        rows = _read_csv(INTERACTIONS_PATH)
        for r in rows:
            # Should parse as ISO-8601
            assert "T" in r["timestamp"] or "+" in r["timestamp"]

    def test_references_valid(self):
        rows = _read_csv(INTERACTIONS_PATH)
        catalog_rows = _read_csv(CATALOG_PATH)
        user_rows = _read_csv(USERS_PATH)
        course_ids = {r["id"] for r in catalog_rows}
        user_ids = {r["user_id"] for r in user_rows}
        for r in rows:
            assert r["course_id"] in course_ids
            assert r["user_id"] in user_ids


class TestDistribution:
    def test_interaction_count_distribution(self):
        rows = _read_csv(INTERACTIONS_PATH)
        from collections import Counter
        counts = Counter(r["user_id"] for r in rows)
        
        regular = 0
        power = 0
        sparse = 0
        for uid, count in counts.items():
            if count <= 2:
                sparse += 1
            elif count >= 16:
                power += 1
            else:
                regular += 1
        
        total = len(counts)
        # Allow some tolerance since it's random
        assert regular >= int(total * 0.70), f"Regular users too few: {regular}/{total}"
        assert power >= int(total * 0.10), f"Power users too few: {power}/{total}"
        assert sparse >= int(total * 0.03), f"Sparse users too few: {sparse}/{total}"


class TestSplits:
    def test_splits_exists(self):
        assert SPLITS_PATH.exists()

    def test_splits_schema(self):
        with open(SPLITS_PATH, encoding="utf-8") as f:
            splits = json.load(f)
        assert "train_interactions" in splits
        assert "test_holdout" in splits
        assert "challenge_users" in splits

    def test_train_test_temporal_split(self):
        with open(SPLITS_PATH, encoding="utf-8") as f:
            splits = json.load(f)
        train_ids = set(splits["train_interactions"])
        test_ids = set(splits["test_holdout"])
        challenge = set(splits["challenge_users"])

        # No overlap between train and test interaction ids
        assert train_ids.isdisjoint(test_ids)

        # All non-challenge users appear in BOTH train and test interaction data
        interactions = _read_csv(INTERACTIONS_PATH)
        challenge_users = set(splits["challenge_users"])
        by_user: dict[str, list[dict]] = {}
        for it in interactions:
            if it["user_id"] not in challenge_users:
                by_user.setdefault(it["user_id"], []).append(it)

        train_interaction_ids = set(splits["train_interactions"])
        test_interaction_ids = set(splits["test_holdout"])

        users_with_train = set()
        users_with_test = set()
        for it in interactions:
            uid = it["user_id"]
            if uid in challenge_users:
                continue
            if it["id"] in train_interaction_ids:
                users_with_train.add(uid)
            if it["id"] in test_interaction_ids:
                users_with_test.add(uid)

        # Every non-challenge user with >=3 interactions has test holdout
        non_challenge_users = set(by_user.keys())
        assert users_with_train == non_challenge_users, "All non-challenge users must have train interactions"
        
        # Users with >=3 interactions should be in test
        users_with_ge3 = {uid for uid, ints in by_user.items() if len(ints) >= 3}
        assert users_with_test == users_with_ge3, f"Users with >=3 interactions should have test holdout. Test users: {users_with_test}, Ge3 users: {users_with_ge3}"

        # Challenge count
        assert len(challenge) == CHALLENGE_USER_COUNT

        # Challenge users have no interactions in test holdout
        challenge_test = set()
        for it in interactions:
            if it["user_id"] in challenge_users and it["id"] in test_interaction_ids:
                challenge_test.add(it["id"])
        assert len(challenge_test) == 0, "Challenge users must have no interactions in test holdout"


class TestTemporalSplit:
    def test_test_interactions_are_latest_by_timestamp(self):
        with open(SPLITS_PATH, encoding="utf-8") as f:
            splits = json.load(f)
        train_ids = set(splits["train_interactions"])
        test_ids = set(splits["test_holdout"])
        challenge = set(splits["challenge_users"])

        interactions = _read_csv(INTERACTIONS_PATH)
        by_user: dict[str, list[dict]] = {}
        for it in interactions:
            if it["user_id"] not in challenge:
                by_user.setdefault(it["user_id"], []).append(it)

        for uid, user_ints in by_user.items():
            if len(user_ints) < 3:
                continue  # 1-2 interactions: all train
            user_test = [it for it in user_ints if it["id"] in test_ids]
            user_train = [it for it in user_ints if it["id"] in train_ids]
            if not user_test:
                continue
            # All test interactions must have timestamps >= all train interactions for this user
            max_train_ts = max(it["timestamp"] for it in user_train) if user_train else ""
            min_test_ts = min(it["timestamp"] for it in user_test)
            assert min_test_ts >= max_train_ts, f"Temporal split violated for user {uid}"

    def test_train_test_counts_match(self):
        with open(SPLITS_PATH, encoding="utf-8") as f:
            splits = json.load(f)
        interactions = _read_csv(INTERACTIONS_PATH)
        total = len(interactions)
        assert splits["train_count"] + splits["test_count"] == total


class TestReproducibility:
    def test_md5_checksums_match(self):
        with open(DATA_DIR / "checksums.json", encoding="utf-8") as f:
            checksums = json.load(f)
        for fname, expected_md5 in checksums.items():
            actual = _md5(DATA_DIR / fname)
            assert actual == expected_md5, f"MD5 mismatch for {fname}"

    def test_regenerate_produces_same_md5(self):
        from src.data.generator import generate_all
        result1 = generate_all(seed=SEED, num_courses=NUM_COURSES, num_users=NUM_USERS)
        result2 = generate_all(seed=SEED, num_courses=NUM_COURSES, num_users=NUM_USERS)
        assert result1["checksums"] == result2["checksums"]
