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
        assert set(rows[0].keys()) == {"user_id", "course_id", "action", "rating", "timestamp"}

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
        assert "train_users" in splits
        assert "test_users" in splits
        assert "challenge_users" in splits

    def test_train_test_split_ratio(self):
        with open(SPLITS_PATH, encoding="utf-8") as f:
            splits = json.load(f)
        train = set(splits["train_users"])
        test = set(splits["test_users"])
        challenge = set(splits["challenge_users"])
        
        # No overlap between train/test/challenge
        assert train.isdisjoint(test)
        assert train.isdisjoint(challenge)
        assert test.isdisjoint(challenge)
        
        # All users accounted for
        all_users = train | test | challenge
        user_rows = _read_csv(USERS_PATH)
        all_user_ids = {r["user_id"] for r in user_rows}
        assert all_users == all_user_ids
        
        # Challenge count
        assert len(challenge) == CHALLENGE_USER_COUNT
        
        # Approximate 80/20 split
        non_challenge = train | test
        train_ratio = len(train) / len(non_challenge)
        assert abs(train_ratio - TRAIN_RATIO) < 0.05


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
