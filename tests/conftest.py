"""Pytest configuration and fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import DATA_DIR, SEED
from src.data.generator import generate_all


@pytest.fixture(scope="session")
def generated_data(tmp_path_factory):
    """Generate data once per test session into a temp dir."""
    import tempfile
    import os

    # We generate into the actual data dir for reproducibility checks
    result = generate_all(seed=SEED, num_courses=500, num_users=200)
    return result


@pytest.fixture(scope="session")
def splits():
    with open(DATA_DIR / "splits.json", encoding="utf-8") as f:
        return json.load(f)
