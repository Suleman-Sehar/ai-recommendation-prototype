"""Project configuration constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

CATALOG_PATH = DATA_DIR / "catalog.csv"
USERS_PATH = DATA_DIR / "users.csv"
INTERACTIONS_PATH = DATA_DIR / "interactions.csv"
SPLITS_PATH = DATA_DIR / "splits.json"

SEED = 42
NUM_COURSES = 500
NUM_USERS = 200
TRAIN_RATIO = 0.8
CHALLENGE_USER_COUNT = 20

TAG_DOMAINS = {
    "technology": ["python", "javascript", "sql", "cloud", "devops"],
    "business": ["management", "marketing", "finance", "strategy", "leadership"],
    "design": ["ux", "ui", "figma", "color-theory", "typography"],
    "science": ["physics", "biology", "chemistry", "astronomy", "mathematics"],
    "personal-development": ["productivity", "mindfulness", "communication", "habits", "career"],
}

ALL_TAGS = [tag for tags in TAG_DOMAINS.values() for tag in tags]
DIFFICULTIES = ["beginner", "intermediate", "advanced"]
USER_TIERS = ["free", "premium"]
INTERACTION_ACTIONS = ["viewed", "completed", "rated"]

DURATION_MIN = 10
DURATION_MAX = 480
DESCRIPTION_MIN_LEN = 50

RECOMMENDATION_N = 10
DIVERSITY_MAX_PER_TAG = 3
EVAL_KS = [5, 10, 20]
