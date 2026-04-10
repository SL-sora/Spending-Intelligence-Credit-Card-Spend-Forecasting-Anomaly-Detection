import os

# Repo root (this file lives at project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EDA_IMG_DIR = os.path.join(PROJECT_ROOT, "EDA_img")
VALIDATION_IMG_DIR = os.path.join(PROJECT_ROOT, "validation_img")
SUMMARY_CSV_DIR = os.path.join(PROJECT_ROOT, "summaries")

WEEKLY_FREQ = "W-MON"
DB_CONNECT_TIMEOUT_SECONDS = 5
AWS_CONNECT_TIMEOUT_SECONDS = 5
AWS_READ_TIMEOUT_SECONDS = 10
AWS_MAX_RETRIES = 2
ANOMALY_ZSCORE_THRESHOLD = 2.0
ANOMALY_MIN_ABS_DELTA = 25.0
ANOMALY_MIN_PCT_DELTA = 0.15
TARGET_INTERVAL_COVERAGE = 0.90

CATEGORIES = [
    "Food and Drink",
    "Travel",
    "Shopping",
    "Groceries",
    "Entertainment",
    "Health",
    "Utilities",
]


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


# Illustrative defaults only (round placeholder amounts). Set per-category budgets via
# BUDGET_<CATEGORY> with spaces as underscores, e.g. BUDGET_FOOD_AND_DRINK=450.
_ILLUSTRATIVE_BUDGET_DEFAULTS: dict[str, int] = {
    "Food and Drink": 400,
    "Travel": 500,
    "Shopping": 350,
    "Groceries": 450,
    "Entertainment": 150,
    "Health": 200,
    "Utilities": 175,
    "Uncategorized": 250,
}

BUDGETS = {
    category: _int_from_env(
        f"BUDGET_{category.replace(' ', '_').upper()}",
        default,
    )
    for category, default in _ILLUSTRATIVE_BUDGET_DEFAULTS.items()
}
