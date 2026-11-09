"""
Settings shared by the scripts: paths, the time split and the features.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "fpl.db"

FORM_WINDOW = 5         # matches in the rolling averages

# Time split: train on earlier rounds, test on the latest full rounds
TEST_START_ROUND = 8
LAST_FULL_ROUND = 9     # round 10 is still partial (unfinished matches)

# Form related features
FORM_SOURCES = {
    "total_points": "avg_points_last",
    "minutes": "avg_minutes_last",
}

# The features used for training.
FORM_COLUMNS = [f"{prefix}_{FORM_WINDOW}" for prefix in FORM_SOURCES.values()]
FEATURE_COLUMNS = FORM_COLUMNS + ["was_home", "value"]

# The baseline prediction is the rolling average of the points
BASELINE_COLUMN = f"{FORM_SOURCES['total_points']}_{FORM_WINDOW}"
