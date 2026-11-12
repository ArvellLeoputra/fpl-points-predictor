"""
Settings shared by the scripts: paths, the time split and the features.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "fpl.db"

# Seasons
CURRENT_SEASON = "2026-27"
PAST_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

FORM_WINDOW = 5         # matches in the rolling averages

# Time split: train on the seasons before the test season
TEST_SEASON = "2025-26"

# Form related features
FORM_SOURCES = {
    "total_points": "avg_points_last",
    "minutes": "avg_minutes_last",
    "starts": "avg_starts_last",
}

# One 0/1 column per position (goalkeeper is the reference)
POSITION_COLUMNS = ["position_2", "position_3", "position_4"]

# Short position names for printing
POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

FORM_COLUMNS = [f"{prefix}_{FORM_WINDOW}" for prefix in FORM_SOURCES.values()]

# The features used for training. opponent_strength is FPL's rating of the opponent
# (for the home or away side it plays), scaled from 0 to 1 within each season.
FEATURE_COLUMNS = FORM_COLUMNS + POSITION_COLUMNS + ["was_home", "value", "opponent_strength"]

# The baseline prediction is the rolling average of the points
BASELINE_COLUMN = f"{FORM_SOURCES['total_points']}_{FORM_WINDOW}"
