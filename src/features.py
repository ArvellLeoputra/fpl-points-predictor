"""
Build the features for the points model from the SQLite database.
"""

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "fpl.db"

FORM_WINDOW = 5         # matches in the rolling averages

# Time split: train on earlier rounds, test on the latest full rounds
TEST_START_ROUND = 8
LAST_FULL_ROUND = 9     # round 10 is still partial (unfinished matches)

FORM_SOURCES = {
    "total_points": "avg_points_last",
    "minutes": "avg_minutes_last",
}

# the baseline prediction is the rolling average of the points
BASELINE_COLUMN = f"{FORM_SOURCES['total_points']}_{FORM_WINDOW}"


def load_matches():
    """
    Read all matches from the database, one row per player per match,
    sorted by player and kickoff time.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT *
        FROM gameweek_stats
        WHERE team_h_score IS NOT NULL
        ORDER BY element, kickoff_time;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])
    return df


def average_of_previous_x(values, x):
    """
    Average of the previous x values, current value excluded.
    `values` is one player's column in time order.
    """
    values = values.shift(1)
    return values.rolling(x, min_periods=1).mean()


def add_form_features(df):
    """
    Add rolling form features: for each source column, the average of the
    player's previous FORM_WINDOW matches (current match excluded).
    `df` must be sorted by player and kickoff time.
    """
    by_player = df.groupby("element")
    for source, prefix in FORM_SOURCES.items():
        df[f"{prefix}_{FORM_WINDOW}"] = by_player[source].transform(
            average_of_previous_x, FORM_WINDOW
        )
    return df


def split_train_test(df):
    """
    Split by time: earlier rounds for training, the latest full rounds
    for testing. Partial rounds after LAST_FULL_ROUND are dropped.
    """
    df = df[df["round"] <= LAST_FULL_ROUND]
    train = df[df["round"] < TEST_START_ROUND].copy()
    test = df[df["round"] >= TEST_START_ROUND].copy()
    return train, test


def baseline_mae(test):
    """
    Mean absolute error of the baseline prediction (average of the
    previous FORM_WINDOW matches) on the test set.
    Returns (all rows, played only).
    """
    rows = test.dropna(subset=[BASELINE_COLUMN])
    error = (rows["total_points"] - rows[BASELINE_COLUMN]).abs()

    mae_all = error.mean()
    mae_played = error[rows["minutes"] > 0].mean()
    return mae_all, mae_played


def main():
    """
    Load the matches, add the features and show a quick check.
    """
    df = load_matches()
    df = add_form_features(df)
    print(df.shape)

    train, test = split_train_test(df)
    print(f"Train: {len(train)} rows, test: {len(test)} rows")

    mae_all, mae_played = baseline_mae(test)
    print(f"Baseline MAE, all rows: {mae_all:.3f}")
    print(f"Baseline MAE, played only: {mae_played:.3f}")


if __name__ == "__main__":
    main()