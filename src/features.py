"""
Build the features for the points model from the SQLite database.
"""

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "fpl.db"

FORM_WINDOW = 5

FORM_SOURCES = {
    "total_points": "avg_points_last",
    "minutes": "avg_minutes_last",
}


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


def main():
    """
    Load the matches, add the features and show a quick check.
    """
    df = load_matches()
    df = add_form_features(df)
    print(df.shape)
    print(df[["element", "round", "total_points", "avg_points_last_5",
              "minutes", "avg_minutes_last_5"]].head(6))


if __name__ == "__main__":
    main()