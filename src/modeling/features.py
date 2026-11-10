"""
Build the features for the points model from the SQLite database.
"""

import sqlite3

import pandas as pd

from src.config import (
    DB_PATH,
    FEATURE_COLUMNS,
    FORM_SOURCES,
    FORM_WINDOW,
    TEST_SEASON,
)


def load_matches():
    """
    Read all matches from the database, one row per player per match,
    sorted by player and kickoff time.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT gs.*,
               p.code,
               p.element_type AS position,
               CASE WHEN gs.was_home = 1 THEN t.strength_overall_away
                    ELSE t.strength_overall_home
               END AS opponent_strength
        FROM gameweek_stats AS gs
        JOIN players AS p ON p.season = gs.season AND p.id = gs.element
        JOIN teams AS t ON t.season = gs.season AND t.id = gs.opponent_team
        WHERE gs.team_h_score IS NOT NULL
        ORDER BY p.code, gs.kickoff_time;
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
    by_player = df.groupby("code")
    for source, prefix in FORM_SOURCES.items():
        df[f"{prefix}_{FORM_WINDOW}"] = by_player[source].transform(
            average_of_previous_x, FORM_WINDOW
        )
    return df


def add_position_features(df):
    """
    Add one 0/1 column per position (goalkeeper is the reference).
    """
    dummies = pd.get_dummies(df["position"], prefix="position", drop_first=True, dtype=int)
    df = pd.concat([df, dummies], axis=1)
    return df


def scale_opponent_strength(df):
    """
    Put opponent_strength on the same scale in every season, from 0 (weakest
    opponent of the season) to 1 (strongest).
    """
    by_season = df.groupby("season")["opponent_strength"]
    low = by_season.transform("min")
    high = by_season.transform("max")
    df["opponent_strength"] = (df["opponent_strength"] - low) / (high - low)
    return df


def load_dataset():
    """
    Load all matches with the form and position features added, and the opponent strength scaled.
    """
    df = load_matches()
    df = add_form_features(df)
    df = add_position_features(df)
    df = scale_opponent_strength(df)
    return df


def split_train_test(df):
    """
    Split by time: the seasons before TEST_SEASON for training,
    TEST_SEASON for testing. Later seasons are dropped.
    """
    train = df[df["season"] < TEST_SEASON].copy()
    test = df[df["season"] == TEST_SEASON].copy()
    return train, test


def make_xy(df):
    """
    Return the feature table X (FEATURE_COLUMNS) and the target y (total_points).
    Rows without a rolling average (a player's first match) are dropped.
    """
    rows = df.dropna(subset=FEATURE_COLUMNS)
    X = rows[FEATURE_COLUMNS]
    y = rows["total_points"]
    return X, y


def main():
    """
    Load the matches, add the features and show a quick check.
    """
    df = load_dataset()
    print(df.shape)

    train, test = split_train_test(df)
    print(f"Train: {len(train)} rows, test: {len(test)} rows")

    X_train, y_train = make_xy(train)
    X_test, y_test = make_xy(test)
    print(X_train.shape, X_test.shape)


if __name__ == "__main__":
    main()
