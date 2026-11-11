"""
Build the rows of the next gameweek (matches that are not played yet),
with the same features as the finished matches.
"""

import sqlite3

import pandas as pd

from src.config import BASELINE_COLUMN, CURRENT_SEASON, DB_PATH
from src.modeling.features import (
    add_form_features,
    add_position_features,
    load_matches,
    scale_opponent_strength,
)

UPCOMING_QUERY = """
    SELECT p.season,
           p.id AS element,
           p.code,
           p.web_name,
           p.team AS club,
           p.element_type AS position,
           p.now_cost AS value,
           p.status,
           f.id AS fixture,
           f.event AS round,
           f.kickoff_time,
           1 AS was_home,
           f.team_a AS opponent_team,
           t.strength_overall_away AS opponent_strength
    FROM fixtures AS f
    JOIN players AS p ON p.season = f.season AND p.team = f.team_h
    JOIN teams AS t ON t.season = f.season AND t.id = f.team_a
    WHERE f.season = ? AND f.event = ? AND f.finished = 0

    UNION ALL

    SELECT p.season,
           p.id AS element,
           p.code,
           p.web_name,
           p.team AS club,
           p.element_type AS position,
           p.now_cost AS value,
           p.status,
           f.id AS fixture,
           f.event AS round,
           f.kickoff_time,
           0 AS was_home,
           f.team_h AS opponent_team,
           t.strength_overall_home AS opponent_strength
    FROM fixtures AS f
    JOIN players AS p ON p.season = f.season AND p.team = f.team_a
    JOIN teams AS t ON t.season = f.season AND t.id = f.team_h
    WHERE f.season = ? AND f.event = ? AND f.finished = 0;
"""


def next_round(season=CURRENT_SEASON):
    """
    The first gameweek of the season with a match that is not finished yet.
    """
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT MIN(event) FROM fixtures WHERE season = ? AND finished = 0"
    round_number = conn.execute(query, (season,)).fetchone()[0]
    conn.close()
    return round_number


def load_upcoming(round_number, season=CURRENT_SEASON):
    """
    One row per player and not finished match of the given gameweek.
    The result columns (points, minutes, starts) do not exist yet.
    """
    conn = sqlite3.connect(DB_PATH)
    params = [season, round_number, season, round_number]
    df = pd.read_sql(UPCOMING_QUERY, conn, params=params)
    conn.close()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])
    return df


def load_dataset_with_upcoming(round_number):
    """
    Finished matches plus the upcoming matches of the gameweek, with the same
    features. An upcoming row gets its form from the previous matches of the player.
    """
    matches = load_matches()
    upcoming = load_upcoming(round_number)

    df = pd.concat([matches, upcoming], ignore_index=True)
    df = df.sort_values(["code", "kickoff_time"]).reset_index(drop=True)
    df = add_form_features(df)
    df = add_position_features(df)
    df = scale_opponent_strength(df)
    return df


def main():
    """
    Show the number of rows of the next gameweek and the form of its most
    expensive player, to check that the upcoming row gets the right form.
    """
    round_number = next_round()
    df = load_dataset_with_upcoming(round_number)

    this_round = df[(df["season"] == CURRENT_SEASON) & (df["round"] == round_number)]
    print(f"Next gameweek: {round_number}, {len(this_round)} rows")

    star = this_round.sort_values("value").iloc[-1]["code"]
    history = df[(df["code"] == star) & (df["season"] == CURRENT_SEASON)]
    print(history[["web_name", "round", "total_points", BASELINE_COLUMN]].tail(7))


if __name__ == "__main__":
    main()
