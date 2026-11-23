"""
Export the predictions and the best squad for the next gameweek, for the dashboard.
"""
import sqlite3
from datetime import date

import pandas as pd

from src.config import CURRENT_SEASON, DB_PATH, ROOT
from src.optimizer.next_gameweek import AVAILABLE, predict_next_gameweek
from src.optimizer.squad import (
    POSITION_NAMES,
    load_club_names,
    one_row_per_player,
    pick_squad,
)

DATA_DIR = ROOT / "dashboard" / "data"


def load_ownership(season):
    """
    Ownership in percent for every player of the season, from the FPL player list.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT code, selected_by_percent FROM players WHERE season = ?",
                     conn, params=(season,))
    conn.close()
    return df.set_index("code")["selected_by_percent"]


def main():
    table = predict_next_gameweek()
    round_number = table["round"].iloc[0]
    table = table[table["status"] == AVAILABLE].copy()

    players = one_row_per_player(table)
    club_names = load_club_names(CURRENT_SEASON)
    home_or_away = table["was_home"].map({1: " (H)", 0: " (A)"})
    table["fixture"] = table["opponent_team"].map(club_names) + home_or_away
    players["fixture"] = table.groupby("code")["fixture"].agg(", ".join)
    players["ownership"] = load_ownership(CURRENT_SEASON)
    players["round"] = round_number
    players["updated"] = date.today().isoformat()

    squad = pick_squad(players, "predicted_points")

    for df in (players, squad):
        df["club"] = df["club"].map(club_names)
        df["position"] = df["position"].map(POSITION_NAMES)

    players = players.drop(columns=["selected", "total_points"])
    squad = squad.drop(columns=["selected", "total_points"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    players.to_csv(DATA_DIR / "next_gameweek.csv")
    squad.to_csv(DATA_DIR / "squad.csv")


if __name__ == "__main__":
    main()
