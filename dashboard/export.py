"""
Export the predictions and the best squad for the next gameweek, for the dashboard.
"""
from src.config import CURRENT_SEASON, ROOT
from src.optimizer.next_gameweek import AVAILABLE, predict_next_gameweek
from src.optimizer.squad import POSITION_NAMES, load_club_names, one_row_per_player, pick_squad

DATA_DIR = ROOT / "dashboard" / "data"


def main():
    table = predict_next_gameweek()
    round_number = table["round"].iloc[0]
    table = table[table["status"] == AVAILABLE]

    players = one_row_per_player(table)
    squad = pick_squad(players, "predicted_points")
    players["round"] = round_number

    club_names = load_club_names(CURRENT_SEASON)
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
