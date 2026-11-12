"""
Pick a squad for the next gameweek with the two-part model and the optimizer.
"""

from src.config import CURRENT_SEASON
from src.modeling.features import make_xy
from src.modeling.models import (
    baseline_predictions,
    fit_two_part_model,
    predict_points,
)
from src.modeling.upcoming import load_dataset_with_upcoming, next_round
from src.optimizer.squad import one_row_per_player, pick_squad, print_squad

AVAILABLE = "a"  # status of a player who is fit


def predict_next_gameweek():
    """
    Train the two-part model on all finished matches and return one row per
    player and match of the next gameweek, with his predicted points.
    """
    round_number = next_round()
    df = load_dataset_with_upcoming(round_number)

    train = df[df["minutes"].notna()]
    X_train, y_train = make_xy(train)
    played_train = train.loc[X_train.index, "minutes"] > 0
    models = fit_two_part_model(X_train, y_train, played_train)

    upcoming = df[(df["season"] == CURRENT_SEASON) & (df["round"] == round_number) & (df["minutes"].isna())]
    X_new, _ = make_xy(upcoming)

    columns = ["code", "web_name", "club", "position", "round", "value", "selected", "status", "total_points"]
    table = upcoming.loc[X_new.index, columns].copy()
    table["predicted_points"] = predict_points(models, X_new)
    table["baseline_points"] = baseline_predictions(X_new)
    return table


def main():
    """
    Pick the best squad for the next gameweek and show it.
    """
    table = predict_next_gameweek()
    round_number = table["round"].iloc[0]
    table = table[table["status"] == AVAILABLE]

    players = one_row_per_player(table)
    squad = pick_squad(players, "predicted_points")

    print(f"Squad for gameweek {round_number}:")
    print_squad(squad, CURRENT_SEASON)


if __name__ == "__main__":
    main()
