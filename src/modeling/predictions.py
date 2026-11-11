"""
Predicted points for every player-match of the test season.
"""

from src.config import TEST_SEASON
from src.modeling.features import (
    load_dataset,
    make_xy,
    split_train_test,
)
from src.modeling.models import (
    baseline_predictions,
    fit_two_part_model,
    predict_points,
)


def predict_season(test_season=TEST_SEASON):
    """
    Train the two-part model on the seasons before the test season and return
    one row per player-match of the test season, with predicted and real points.
    """
    df = load_dataset()

    train, test = split_train_test(df, test_season)
    X_train, y_train = make_xy(train)
    X_test, _ = make_xy(test)

    played_train = train.loc[y_train.index, "minutes"] > 0
    models = fit_two_part_model(X_train, y_train, played_train)

    columns = ["code", "web_name", "club", "position", "round", "value", "total_points"]
    table = test.loc[X_test.index, columns].copy()
    table["predicted_points"] = predict_points(models, X_test)
    table["baseline_points"] = baseline_predictions(X_test)

    return table


def main():
    """
    Show the ten players with the highest predicted points in gameweek 10 of 2026-27.
    """
    table = predict_season("2026-27")
    week = table[table["round"] == 10]
    print(week.sort_values("predicted_points", ascending=False).head(10))


if __name__ == "__main__":
    main()
