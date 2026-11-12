"""
Predicted points for every player-match of the test season.
"""

import pandas as pd

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


def predict_rows(train, test):
    """
    Train the two-part model on the train rows and return one row
    per test player-match, with predicted and real points.
    """
    X_train, y_train = make_xy(train)
    X_test, _ = make_xy(test)

    played_train = train.loc[y_train.index, "minutes"] > 0
    models = fit_two_part_model(X_train, y_train, played_train)

    columns = ["code", "web_name", "club", "position", "round", "value", "selected", "total_points"]
    table = test.loc[X_test.index, columns].copy()
    table["predicted_points"] = predict_points(models, X_test)
    table["baseline_points"] = baseline_predictions(X_test)

    return table


def predict_season(test_season=TEST_SEASON):
    """
    Train once on the seasons before the test season and predict the whole test season.
    """
    df = load_dataset()
    train, test = split_train_test(df, test_season)
    return predict_rows(train, test)


def predict_season_retrained(test_season=TEST_SEASON):
    """
    Predict the test season gameweek by gameweek. For every gameweek the model is
    trained again on the earlier seasons and the finished gameweeks of the test season.
    """
    df = load_dataset()
    prev_seasons = df[df["season"] < test_season]
    curr_season = df[df["season"] == test_season]

    tables = []
    for round_number in sorted(curr_season["round"].unique()):
        if round_number == 1:
            continue  # no form yet, so nothing to predict
        finished = curr_season[curr_season["round"] < round_number]
        train = pd.concat([prev_seasons, finished])
        test = curr_season[curr_season["round"] == round_number]
        tables.append(predict_rows(train, test))
    return pd.concat(tables)


def main():
    """
    Show the ten players with the highest predicted points in gameweek 10 of 2026-27.
    """
    table = predict_season("2026-27")
    week = table[table["round"] == 10]
    print(week.sort_values("predicted_points", ascending=False).head(10))


if __name__ == "__main__":
    main()
