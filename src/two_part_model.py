"""
Two-part prediction: P(player plays) x expected points if he plays
"""

from sklearn.ensemble import HistGradientBoostingRegressor

from config import BASELINE_COLUMN
from train import mae_by_group, print_results
from minutes_model import train_play_model
from features import (
    add_form_features,
    add_position_features,
    load_matches,
    make_xy,
    split_train_test,
)


def train_points_model(X_train, y_train, played_train):
    """
    Train a model for the expected points of a player who plays.
    Only the rows where the player played are used for training.
    """
    model = HistGradientBoostingRegressor(loss="squared_error")
    model.fit(X_train[played_train], y_train[played_train])
    return model


def rmse(y_true, predictions):
    """
    Root mean squared error: the typical miss, with large misses counting more.
    """
    return ((y_true - predictions) ** 2).mean() ** 0.5


def main():
    """
    Combine both parts and compare with the baseline and a single model.
    """
    df = load_matches()
    df = add_form_features(df)
    df = add_position_features(df)

    train, test = split_train_test(df)
    X_train, y_train = make_xy(train)
    X_test, y_test = make_xy(test)

    played_train = train.loc[y_train.index, "minutes"] > 0
    played_test = test.loc[y_test.index, "minutes"] > 0

    # Probability that the player plays
    play_model = train_play_model(X_train, played_train)
    probabilities = play_model.predict_proba(X_test)[:, 1]

    # Expected points if the player plays
    points_model = train_points_model(X_train, y_train, played_train)
    expected_points = points_model.predict(X_test)

    # Expected points = P(plays) x points if he plays
    two_part_predictions = probabilities * expected_points

    # One model that predicts the mean directly, trained on all rows
    single_model = HistGradientBoostingRegressor(loss="squared_error")
    single_model.fit(X_train, y_train)
    single_predictions = single_model.predict(X_test)

    results = {
        "baseline": mae_by_group(y_test, X_test[BASELINE_COLUMN], played_test),
        "two-part": mae_by_group(y_test, two_part_predictions, played_test),
    }
    print_results(results)
    print()

    # Compare the three on the same test rows
    print(f"RMSE baseline:      {rmse(y_test, X_test[BASELINE_COLUMN]):.3f}")
    print(f"RMSE single trees:  {rmse(y_test, single_predictions):.3f}")
    print(f"RMSE two-part:      {rmse(y_test, two_part_predictions):.3f}")


if __name__ == "__main__":
    main()
