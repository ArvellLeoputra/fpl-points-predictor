"""
Train the models and compare their test MAE with the baseline.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

from config import BASELINE_COLUMN
from features import (
    add_form_features,
    add_position_features,
    load_matches,
    make_xy,
    split_train_test,
)


def mae_by_group(y_true, predictions, played):
    """
    Mean absolute error on all rows, on rows where the player played,
    and on rows where he did not. Returns (all, played, not played).
    """
    error = (y_true - predictions).abs()
    mae_all = error.mean()
    mae_played = error[played].mean()
    mae_not_played = error[~played].mean()
    return mae_all, mae_played, mae_not_played


def print_results(results):
    """
    Print one row per method with its MAE on all rows, on rows where
    the player played, and on rows where he did not.
    """
    print(f"{'':16}{'all':>8}{'played':>8}{'not played':>12}")
    for name, (mae_all, mae_played, mae_not_played) in results.items():
        print(f"{name:16}{mae_all:8.3f}{mae_played:8.3f}{mae_not_played:12.3f}")


def main():
    """
    Train the models and compare their test MAE with the baseline.
    """
    df = load_matches()
    df = add_form_features(df)
    df = add_position_features(df)

    # Split the data into train and test sets
    train, test = split_train_test(df)

    # Create the feature and target tables for training and testing
    X_train, y_train = make_xy(train)
    X_test, y_test = make_xy(test)

    # Classify rows in the test set whether the player played or not
    played = test.loc[y_test.index, "minutes"] > 0

    results = {"baseline": mae_by_group(y_test, X_test[BASELINE_COLUMN], played)}

    # Train a linear model and two boosted-tree models
    models = {
        "linear": LinearRegression(),
        "trees (median)": HistGradientBoostingRegressor(loss="absolute_error"),
        "trees (mean)": HistGradientBoostingRegressor(loss="squared_error")
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = mae_by_group(y_test, model.predict(X_test), played)
    print_results(results)

    # Print the coefficients of the linear regression model
    linear = models["linear"]
    for name, weight in zip(X_train.columns, linear.coef_):
        print(f"{name}: {weight:.3f}")
    print(f"intercept: {linear.intercept_:.3f}")


if __name__ == "__main__":
    main()
