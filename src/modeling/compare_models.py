"""
Train the models and compare their test MAE with the baseline.
"""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from src.modeling.evaluate import (
    mean_absolute_error,
    by_group,
    print_results,
)
from src.modeling.features import (
    load_dataset,
    make_xy,
    split_train_test,
)
from src.modeling.models import baseline_predictions


def main():
    """
    Train the models and compare their test MAE with the baseline.
    """
    df = load_dataset()

    # Split the data into train and test sets
    train, test = split_train_test(df)

    # Create the feature and target tables for training and testing
    X_train, y_train = make_xy(train)
    X_test, y_test = make_xy(test)

    # True for the test rows where the player played
    played_test = test.loc[y_test.index, "minutes"] > 0

    results = {}

    base_predictions = baseline_predictions(X_test)
    results["baseline"] = by_group(mean_absolute_error, y_test, base_predictions, played_test)

    # Train a linear model and two boosted-tree models
    regressors = {
        "linear": LinearRegression(),
        "trees (median)": HistGradientBoostingRegressor(loss="absolute_error", random_state=0),
        "trees (mean)": HistGradientBoostingRegressor(loss="squared_error", random_state=0),
    }

    for name, model in regressors.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = by_group(mean_absolute_error, y_test, predictions, played_test)
    print_results(results)
    print()

    # Print the coefficients of the linear regression model
    linear = regressors["linear"]
    for feature, weight in zip(X_train.columns, linear.coef_):
        print(f"{feature}: {weight:.3f}")
    print(f"intercept: {linear.intercept_:.3f}")


if __name__ == "__main__":
    main()
