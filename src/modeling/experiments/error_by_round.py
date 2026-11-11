"""
Error of the baseline and the two-part model per block of gameweeks
"""

import pandas as pd

from src.modeling.evaluate import mean_absolute_error
from src.modeling.features import (
    load_dataset,
    make_xy,
    split_train_test,
)
from src.modeling.models import (
    baseline_predictions,
    train_play_model,
    train_points_model,
)


def main():
    """
    Train the baseline and the two-part model and print the MAE per block of gameweeks.
    """
    df = load_dataset()

    train, test = split_train_test(df)
    X_train, y_train = make_xy(train)
    X_test, y_test = make_xy(test)

    rounds = test.loc[y_test.index, "round"]
    blocks = pd.cut(rounds, bins=[0, 5, 15, 38], labels=["1-5", "6-15", "16-38"])

    played_train = train.loc[y_train.index, "minutes"] > 0

    play_model = train_play_model(X_train, played_train)
    probabilities = play_model.predict_proba(X_test)[:, 1]

    points_model = train_points_model(X_train[played_train], y_train[played_train])
    expected_points = points_model.predict(X_test)

    predictions = {
        "baseline": baseline_predictions(X_test),
        "two-part": probabilities * expected_points,
    }

    results = {}
    for name, values in predictions.items():
        results[name] = {
            block: mean_absolute_error(y_test[blocks == block], values[blocks == block])
            for block in blocks.cat.categories
        }
        
    print("MAE per block of gameweeks (test season):")
    print(pd.DataFrame(results).round(3))


if __name__ == "__main__":
    main()
