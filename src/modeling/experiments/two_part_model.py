"""
Two-part prediction: P(player plays) x expected points if he plays
"""

from src.modeling.evaluate import (
    mean_absolute_error,
    root_mean_square_error,
    by_group,
    print_results,
)
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
    Combine both parts and compare with the baseline and a single model.
    """
    df = load_dataset()

    train, test = split_train_test(df)
    X_train, y_train = make_xy(train)
    X_test, y_test = make_xy(test)

    played_train = train.loc[y_train.index, "minutes"] > 0
    played_test = test.loc[y_test.index, "minutes"] > 0

    # Single model: learns from all rows
    single_model = train_points_model(X_train, y_train)

    # Two-part model, part 1: probability that the player plays
    play_model = train_play_model(X_train, played_train)
    probabilities = play_model.predict_proba(X_test)[:, 1]

    # Two-part model, part 2: expected points if he plays
    points_model = train_points_model(X_train[played_train], y_train[played_train])
    expected_points = points_model.predict(X_test)

    predictions = {
        "baseline": baseline_predictions(X_test),
        "single model": single_model.predict(X_test),
        "two-part": probabilities * expected_points,
    }

    measures = {
        "MAE": mean_absolute_error,
        "RMSE": root_mean_square_error,
    }
    for title, measure in measures.items():
        results = {
            name: by_group(measure, y_test, values, played_test)
            for name, values in predictions.items()
        }
        print(f"{title}:")
        print_results(results)
        print()


if __name__ == "__main__":
    main()
