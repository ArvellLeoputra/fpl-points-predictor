"""
Measure how far the predictions are from the real points, and print the results.
"""


def mean_absolute_error(y_true, predictions):
    """
    Mean absolute error (MAE): the average number of points the prediction is off.
    """
    return (y_true - predictions).abs().mean()


def root_mean_square_error(y_true, predictions):
    """
    Root mean squared error (RMSE): the typical miss, with large misses counting more.
    """
    return ((y_true - predictions) ** 2).mean() ** 0.5


def by_group(measure, y_true, predictions, played):
    """
    Apply an error measure (MAE or RMSE) on all rows, on rows where the
    player played, and on rows where he did not.
    Returns (all, played, not played).
    """
    return (
        measure(y_true, predictions),
        measure(y_true[played], predictions[played]),
        measure(y_true[~played], predictions[~played]),
    )


def print_results(results):
    """
    Print one row per method with its error on all rows, on rows where
    the player played, and on rows where he did not.
    """
    print(f"{'':16}{'all':>8}{'played':>8}{'not played':>12}")
    for name, (all_rows, played, not_played) in results.items():
        print(f"{name:16}{all_rows:8.3f}{played:8.3f}{not_played:12.3f}")
