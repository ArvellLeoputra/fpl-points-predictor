import pandas as pd
import pytest

from src.modeling.evaluate import by_group, mean_absolute_error, root_mean_square_error


def test_mae_perfect_prediction():
    y_true = pd.Series([2, 6, 1])
    assert mean_absolute_error(y_true, y_true) == 0


def test_mae_small_example():
    # misses are 0, 0 and 2, so the average miss is 2 / 3
    y_true = pd.Series([1, 2, 3])
    predictions = pd.Series([1, 2, 5])
    assert mean_absolute_error(y_true, predictions) == pytest.approx(2 / 3)


def test_mae_both_directions():
    # one prediction is 2 too high, one is 2 too low, both count as 2
    y_true = pd.Series([4, 4])
    predictions = pd.Series([6, 2])
    assert mean_absolute_error(y_true, predictions) == 2


def test_rmse_small_example():
    # squared misses are 0, 0 and 4, mean is 4 / 3
    y_true = pd.Series([1, 2, 3])
    predictions = pd.Series([1, 2, 5])
    assert root_mean_square_error(y_true, predictions) == pytest.approx((4 / 3) ** 0.5)


def test_rmse_vs_mae():
    # same total miss (4 points), but in one big miss instead of two small ones
    y_true = pd.Series([0, 0])
    one_big_miss = pd.Series([4, 0])
    two_small_misses = pd.Series([2, 2])
    assert mean_absolute_error(y_true, one_big_miss) == mean_absolute_error(y_true, two_small_misses)
    assert root_mean_square_error(y_true, one_big_miss) > root_mean_square_error(y_true, two_small_misses)


def test_played_and_not_played():
    y_true = pd.Series([5, 3, 0, 0])
    predictions = pd.Series([3, 3, 1, 0])
    played = pd.Series([True, True, False, False])

    all_rows, played_rows, not_played_rows = by_group(mean_absolute_error, y_true, predictions, played)

    assert all_rows == pytest.approx(0.75)        # misses 2, 0, 1, 0
    assert played_rows == pytest.approx(1.0)      # misses 2, 0
    assert not_played_rows == pytest.approx(0.5)  # misses 1, 0
