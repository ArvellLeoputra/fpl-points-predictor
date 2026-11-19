import pandas as pd

from src.modeling.features import average_of_previous_x


def test_no_form():
    values = pd.Series([6, 2, 9])
    result = average_of_previous_x(values, 5)
    assert pd.isna(result[0])


def test_leakage():
    values = pd.Series([10, 0, 0])
    result = average_of_previous_x(values, 5)
    expected = pd.Series([float("nan"), 10, 5])
    pd.testing.assert_series_equal(result, expected)


def test_correctness():
    values = pd.Series([1, 2, 3, 4])
    result = average_of_previous_x(values, 2)
    expected = pd.Series([float("nan"), 1, 1.5, 2.5])
    pd.testing.assert_series_equal(result, expected)
