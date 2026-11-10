"""
The baseline and the models used in the experiments.
"""

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from src.config import BASELINE_COLUMN


def baseline_predictions(X):
    """
    The baseline prediction: the player's average points
    over the previous FORM_WINDOW matches.
    """
    return X[BASELINE_COLUMN]


def train_play_model(X_train, played_train):
    """
    Train a model for the probability that a player plays.
    """
    model = HistGradientBoostingClassifier(random_state=0)
    model.fit(X_train, played_train)
    return model


def train_points_model(X_train, y_train):
    """
    Train a model for the expected points of a player.
    It learns from whichever rows it is given.
    """
    model = HistGradientBoostingRegressor(loss="squared_error", random_state=0)
    model.fit(X_train, y_train)
    return model
