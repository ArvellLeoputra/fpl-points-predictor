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


def fit_two_part_model(X_train, y_train, played_train):
    """
    Train both parts of the two-part model: the probability that the player
    plays, and the points if he plays (learned only from rows where he played).
    """
    play_model = train_play_model(X_train, played_train)
    points_model = train_points_model(X_train[played_train], y_train[played_train])
    return play_model, points_model


def predict_points(models, X):
    """
    Expected points: probability that the player plays x points if he plays.
    """
    play_model, points_model = models
    return play_model.predict_proba(X)[:, 1] * points_model.predict(X)
