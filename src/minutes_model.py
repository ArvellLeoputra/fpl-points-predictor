"""
Predict whether a player plays (at least one minute) in the next match.
"""

from features import (
    load_dataset,
    make_xy,
    split_train_test,
)
from models import train_play_model


def main():
    """
    Compare a simple rule with a classifier for 'does the player play?'
    """
    df = load_dataset()

    train, test = split_train_test(df)
    X_train, y_train = make_xy(train)
    X_test, y_test = make_xy(test)

    played_train = train.loc[y_train.index, "minutes"] > 0
    played_test = test.loc[y_test.index, "minutes"] > 0

    # Rule: a player plays if he had minutes in the last 5 gameweeks
    rule_predictions = X_test["avg_minutes_last_5"] > 0

    # Classifier: learns from all features
    play_model = train_play_model(X_train, played_train)
    play_predictions = play_model.predict(X_test)
    play_probabilities = play_model.predict_proba(X_test)[:, 1]

    rule_accuracy = (rule_predictions == played_test).mean()
    model_accuracy = (play_predictions == played_test).mean()

    print("Share of players who play (test gameweeks):")
    print(f"  real:                       {played_test.mean():.3f}")
    print(f"  rule says plays:            {rule_predictions.mean():.3f}")
    print(f"  classifier says plays:      {play_predictions.mean():.3f}")
    print(f"  classifier avg probability: {play_probabilities.mean():.3f}")
    print()
    print("Accuracy of 'does the player play?':")
    print(f"  rule (last 5 gameweeks):    {rule_accuracy:.3f}")
    print(f"  classifier (all features):  {model_accuracy:.3f}")


if __name__ == "__main__":
    main()
