"""
Compare training once on the earlier seasons with training again every gameweek.
"""

from src.modeling.evaluate import mean_absolute_error, root_mean_square_error
from src.modeling.predictions import predict_season, predict_season_retrained
from src.optimizer.backtest import FIRST_ROUND, backtest


def main():
    """
    Print the errors of the predictions and the real points of the picked squads
    for both ways of training, from FIRST_ROUND on.
    """
    once = predict_season()
    weekly = predict_season_retrained().loc[once.index]  # same rows, same order

    print(f"{'':10}{'RMSE':>8}{'MAE':>8}{'model':>8}{'baseline':>10}{'popular':>9}")
    for name, table in [("once", once), ("weekly", weekly)]:
        rows = table[table["round"] >= FIRST_ROUND]
        rmse = root_mean_square_error(rows["total_points"], rows["predicted_points"])
        mae = mean_absolute_error(rows["total_points"], rows["predicted_points"])
        averages = backtest(table).mean()
        print(f"{name:10}{rmse:8.3f}{mae:8.3f}{averages['model']:8.1f}"
              f"{averages['baseline']:10.1f}{averages['popular']:9.1f}")


if __name__ == "__main__":
    main()
