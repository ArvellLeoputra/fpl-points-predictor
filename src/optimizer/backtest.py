"""
Backtest: pick a squad for every gameweek of the test season with different
point predictions and add up the real points of each squad.
"""

import pandas as pd

from src.modeling.predictions import predict_season
from src.optimizer.squad import one_row_per_player, pick_squad

FIRST_ROUND = 6

# squads are picked by these columns; the real points are always total_points
METHODS = {
    "model": "predicted_points",
    "baseline": "baseline_points",
    "perfect": "total_points",
}


def backtest(table, first_round=FIRST_ROUND):
    """
    For every gameweek from first_round on, pick one squad per method and
    add up its real points. Returns one row per gameweek, one column per method.
    """
    results = []
    for round_number in sorted(table["round"].unique()):
        if round_number < first_round:
            continue
        players = one_row_per_player(table[table["round"] == round_number])
        row = {"round": round_number}
        for method, column in METHODS.items():
            squad = pick_squad(players, column)
            row[method] = squad.loc[squad["starts"], "total_points"].sum()
        results.append(row)
    return pd.DataFrame(results).set_index("round")


def main():
    """
    Backtest on the test season and print the average real points per gameweek.
    """
    results = backtest(predict_season())
    print(results)
    print()
    print("Average real points per gameweek:")
    print(results.mean().round(1))
    print()
    print(f"Model beats baseline in {(results['model'] > results['baseline']).mean():.0%} of the gameweeks")


if __name__ == "__main__":
    main()
