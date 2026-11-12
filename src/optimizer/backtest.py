"""
Backtest: pick a squad for every gameweek of the test season with different
point predictions and add up the real points of each squad.
"""

import pandas as pd

from src.config import ROOT
from src.modeling.predictions import predict_season_retrained
from src.optimizer.squad import one_row_per_player, pick_squad, squad_points

FIRST_ROUND = 6
RESULTS_PATH = ROOT / "docs" / "backtest_results.csv"

# squads are picked by these columns; the real points are always total_points
METHODS = {
    "model": "predicted_points",
    "baseline": "baseline_points",
    "popular": "selected",
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
            row[method] = squad_points(squad, "total_points")
        results.append(row)
    return pd.DataFrame(results).set_index("round")


def main():
    """
    Backtest on the test season and print the average real points per gameweek.
    """
    results = backtest(predict_season_retrained())
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS_PATH)
    print(results)
    print()
    print("Average real points per gameweek:")
    print(results.mean().round(1))
    print()
    print(f"Model beats baseline in {(results['model'] > results['baseline']).mean():.0%} of the gameweeks")
    print(f"Model beats most-owned in {(results['model'] > results['popular']).mean():.0%} of the gameweeks")


if __name__ == "__main__":
    main()
