"""
Pick the best squad of 15 players for one gameweek (mixed-integer program).
"""

import pulp

from src.modeling.predictions import predict_season

BUDGET = 1000  # £100m, prices are in tenths of £m
SQUAD_SIZE = 15
POSITION_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}  # goalkeeper, defender, midfielder, forward
MAX_PER_CLUB = 3


def one_row_per_player(week):
    """
    A player has two rows in a double gameweek. Add up his points, keep the rest.
    """
    return week.groupby("code").agg(
        web_name=("web_name", "first"),
        club=("club", "first"),
        position=("position", "first"),
        value=("value", "first"),
        predicted_points=("predicted_points", "sum"),
        total_points=("total_points", "sum"),
    )


def pick_squad(players, points_column):
    """
    Choose the 15 players with the most points (points_column) that fit the rules:
    budget, number per position and at most MAX_PER_CLUB players from one club.
    """
    problem = pulp.LpProblem("squad", pulp.LpMaximize)
    pick = problem.add_variable_dict("pick", players.index, cat="Binary")

    problem += pulp.lpSum(players.loc[i, points_column] * pick[i] for i in players.index)

    problem += pulp.lpSum(pick.values()) == SQUAD_SIZE
    problem += pulp.lpSum(players.loc[i, "value"] * pick[i] for i in players.index) <= BUDGET
    for position, count in POSITION_COUNTS.items():
        in_position = players.index[players["position"] == position]
        problem += pulp.lpSum(pick[i] for i in in_position) == count
    for club in players["club"].unique():
        in_club = players.index[players["club"] == club]
        problem += pulp.lpSum(pick[i] for i in in_club) <= MAX_PER_CLUB

    problem.solve(pulp.HiGHS(msg=False))
    return players[[pick[i].value() > 0.5 for i in players.index]]


def main():
    """
    Pick a squad for gameweek 10 of 2026-27 and show what it costs and scores.
    """
    table = predict_season("2026-27")
    players = one_row_per_player(table[table["round"] == 10])
    squad = pick_squad(players, "predicted_points")

    print(squad.sort_values(["position", "predicted_points"], ascending=[True, False]))
    print(f"Price: {squad['value'].sum() / 10:.1f}m")
    print(f"Predicted points: {squad['predicted_points'].sum():.1f}")
    print(f"Real points: {squad['total_points'].sum()}")


if __name__ == "__main__":
    main()
