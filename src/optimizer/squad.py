"""
Pick the best squad of 15 players for one gameweek (mixed-integer program).
"""

import pulp

from src.modeling.predictions import predict_season

BUDGET = 1000  # £100m, prices are in tenths of £m
SQUAD_SIZE = 15
POSITION_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}  # goalkeeper, defender, midfielder, forward
MAX_PER_CLUB = 3

STARTING_ELEVEN = 11
FORMATION_LIMITS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}  # fewest and most starters per position
BENCH_WEIGHT = 0.1  # a bench player counts a little, because he can replace a starter


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
        baseline_points=("baseline_points", "sum"),
        total_points=("total_points", "sum"),
    )


def pick_squad(players, points_column):
    """
    Choose the 15 players that fit the rules (budget, number per position, at most
    MAX_PER_CLUB per club) and the starting 11 among them (formation limits).
    The starters count fully, the bench with BENCH_WEIGHT.
    Returns the squad with a True/False column "starts".
    """
    problem = pulp.LpProblem("squad", pulp.LpMaximize)
    pick = problem.add_variable_dict("pick", players.index, cat="Binary")
    start = problem.add_variable_dict("start", players.index, cat="Binary")

    points = players[points_column]
    problem += pulp.lpSum(
        points[i] * (start[i] + BENCH_WEIGHT * (pick[i] - start[i]))
        for i in players.index
    )

    problem += pulp.lpSum(pick.values()) == SQUAD_SIZE
    problem += pulp.lpSum(players.loc[i, "value"] * pick[i] for i in players.index) <= BUDGET
    for position, count in POSITION_COUNTS.items():
        in_position = players.index[players["position"] == position]
        problem += pulp.lpSum(pick[i] for i in in_position) == count
    for club in players["club"].unique():
        in_club = players.index[players["club"] == club]
        problem += pulp.lpSum(pick[i] for i in in_club) <= MAX_PER_CLUB

    for i in players.index:
        problem += start[i] <= pick[i]
    problem += pulp.lpSum(start.values()) == STARTING_ELEVEN
    for position, (fewest, most) in FORMATION_LIMITS.items():
        starters = pulp.lpSum(start[i] for i in players.index[players["position"] == position])
        problem += starters >= fewest
        problem += starters <= most

    problem.solve(pulp.HiGHS(msg=False))
    squad = players[[pick[i].value() > 0.5 for i in players.index]].copy()
    squad["starts"] = [start[i].value() > 0.5 for i in squad.index]
    return squad


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
    print(f"Real points (starting 11): {squad.loc[squad['starts'], 'total_points'].sum()}")


if __name__ == "__main__":
    main()
