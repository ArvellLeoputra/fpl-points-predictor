"""
Pick the best squad of 15 players for one gameweek.
"""

import sqlite3

import pandas as pd
import pulp

from src.config import DB_PATH, POSITION_NAMES
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
        selected=("selected", "first"),
    )


def pick_squad(players, points_column):
    """
    Choose the 15 players that fit the rules (budget, number per position, at most
    MAX_PER_CLUB per club) and the starting 11 among them (formation limits).
    Returns the squad with a True/False column "starts" and "captain".
    """
    problem = pulp.LpProblem("squad", pulp.LpMaximize)
    pick = problem.add_variable_dict("pick", players.index, cat="Binary")
    start = problem.add_variable_dict("start", players.index, cat="Binary")
    captain = problem.add_variable_dict("captain", players.index, cat="Binary")

    points = players[points_column]
    problem += pulp.lpSum(
        points[i] * (start[i] + BENCH_WEIGHT * (pick[i] - start[i]) + captain[i]) for i in players.index
    )

    # Total pick and budget constraints
    problem += pulp.lpSum(pick.values()) == SQUAD_SIZE
    problem += pulp.lpSum(players.loc[i, "value"] * pick[i] for i in players.index) <= BUDGET

    # Player limit picked on each position
    for position, count in POSITION_COUNTS.items():
        in_position = players.index[players["position"] == position]
        problem += pulp.lpSum(pick[i] for i in in_position) == count

    # Max 3 players per club
    for club in players["club"].unique():
        in_club = players.index[players["club"] == club]
        problem += pulp.lpSum(pick[i] for i in in_club) <= MAX_PER_CLUB

    # 11 starters (start = 1) and they must be in squad (pick = 1)
    for i in players.index:
        problem += start[i] <= pick[i]
    problem += pulp.lpSum(start.values()) == STARTING_ELEVEN

    # One captain and captain must be in squad and starting
    problem += pulp.lpSum(captain.values()) == 1
    for i in players.index:
        problem += captain[i] <= start[i]

    # Starting player limit on each position
    for position, (fewest, most) in FORMATION_LIMITS.items():
        starters = pulp.lpSum(start[i] for i in players.index[players["position"] == position])
        problem += starters >= fewest
        problem += starters <= most

    problem.solve(pulp.HiGHS(msg=False))
    squad = players[[pick[i].value() > 0.5 for i in players.index]].copy()
    squad["starts"] = [start[i].value() > 0.5 for i in squad.index]
    squad["captain"] = [captain[i].value() > 0.5 for i in squad.index]
    return squad


def squad_points(squad, points_column):
    """
    Points of a picked squad in the given column: the 11 starters plus the
    captain once more, because his points count double.
    """
    starters = squad[squad["starts"]]
    captain = squad[squad["captain"]]
    return starters[points_column].sum() + captain[points_column].sum()


def load_club_names(season):
    """
    Short club names by club id for one season.
    """
    conn = sqlite3.connect(DB_PATH)
    teams = pd.read_sql("SELECT id, short_name FROM teams WHERE season = ?", conn, params=[season])
    conn.close()
    return dict(zip(teams["id"], teams["short_name"]))


def print_squad(squad, season):
    """
    Print the squad with club and position names, the price and the predicted points.
    """
    # sort first, while the positions are still the numbers 1 to 4
    squad = squad.sort_values(["position", "predicted_points"], ascending=[True, False]).copy()

    squad["role"] = "bench"
    squad.loc[squad["starts"], "role"] = "start"
    squad.loc[squad["captain"], "role"] = "captain"

    squad["price"] = squad["value"] / 10
    squad["club"] = squad["club"].map(load_club_names(season))
    squad["position"] = squad["position"].map(POSITION_NAMES)

    columns = ["web_name", "club", "position", "price", "predicted_points", "role"]
    print(squad[columns].to_string(index=False))
    print(f"Price: {squad['price'].sum():.1f}m")
    print(f"Predicted points: {squad_points(squad, 'predicted_points'):.1f}")


def main():
    """
    Pick a squad for gameweek 10 of 2026-27 and show what it costs and scores.
    """
    table = predict_season("2026-27")
    players = one_row_per_player(table[table["round"] == 10])
    squad = pick_squad(players, "predicted_points")

    print_squad(squad, "2026-27")
    print(f"Real points (starting 11 + captain): {squad_points(squad, 'total_points')}")


if __name__ == "__main__":
    main()
