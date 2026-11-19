import pandas as pd

from src.optimizer.squad import pick_squad


def make_players():
    """
    36 made-up players: 4 goalkeepers, 12 defenders, 12 midfielders, 8 forwards,
    spread over 10 clubs. Everyone costs £5.0m.
    """
    rows = []
    counts = {1: 4, 2: 12, 3: 12, 4: 8}  # goalkeeper, defender, midfielder, forward
    i = 0
    for position, n in counts.items():
        for k in range(n):
            rows.append({
                "web_name": f"Player {i}",
                "club": i % 10,
                "position":position,
                "value": 50,
                "predicted_points": float(k + 1),
            })
            i += 1
    return pd.DataFrame(rows)


def pick(players):
    return pick_squad(players, "predicted_points")


def test_squad_has_15_players():
    squad = pick(make_players())
    assert len(squad) == 15


def test_squad_has_right_composition():
    squad = pick(make_players())
    counts = squad["position"].value_counts().to_dict()
    assert counts == {1: 2, 2: 5, 3: 5, 4: 3}  # 2 goalkeepers, 5 defenders, 5 midfielders, 3 forwards


def test_11_starters_and_1_captain():
    squad = pick(make_players())
    assert squad["starts"].sum() == 11
    assert squad["captain"].sum() == 1


def test_captain_is_a_starter():
    squad = pick(make_players())
    captain = squad[squad["captain"]]
    assert captain["starts"].all()


def test_at_most_3_players_per_club():
    # 4 midfielders from club 0 are by far the best, so the optimizer would like all 4
    players = make_players()
    stars = players.index[players["position"] == 3][:4]
    players.loc[stars, "club"] = 0
    players.loc[stars, "predicted_points"] = 100.0

    squad = pick(players)

    assert squad.index.isin(stars).sum() == 3
    assert squad["club"].value_counts().max() <= 3


def test_budget_is_respected():
    # 8 great but expensive players (£15m each, 5 midfielders and 3 forwards,
    # all from different clubs). They fit the position rules, but all 8 would
    # cost £120m, more than the £100m budget.
    players = make_players()
    midfielders = players.index[players["position"] == 3][:5]
    forwards = players.index[players["position"] == 4][:3]
    stars = midfielders.append(forwards)
    players.loc[stars, "club"] = range(8)
    players.loc[stars, "value"] = 150
    players.loc[stars, "predicted_points"] = 100.0

    squad = pick(players)

    assert squad["value"].sum() <= 1000
    assert squad.index.isin(stars).sum() < 8
