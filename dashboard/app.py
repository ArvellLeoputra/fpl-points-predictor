"""
Dashboard for the FPL Points Predictor.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
BACKTEST_PATH = HERE.parent / "docs" / "backtest_results.csv"
REPO_URL = "https://github.com/ArvellLeoputra/fpl-points-predictor"

POSITIONS = ["GKP", "DEF", "MID", "FWD"]
GREEN = "#1a9e4b"
GREY = "#9aa3ad"
GOLD = "#d4a72c"

st.set_page_config(page_title="FPL Points Predictor", page_icon="⚽", layout="wide")
css = (HERE / "style.css").read_text()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_players():
    players = pd.read_csv(DATA_DIR / "next_gameweek.csv")
    players["price"] = players["value"] / 10
    return players


@st.cache_data(show_spinner=False)
def load_squad():
    squad = pd.read_csv(DATA_DIR / "squad.csv")
    squad["price"] = squad["value"] / 10
    return squad


@st.cache_data(show_spinner=False)
def load_backtest():
    return pd.read_csv(BACKTEST_PATH, index_col="round")


def card_detail(player, show):
    # The highlighted last line of a card: points, fixture or ownership
    if show == "Fixture":
        return player.fixture
    if show == "Ownership":
        return f"{player.ownership:.1f}% owned"
    return f"{player.predicted_points:.1f} pts"


def player_card(player, show, captain=False):
    # One player on the pitch: name, club, price and one chosen detail
    captain_badge = '<span class="captain">C</span>' if captain else ""

    name = f'<div class="card-name">{player.web_name}{captain_badge}</div>'
    info = f'<div class="card-info">{player.club} &middot; £{player.price:.1f}m</div>'
    detail = f'<div class="card-detail">{card_detail(player, show)}</div>'

    return f'<div class="card">{name}{info}{detail}</div>'


def pitch(starters, show):
    # The starting eleven on a pitch, one row per position, goalkeeper at the top
    rows = []
    for position in POSITIONS:
        line = starters[starters["position"] == position].sort_values("predicted_points", ascending=False)

        cards = ""
        for p in line.itertuples():
            cards += player_card(p, show, captain=p.captain)

        # Put this position's cards side by side in one centered row
        rows.append(f'<div class="pitch-row">{cards}</div>')

    return f'<div class="pitch">{"".join(rows)}</div>'


players = load_players()
squad = load_squad()
results = load_backtest()

round_number = players["round"].iloc[0]
updated_on = pd.to_datetime(players["updated"].iloc[0])
updated = f"{updated_on.day} {updated_on:%b %Y}"

starters = squad[squad["starts"]]
bench = squad[~squad["starts"]]
captain = squad[squad["captain"]].iloc[0]
squad_points = starters["predicted_points"].sum() + captain["predicted_points"]

# Sidebar
with st.sidebar:
    st.title("FPL Points Predictor")
    st.markdown(
        "A model predicts every player's points for the next gameweek. "
        "An optimizer then picks the best 15-player squad within the £100m budget."
    )
    st.caption(f"Data updated twice a week. [Code on GitHub]({REPO_URL})")

# Header and key numbers
st.header(f"Gameweek {round_number}")
st.caption(f"Predictions for the next gameweek, based on FPL data from {updated}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Squad predicted points", f"{squad_points:.1f}", border=True)
col2.metric("Captain", captain["web_name"], border=True)
col3.metric("Squad cost", f"£{squad['price'].sum():.1f}m", border=True)
col4.metric("Available players", len(players), border=True)

tab_squad, tab_players, tab_backtest = st.tabs(["Best squad", "All players", "Backtest"])

with tab_squad:
    show = st.segmented_control("Show", ["Points", "Fixture", "Ownership"], default="Points", required=True)

    pitch_col, bench_col = st.columns([3, 1])
    with pitch_col:
        st.markdown(pitch(starters, show), unsafe_allow_html=True)
    with bench_col:
        st.subheader("Bench")
        sorted_bench = bench.sort_values("position", key=lambda col: col.map(POSITIONS.index))

        cards = ""
        for p in sorted_bench.itertuples():
            cards += player_card(p, show)

        st.markdown(f'<div class="bench">{cards}</div>', unsafe_allow_html=True)
        st.caption("The captain's points count double.")

with tab_players:
    search_col, filter_col, slider_col = st.columns([2, 2, 1])
    search = search_col.text_input("Search player", placeholder="e.g. Haaland")
    positions = filter_col.pills("Position", POSITIONS, selection_mode="multi", default=POSITIONS)
    max_price = slider_col.slider("Max price (£m)", 3.5, 16.0, 16.0, step=0.5)

    matches_search = players["web_name"].str.contains(search, case=False, regex=False)
    right_position = players["position"].isin(positions)
    affordable = players["price"] <= max_price
    in_squad = players["code"].isin(squad["code"])

    table = players.assign(in_squad=in_squad)[matches_search & right_position & affordable]
    table = table.sort_values("predicted_points", ascending=False)

    st.dataframe(
        table[["in_squad", "web_name", "club", "position", "price", "fixture", "ownership", "predicted_points"]],
        hide_index=True,
        width="stretch",
        height=560,
        column_config={
            "in_squad": st.column_config.CheckboxColumn("In squad", width="small"),
            "web_name": "Player",
            "club": "Club",
            "position": "Position",
            "price": st.column_config.NumberColumn("Price", format="£%.1fm"),
            "fixture": "Next match",
            "ownership": st.column_config.NumberColumn("Owned by", format="%.1f%%"),
            "predicted_points": st.column_config.ProgressColumn(
                "Predicted points", format="%.1f", min_value=0,
                max_value=float(players["predicted_points"].max()),
            ),
        },
    )

with tab_backtest:
    st.caption("A new squad every gameweek of the 2025-26 season, starting 11 plus captain")

    names = {"model": "Model", "baseline": "Form", "popular": "Most owned"}
    points = results[["model", "baseline", "popular"]]
    totals = points.cumsum().rename(columns=names)
    final = totals.iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", int(final["Model"]), border=True)
    col2.metric("Picking by form", int(final["Form"]),
                delta=f"{int(final['Form'] - final['Model'])} vs model", delta_color="off", border=True)
    col3.metric("Most-owned squad", int(final["Most owned"]),
                delta=f"{int(final['Most owned'] - final['Model'])} vs model", delta_color="off", border=True)

    st.subheader("Total points over the season")
    st.line_chart(totals, color=[GREEN, GREY, GOLD], height=380)

    wins = (results["model"] > results["popular"]).sum()
    st.caption(f"The model beat the most-owned squad in {wins} of {len(results)} gameweeks.")
