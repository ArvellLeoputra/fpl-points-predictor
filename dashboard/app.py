"""
Dashboard for the FPL Points Predictor.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
POSITIONS = ["GKP", "DEF", "MID", "FWD"]
COLUMNS = ["web_name", "club", "position", "price", "predicted_points"]
COLUMN_CONFIG = {
    "web_name": "Player",
    "club": "Club",
    "position": "Position",
    "price": st.column_config.NumberColumn("Price", format="£%.1fm"),
    "predicted_points": st.column_config.NumberColumn("Predicted points", format="%.1f"),
}


# Load player data
@st.cache_data(show_spinner=False)
def load_players():
    players = pd.read_csv(DATA_DIR / "next_gameweek.csv")
    players["price"] = players["value"] / 10
    return players

# Load squad data
@st.cache_data
def load_squad():
    squad = pd.read_csv(DATA_DIR / "squad.csv")
    squad["price"] = squad["value"] / 10
    squad["position"] = pd.Categorical(squad["position"], categories=POSITIONS, ordered=True)
    return squad.sort_values(["position", "predicted_points"], ascending=[True, False])

# Load backtest data
@st.cache_data
def load_backtest():
    path = Path(__file__).parent.parent / "docs" / "backtest_results.csv"
    return pd.read_csv(path, index_col="round")


players = load_players()
round_number = players["round"].iloc[0]

st.title(f"FPL Points Predictor: Gameweek {round_number}")

tab_players, tab_squad, tab_backtest = st.tabs(["Next gameweek", "Best squad", "Backtest"])

with tab_players:
    positions = st.multiselect("Position", options=POSITIONS, default=POSITIONS)
    max_price = st.slider("Max price (£m)", 3.5, 16.0, 16.0, step=0.5)

    right_position = players["position"].isin(positions)
    affordable = players["price"] <= max_price

    shown = players[right_position & affordable]

    table = shown.sort_values("predicted_points", ascending=False)
    table = table[COLUMNS]

    st.dataframe(
        table,
        hide_index=True,
        column_config=COLUMN_CONFIG
    )

with tab_squad:
    squad = load_squad()
    starters = squad[squad["starts"]]
    bench = squad[~squad["starts"]]
    captain = squad[squad["captain"]]

    total = starters["predicted_points"].sum() + captain["predicted_points"].sum()
    st.metric("Predicted points", f"{total:.1f}")
    st.caption(f"Captain: {captain['web_name'].iloc[0]} (his points count double)")

    st.subheader("Starting XI")
    st.dataframe(starters[COLUMNS], hide_index=True, column_config=COLUMN_CONFIG)

    st.subheader("Bench")
    st.dataframe(bench[COLUMNS], hide_index=True, column_config=COLUMN_CONFIG)

with tab_backtest:
    results = load_backtest()
    names = {"model": "Model", "baseline": "Form", "popular": "Most owned"}
    points = results[list(names)]
    totals = points.cumsum().rename(columns=names)

    model_total = totals["Model"].iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", int(model_total))
    col2.metric("Form", int(totals["Form"].iloc[-1]), delta=int(totals["Form"].iloc[-1] - model_total))
    col3.metric("Most owned", int(totals["Most owned"].iloc[-1]),
                delta=int(totals["Most owned"].iloc[-1] - model_total))

    st.subheader("Total points over the season")
    st.line_chart(totals)

    wins = (results["model"] > results["popular"]).sum()
    st.caption(f"The model beat the most-owned squad in {wins} of {len(results)} gameweeks.")
