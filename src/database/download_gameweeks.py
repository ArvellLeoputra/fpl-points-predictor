"""
Download FPL players, teams, positions and per-gameweek history from the official API.
"""

import argparse
import logging
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import ROOT

BASE_URL = "https://fantasy.premierleague.com/api"
OUT_DIR = ROOT / "data" / "raw"
REQUEST_DELAY = 0.1
TIMEOUT = 30

log = logging.getLogger(__name__)


def make_session():
    """
    Create a session that retries on rate limits and server errors.
    """
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504)
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "fpl-points-predictor/0.1"})
    return session


def get_json(session, endpoint):
    """
    Ask the API for one endpoint and return the data.
    """
    response = session.get(f"{BASE_URL}/{endpoint}/", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_bootstrap(session):
    """
    Download the main FPL overview.
    """
    return get_json(session, "bootstrap-static")


def build_dataframe(data, section):
    """
    Return the given section of the overview (for example "elements",
    "teams" or "element_types") as a DataFrame with one row per record.
    """
    return pd.DataFrame(data[section])


def fetch_histories(session, player_ids):
    """
    Get the gameweek history of every player.
    Returns the combined table and a list of ids that failed.
    """
    frames = []  # list of DataFrames, one per player
    failed = []  # list of player ids that failed to download

    for i, pid in enumerate(player_ids, start=1):
        try:
            data = get_json(session, f"element-summary/{pid}")
            rows = data["history"]
            if rows:
                frames.append(pd.DataFrame(rows))
        except requests.RequestException as err:
            log.warning("Player %s failed: %s", pid, err)
            failed.append(pid)

        if i % 50 == 0:
            log.info("%d / %d players done", i, len(player_ids))
        time.sleep(REQUEST_DELAY)

    if not frames:
        raise RuntimeError("No gameweek history was downloaded.")

    history = pd.concat(frames, ignore_index=True)
    return history, failed


def main(limit=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = make_session()
    data = fetch_bootstrap(session)

    players = build_dataframe(data, "elements")
    teams = build_dataframe(data, "teams")
    positions = build_dataframe(data, "element_types")

    player_ids = players["id"].tolist()[:limit]

    history, failed = fetch_histories(session, player_ids)

    players.to_csv(OUT_DIR / "players.csv", index=False)
    history.to_csv(OUT_DIR / "gameweek_stats.csv", index=False)
    teams.to_csv(OUT_DIR / "teams.csv", index=False)
    positions.to_csv(OUT_DIR / "positions.csv", index=False)

    log.info("History shape: %s", history.shape)
    log.info("Players with history: %d", history["element"].nunique())
    if failed:
        log.warning("Failed player ids: %s", failed)
    log.info("Teams shape: %s", teams.shape)
    log.info("Positions shape: %s", positions.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FPL data.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only download the first N players (for testing)"
    )
    args = parser.parse_args()
    main(limit=args.limit)
