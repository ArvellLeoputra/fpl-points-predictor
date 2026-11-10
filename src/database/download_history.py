"""
Download past seasons (2022-23 to 2025-26) of FPL data from the public vaastav dataset on GitHub.
"""

import logging
from pathlib import Path

import requests

from src.config import ROOT, PAST_SEASONS

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
FILES = ["gws/merged_gw.csv", "players_raw.csv", "teams.csv"]
OUT_DIR = ROOT / "data" / "raw" / "history"
TIMEOUT = 30

log = logging.getLogger(__name__)


def download_file(season, file):
    """
    Download one file of one season and save it under data/raw/history/<season>/.
    """
    url = f"{BASE_URL}/{season}/{file}"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()

    path = OUT_DIR / season / Path(file).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    log.info("Saved %s", path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for season in PAST_SEASONS:
        for file in FILES:
            download_file(season, file)


if __name__ == "__main__":
    main()
