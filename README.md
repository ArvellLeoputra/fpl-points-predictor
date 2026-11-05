# FPL Points Predictor

Predict Fantasy Premier League player points for the next gameweek,
using a SQL database of FPL data and machine learning.

**Status:** work in progress. Currently the data download is done.

## Planned pipeline

1. Download player and gameweek data from the official FPL API
2. Store it in a SQLite database and explore it with SQL
3. Build features and a baseline ("average of last 5 gameweeks")
4. Train models and compare them against the baseline
5. Later: minutes model and squad optimizer

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Download all players and their gameweek history into `data/raw/`:

```bash
python src/download_gameweeks.py
```

Quick test with only 5 players:

```bash
python src/download_gameweeks.py --limit 5
```

## Data source

Official FPL API (`fantasy.premierleague.com/api`). Raw data is not stored
in this repo, because the script downloads it.