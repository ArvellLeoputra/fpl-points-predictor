# FPL Points Predictor

The goal is to predict Fantasy Premier League (FPL) player points for the next match. So far, the project builds a SQLite database from the official FPL API, explores it with SQL, and compares machine learning models with a simple baseline on past gameweeks.

## Results

Test set: gameweeks 8 and 9 of the 2026/27 season (1,337 player-matches). Models are trained on gameweeks 1 to 7 only. Numbers are mean absolute error (MAE) in FPL points.

| Method | All rows | Played only | Not played |
|---|---|---|---|
| Baseline: average of last 5 matches | 1.185 | 2.097 | 0.413 |
| Linear regression | 1.185 | **1.914** | 0.567 |
| Gradient-boosted trees (median) | **1.108** | 2.103 | **0.265** |
| Gradient-boosted trees (mean) | 1.261 | 2.076 | 0.571 |

Median / mean: the trees are trained to minimize the absolute error (median) or the squared error (mean). MAE = average number of points the prediction is off.

Columns: *All rows* is every player-match in the test set. *Played only* is the rows where the player had at least one minute. *Not played* is the rows where he had 0 minutes.

## Short takeaways

- Minutes and price explain more than recent form. In the linear model, going from 0 to 90 average minutes adds about 2.8 predicted points, while a player whose average is 6 points higher gets only about 0.6 more.
- A straight formula cannot predict exactly 0 for players who do not play. Linear regression beats the baseline on players who played (1.914 vs 2.097), but is worse on players who did not play (0.567 vs 0.413). The trees can predict about 0 for players who are not expected to play, and they are the best on those rows (0.265 vs 0.413).
- On players who played, the best model (linear regression, 1.914) improves the baseline by only 0.18 points. Position and opponent strength are the next features.
- Trees trained for the mean have a higher MAE than trees trained for the median (1.261 vs 1.108). The mean is the expected number of points, which a squad optimizer needs, so MAE is not the right measure for this version.

Only 9 gameweeks of one season are available so far, the test set is two gameweeks, and the models are not tuned yet, so these results are preliminary.

## How it works

1. **Download** players, teams, positions and the match history of every player from the official FPL API (`src/download_gameweeks.py`).
2. **Database**: build a SQLite database from `sql/schema.sql` and the CSV files, with primary and foreign keys and a consistency check (`src/build_db.py`).
3. **SQL exploration**: value for money, rolling form with a window function, baseline error (`sql/*.sql`). Run one with:

   ```bash
   sqlite3 data/fpl.db < sql/value_for_money.sql
   ```

4. **Features** (`src/features.py`): rolling averages of points and minutes over the previous 5 matches, home or away, and price.
5. **Models** (`src/train.py`): linear regression and gradient-boosted trees, compared with the baseline.

Design choices:

- **Only information from before the match.** Features use only what is known before kickoff: rolling averages of the player's earlier matches (the current match is excluded), home or away, and the price in that match.
- **Time-based split.** Train on earlier gameweeks, test on later ones. A random split would let the model learn from the future.
- **Only finished matches.** Matches that were not finished at download time have placeholder values and are removed. Gameweek 10 was not finished, which is why the test set stops at gameweek 9.
- **Fair evaluation.** Every method is scored on the same rows, split into players who played and players who did not. More than half of the rows are players who did not play, and they are easy to predict, which makes the overall MAE look better than it is.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run from the repository root:

```bash
python src/download_gameweeks.py   # download raw data into data/raw/
python src/build_db.py             # build data/fpl.db
python src/train.py                # train models and print the results table
```

Quick test with only 5 players:

```bash
python src/download_gameweeks.py --limit 5
```

The database and raw data are not stored in this repository. They are rebuilt by the first two commands. Re-run them after each gameweek to update the data.

## Project structure

```
data/
  raw/                    CSV files from the API (created by the scripts)
  fpl.db                  SQLite database (created by build_db.py)
src/
  download_gameweeks.py   download from the FPL API
  build_db.py             schema + CSV files -> SQLite
  config.py               paths, time split, feature settings
  features.py             features, train/test split
  train.py                models and comparison with the baseline
sql/
  schema.sql              table definitions
  value_for_money.sql     points per million
  rolling_form.sql        average of last 5 matches (window function)
  baseline.sql            baseline error
```

## Next steps

- Position and opponent strength features
- A minutes model: will the player start?
- More seasons of data
- Squad optimizer (mixed-integer program) and a backtest
- Automated weekly refresh

## Data source

Official FPL API (`fantasy.premierleague.com/api`).
