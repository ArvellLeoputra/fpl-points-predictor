# FPL Points Predictor

The goal is to predict Fantasy Premier League (FPL) player points for the next match. So far, the project builds a SQLite database from the official FPL API, explores it with SQL, and compares machine learning models, including a two-part model, with a simple baseline on past gameweeks.

## Results

Test set: gameweeks 8 and 9 of the 2026/27 season (1,337 player-matches). Models are trained on gameweeks 1 to 7 only. The numbers are the mean absolute error (MAE), the average number of FPL points the prediction is off. *Played* means the player had at least one minute, *Not played* means 0 minutes.

| Method | All rows | Played | Not played |
|---|---|---|---|
| Baseline: average of last 5 matches | 1.185 | 2.097 | 0.413 |
| Linear regression | 1.185 | **1.894** | 0.584 |
| Gradient-boosted trees (median) | **1.081** | 2.074 | **0.240** |
| Gradient-boosted trees (mean) | 1.237 | 2.062 | 0.539 |

The trees minimize either the absolute error (median) or the squared error (mean).

Columns: *All rows* is every player-match in the test set. *Played only* is the rows where the player had at least one minute. *Not played* is the rows where he had 0 minutes.

### Two-part model

Expected points = probability that the player plays × expected points if he plays (both gradient-boosted trees). Since it predicts the mean, it is compared by RMSE (root mean squared error) and not by MAE.

| Method | All rows | Played | Not played |
|---|---|---|---|
| Baseline: average of last 5 matches | 2.142 | 2.958 | **1.030** |
| Single model: gradient-boosted trees (mean) | 2.157 | 2.959 | 1.086 |
| Two-part model | 2.145 | **2.927** | 1.116 |

On all rows the three are within 0.02 of each other, which is within the noise of a two-gameweek test set. The two-part model is slightly best on players who played and slightly worse on players who did not.

## Short takeaways

- Minutes and price matter more than recent form. In the linear model, going from 0 to 90 average minutes adds about 2.9 predicted points. A player whose average is 6 points higher gets only about 0.6 more.
- A straight formula cannot predict exactly 0 for players who do not play. Linear regression beats the baseline on players who played (1.894 vs 2.097), but is worse on players who did not play (0.584 vs 0.413). The trees can predict about 0 for players who are not expected to play, and they are the best on those rows (0.240 vs 0.413).
- On players who played, the best model (linear regression, 1.894) improves the baseline by only 0.20 points. Adding position and opponent strength changed the results by about 0.01 to 0.02, which is within the noise of a two-gameweek test set. Position adds little once price and minutes are known, because price already reflects the position. A stronger opponent lowers the predicted points by about 0.2 per rating step.
- Trees trained for the mean have a higher MAE than trees trained for the median (1.237 vs 1.081). The mean is the expected number of points, which a squad optimizer needs, so these models are judged by RMSE.

These results are preliminary: only 9 gameweeks of one season, a two-gameweek test set, and no tuning yet.

## How it works

1. **Download** players, teams, positions and the match history of every player from the official FPL API (`src/download_gameweeks.py`).
2. **Database**: build a SQLite database from `sql/schema.sql` and the CSV files, with primary and foreign keys and a consistency check (`src/build_db.py`).
3. **SQL exploration**: value for money, rolling form with a window function, baseline error (`sql/*.sql`). Run one with:

   ```bash
   sqlite3 data/fpl.db < sql/value_for_money.sql
   ```

4. **Features** (`src/features.py`): rolling averages of points, minutes and starts over the previous 5 matches, position, home or away, opponent strength, and price.
5. **Models** (`src/models.py`) and **scoring** (`src/evaluate.py`), used by three experiments:
   - `compare_models.py`: linear regression and trees against the baseline
   - `minutes_model.py`: does the player play? rule against classifier
   - `two_part_model.py`: P(plays) × points if he plays, against one model

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
python src/compare_models.py       # results table of the models
python src/minutes_model.py        # does the player play?
python src/two_part_model.py       # two-part model, compared by RMSE
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
  features.py             features and train/test split
  models.py               baseline and model training
  evaluate.py             error measures and results table
  compare_models.py       linear regression and trees vs baseline
  minutes_model.py        does the player play?
  two_part_model.py       two-part model vs one model
sql/
  schema.sql              table definitions
  value_for_money.sql     points per million
  rolling_form.sql        average of last 5 matches
  baseline.sql            baseline error
```

## Next steps

- More seasons of data (the test set is too small to tell the models apart)
- Better information about who plays (injuries, rotation)
- Squad optimizer (mixed-integer program) and a backtest
- Automated weekly refresh

## Data source

Official FPL API (`fantasy.premierleague.com/api`).
