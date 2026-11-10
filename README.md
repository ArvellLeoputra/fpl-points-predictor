# FPL Points Predictor

Predict Fantasy Premier League (FPL) player points for the next match. The project builds a SQLite database from the official FPL API and an archive of past seasons, explores it with SQL, and compares machine learning models, including a two-part model, with a simple baseline.

## Results

The models are trained on the seasons 2022-23 to 2024-25 and tested on the full season 2025-26 (29,500 player-matches). The numbers are the mean absolute error (MAE), the average number of FPL points the prediction is off. *Played* means the player had at least one minute, *Not played* means 0 minutes.

| Method | All rows | Played | Not played |
|---|---|---|---|
| Baseline: average of last 5 matches | 1.053 | 2.214 | 0.320 |
| Linear regression | 1.033 | 1.990 | 0.429 |
| Gradient-boosted trees (median) | **0.907** | 2.161 | **0.116** |
| Gradient-boosted trees (mean) | 1.003 | **1.962** | 0.398 |

The trees minimize either the absolute error (median) or the squared error (mean).

### Two-part model

Expected points = probability that the player plays × expected points if he plays (both gradient-boosted trees). Since it predicts the mean, it is compared by RMSE (root mean squared error) and not by MAE.

| Method | All rows | Played | Not played |
|---|---|---|---|
| Baseline: average of last 5 matches | 2.107 | 3.184 | 0.920 |
| Single model: gradient-boosted trees (mean) | 1.970 | 3.008 | 0.785 |
| Two-part model | 1.972 | 3.010 | 0.789 |

Both models beat the baseline. The two-part model does not beat the single model: the two are within 0.004 of each other.

## Short takeaways

- With three seasons of training data, the models clearly beat the baseline. On players who played, the best model (trees, mean) is 0.25 points better (1.962 vs 2.214, about 11%), and the RMSE drops from 2.107 to 1.970.
- A straight formula cannot predict exactly 0 for players who do not play. Linear regression is worse than the baseline on those rows (0.429 vs 0.320), while the trees (median) predict about 0 and win by far (0.116). They also win on all rows (0.907), because 61% of the rows are players who did not play, but they are the weakest tree on players who played (2.161).
- Splitting the problem into "does he play?" and "points if he plays" does not help (RMSE 1.972 vs 1.970 for one model). The gain comes from the data and the features. The "does he play?" classifier is right on 85.9% of the rows; a simple rule ("had minutes in the last 5 gameweeks") gets 83.5%.
- In the linear model, a player whose average is 6 points higher gets about 1.4 more predicted points, and going from 0 to 90 average minutes adds about 1.9. Playing at home adds about 0.2, and the strongest opponent instead of the weakest takes off about 0.5. Position adds little once price and minutes are known.

These results are preliminary: one test season, no tuning yet, and differences of about 0.01 or less are noise (a different random seed moved some numbers by up to 0.015).

## How it works

1. **Download**: players, teams, positions and the match history of the current season from the official FPL API (`src/database/download_gameweeks.py`), and the seasons 2022-23 to 2025-26 from a public archive of that API (`src/database/download_history.py`).
2. **Database**: build a SQLite database from `sql/schema.sql` and the CSV files. Every table has a `season` column, with primary and foreign keys and a consistency check (`src/database/build_db.py`). Managers (a separate type in 2024-25) and duplicated matches are removed.
3. **SQL exploration**: value for money, rolling form with a window function, baseline error (`sql/*.sql`). Run one with:

   ```bash
   sqlite3 data/fpl.db < sql/value_for_money.sql
   ```

4. **Features** (`src/modeling/features.py`): rolling averages of points, minutes and starts over the previous 5 matches, position, home or away, opponent strength and price.
5. **Models** (`models.py`) and **scoring** (`evaluate.py`), used by three experiments, all in `src/modeling/`:
   - `compare_models.py`: linear regression and trees against the baseline
   - `minutes_model.py`: does the player play? rule against classifier
   - `two_part_model.py`: P(plays) × points if he plays, against one model

Design choices:

- **Only information from before the match.** The form uses the player's earlier matches only (the current match is excluded). Price, home or away and the opponent are known before kickoff.
- **One person, one id across seasons.** FPL player ids change every season, so the form is grouped by the player `code`. The form of the last matches of one season carries over into the next.
- **Opponent strength on the same scale.** The ratings in the archive run from about 1000 to 1370, in the current API from 2 to 5. Each season is rescaled from 0 (weakest opponent) to 1 (strongest).
- **Time-based split.** Train on earlier seasons, test on the next one. A random split would let the model learn from the future.
- **Only finished matches.** Matches that were not finished at download time have placeholder values and are removed.
- **Fair evaluation.** Every method is scored on the same rows, split into players who played and players who did not. Most rows are players who did not play. They are easy to predict, which makes the overall MAE look better than it is.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run from the repository root:

```bash
python -m src.database.download_history     # past seasons into data/raw/history/ (only once)
python -m src.database.download_gameweeks   # current season into data/raw/
python -m src.database.build_db             # build data/fpl.db
python -m src.modeling.compare_models       # results table of the models
python -m src.modeling.minutes_model        # does the player play?
python -m src.modeling.two_part_model       # two-part model, compared by MAE and RMSE
```

Quick test with only 5 players:

```bash
python -m src.database.download_gameweeks --limit 5
```

The database and raw data are not stored in this repository. They are rebuilt by the first two commands. Re-run them after each gameweek to update the data.

## Project structure

```
data/
  raw/                      CSV files from the API and the archive
  fpl.db                    SQLite database
src/
  config.py                 paths, seasons, feature settings
  database/
    download_gameweeks.py   current season from the FPL API
    download_history.py     past seasons from the archive
    build_db.py             schema + CSV files -> SQLite
  modeling/
    features.py             features and train/test split
    models.py               baseline and model training
    evaluate.py             error measures and results table
    compare_models.py       linear regression and trees vs baseline
    minutes_model.py        does the player play?
    two_part_model.py       two-part model vs one model
sql/
  schema.sql                table definitions
  value_for_money.sql       points per million
  rolling_form.sql          average of last 5 matches
  baseline.sql              baseline error
```

## Next steps

- Squad optimizer (mixed-integer program) and a backtest on the season 2025-26
- Better information about who plays (injuries, rotation)
- More seasons (2020-21 and 2021-22 have no `starts` column) and model tuning
- Automated weekly refresh

## Data source

Current season: official FPL API (`fantasy.premierleague.com/api`).

Past seasons: the public archive of that API by vaastav, [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League).
