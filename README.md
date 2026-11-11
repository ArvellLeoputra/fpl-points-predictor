# FPL Points Predictor

Predict Fantasy Premier League (FPL) player points for the next match. The project builds a SQLite database from the official FPL API and an archive of past seasons, explores it with SQL, and compares machine learning models, including a two-part model, with a simple baseline.

## Results

The models are trained on the seasons 2022-23 to 2024-25 and tested on the full season 2025-26 (28,906 player-matches).

Two measures: RMSE counts big misses more, so a model has to predict the expected points of a player to score well. I choose the best model by RMSE. MAE is the average number of FPL points the prediction is off. *Played* means the player had at least one minute, *Not played* means 0 minutes.

### RMSE: decides the best model

| Method | All rows | Played | Not played |
|---|---|---|---|
| Baseline: average of last 5 matches | 2.108 | 3.201 | 0.893 |
| Linear regression | 1.967 | 3.014 | 0.769 |
| Gradient-boosted trees (median) | 2.136 | 3.399 | **0.415** |
| Gradient-boosted trees (mean) | 1.958 | 2.998 | 0.773 |
| Two-part model | **1.955** | **2.997** | 0.763 |

### MAE: the average miss in points

| Method | All rows | Played | Not played |
|---|---|---|---|
| Baseline: average of last 5 matches | 1.042 | 2.223 | 0.300 |
| Linear regression | 1.021 | 1.986 | 0.414 |
| Gradient-boosted trees (median) | **0.886** | 2.096 | **0.126** |
| Gradient-boosted trees (mean) | 0.989 | 1.970 | 0.372 |
| Two-part model | 0.981 | **1.968** | 0.361 |

- The trees minimize either the absolute error (median) or the squared error (mean).
- The two-part model is the probability that the player plays × the expected points if he plays (both gradient-boosted trees, mean). It is within 0.003 of the single trees (mean) on RMSE, so it does not clearly beat it.
- Trees (median) win the MAE on all rows, but have the worst RMSE, even worse than the baseline. The median of similar players is almost always about 0, so they predict about 0 for almost everyone and miss every big score.

## Short takeaways

- The models clearly beat the baseline. On players who played, the trees (mean) are 0.25 points better (1.970 vs 2.223, about 11%). The RMSE drops from 2.108 to 1.955 (7%) with the two-part model.
- A straight formula cannot predict exactly 0 for players who do not play. Linear regression is worse than the baseline on those rows (MAE 0.414 vs 0.300), while the trees (median) predict about 0 (0.126). They win the MAE on all rows (0.886) because 61% of the rows are players who did not play, but they are the weakest of the models on players who played (2.096).
- Splitting the problem into "does he play?" and "points if he plays" does not help much (RMSE 1.955 vs 1.958 for one model). The classifier is right on 86.5% of the rows, a simple rule ("had minutes in the last 5 gameweeks") on 84.4%.
- In the linear model, a player whose average is 6 points higher gets about 1.4 more predicted points, and going from 0 to 90 average minutes adds about 2.2. Playing at home adds about 0.2, and the strongest opponent instead of the weakest takes off about 0.5. Position adds little once price and minutes are known.

These results are preliminary: one test season, no tuning yet, and differences of about 0.01 or less are noise (a different random seed moved some numbers by up to 0.015).

## Where the errors are

The error per block of gameweeks in the test season (MAE, all rows):

| Gameweeks | Rows | Baseline | Two-part model |
|---|---|---|---|
| 1-5 | 2,847 | 1.167 | 1.045 |
| 6-15 | 7,484 | 1.058 | 1.004 |
| 16-38 | 18,575 | 1.016 | 0.962 |

Here the MAE is used for both methods, so the numbers can be compared with the MAE columns in the table above.

The first gameweeks are harder, because the form restarts every season and there is little of it yet. The model helps most there (0.12 better than the baseline, against 0.05 later). The first block has only 10% of the rows, so even a perfect fix would improve the total by less than 0.01.

## What did not help

- **Rolling xG, xA, ICT and BPS** (average of the last 5 matches): the MAE changed by about 0.005. The form of points, minutes and starts already contains most of it.
- **Relative ownership** (owners of the player divided by the owners of the most owned player in the same gameweek): again about 0.005, and the "does he play?" accuracy stayed at about 86.5%.

Both are within the noise, so they are not in the model.

## How it works

1. **Download**: players, teams, positions and the match history of the current season from the official FPL API (`src/database/download_gameweeks.py`), and the seasons 2022-23 to 2025-26 from a public archive of that API (`src/database/download_history.py`).
2. **Database**: build a SQLite database from `sql/schema.sql` and the CSV files. Every table has a `season` column, with primary and foreign keys and a consistency check (`src/database/build_db.py`). Managers (a separate type in 2024-25) and duplicated matches are removed.
3. **SQL exploration**: value for money, rolling form with a window function, baseline error (`sql/*.sql`). Run one with:

   ```bash
   sqlite3 data/fpl.db < sql/value_for_money.sql
   ```

4. **Features** (`src/modeling/features.py`): rolling averages of points, minutes and starts over the previous 5 matches of the same season, position, home or away, opponent strength and price.
5. **Models** (`models.py`) and **scoring** (`evaluate.py`), used by four scripts, all in `src/modeling/experiments/`:
   - `compare_models.py`: linear regression and trees against the baseline, by MAE and RMSE
   - `minutes_model.py`: does the player play? rule against classifier
   - `two_part_model.py`: P(plays) × points if he plays, against one model
   - `error_by_round.py`: error per block of gameweeks

Design choices:

- **Only information from before the match.** The form uses the player's earlier matches only (the current match is excluded). Price, home or away and the opponent are known before kickoff.
- **One person, one id across seasons.** FPL player ids change every season, so players are identified by the `code`.
- **The form restarts every season.** The form uses only the matches of the same season. A match from months ago says little about the player now. A player's first match of a season has no form yet, so these rows are dropped (2,427 training rows, 841 test rows).
- **Opponent strength on the same scale.** The ratings in the archive run from about 975 to 1370, in the current API from 2 to 5. Each season is rescaled from 0 (weakest opponent) to 1 (strongest).
- **Time-based split.** Train on earlier seasons, test on the next one. A random split would let the model learn from the future.
- **Only finished matches.** Matches that were not finished at download time have placeholder values and are removed.
- **Fair evaluation.** Every method is scored on the same rows, and also on the players who played alone. Most rows are players who did not play. They are easy to predict, which makes the overall error look better than it is.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run from the repository root:

```bash
python -m src.database.download_history                 # past seasons into data/raw/history/ (only once)
python -m src.database.download_gameweeks               # current season into data/raw/
python -m src.database.build_db                         # build data/fpl.db
python -m src.modeling.experiments.compare_models       # results table of the models (MAE and RMSE)
python -m src.modeling.experiments.minutes_model        # does the player play?
python -m src.modeling.experiments.two_part_model       # two-part model, compared by MAE and RMSE
python -m src.modeling.experiments.error_by_round       # error per block of gameweeks
```

Quick test with only 5 players:

```bash
python -m src.database.download_gameweeks --limit 5
```

The database and raw data are not stored in this repository. They are rebuilt by the first three commands. Re-run them after each gameweek to update the data.

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
    experiments/
      compare_models.py     linear regression and trees vs baseline
      minutes_model.py      does the player play?
      two_part_model.py     two-part model vs one model
      error_by_round.py     error per block of gameweeks
sql/
  schema.sql                table definitions
  value_for_money.sql       points per million
  rolling_form.sql          average of last 5 matches
  baseline.sql              baseline error
```

## Next steps

- Squad optimizer (mixed-integer program) and a backtest on the season 2025-26
- Better information about who plays (injuries, rotation); ownership did not help
- More seasons (2020-21 and 2021-22 have no `starts` column) and model tuning
- Automated weekly refresh

## Data source

Current season: official FPL API (`fantasy.premierleague.com/api`).

Past seasons: the public archive of that API by vaastav, [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League).
