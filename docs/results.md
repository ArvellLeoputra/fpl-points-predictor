# Results in detail

[← Back to the README](../README.md)

All results are for the season 2025-26. The numbers come from the scripts in `src/modeling/experiments/` and `src/optimizer/`.

## Prediction error

The models are trained on 2022-23 to 2024-25 and tested on 2025-26. RMSE counts big misses more, so it decides the best model. MAE is the average number of points the prediction is off. *Played* means the player had at least one minute, *Not played* means 0 minutes.

### RMSE

| Method | All rows | Played | Not played |
|---|---|---|---|
| Baseline: average of last 5 matches | 2.108 | 3.201 | 0.893 |
| Linear regression | 1.967 | 3.014 | 0.769 |
| Gradient-boosted trees (median) | 2.136 | 3.399 | **0.415** |
| Gradient-boosted trees (mean) | 1.958 | 2.998 | 0.773 |
| Two-part model | **1.955** | **2.997** | 0.763 |

### MAE

| Method | All rows | Played | Not played |
|---|---|---|---|
| Baseline: average of last 5 matches | 1.042 | 2.223 | 0.300 |
| Linear regression | 1.021 | 1.986 | 0.414 |
| Gradient-boosted trees (median) | **0.886** | 2.096 | **0.126** |
| Gradient-boosted trees (mean) | 0.989 | 1.970 | 0.372 |
| Two-part model | 0.981 | **1.968** | 0.361 |

- The trees minimize either the absolute error (median) or the squared error (mean).
- On players who played, the trees (mean) are 0.25 points better than the baseline (MAE 1.970 vs 2.223, about 11%). The RMSE drops from 2.108 to 1.955 (7%) with the two-part model.
- A straight formula cannot predict exactly 0 for players who do not play. Linear regression is worse than the baseline on those rows (MAE 0.414 vs 0.300), while the trees (median) predict about 0 (0.126). They win the MAE on all rows (0.886) because 61% of the rows are players who did not play, but they are the worst of the three models on players who played (2.096).
- The median of similar players is almost always about 0, so the trees (median) predict about 0 for almost everyone and miss every big score. That is why they have the worst RMSE.
- Splitting the problem into "does he play?" and "points if he plays" hardly helps (RMSE 1.955 vs 1.958 for one model). The differences between the two are within noise, which is about 0.01.

### Does the player play?

| | Share who play |
|---|---|
| Real | 38.6% |
| Rule: had minutes in the last 5 gameweeks | 51.0% |
| Classifier (all features) | 41.8% |

The classifier is right on 86.5% of the rows, the simple rule on 84.4%. The rule says "plays" too often, because a player who played earlier is not always in the squad this week.

### The linear model

| Feature | Coefficient |
|---|---|
| avg_points_last_5 | 0.225 |
| avg_minutes_last_5 | 0.024 |
| avg_starts_last_5 | -0.225 |
| was_home | 0.199 |
| opponent_strength (0 to 1) | -0.530 |
| value (tenths of £m) | 0.030 |
| position (defender, midfielder, forward) | -0.09, -0.08, -0.10 |

A player whose average is 6 points higher gets about 1.4 more predicted points, and going from 0 to 90 average minutes adds about 2.2. Playing at home adds about 0.2, and the strongest opponent instead of the weakest takes off about 0.5. The features are related to each other (average minutes and starts, for example), so single coefficients should not be over-read.

## Where the errors are

MAE per block of gameweeks in the test season (all rows):

| Gameweeks | Baseline | Two-part model |
|---|---|---|
| 1-5 | 1.167 | 1.045 |
| 6-15 | 1.058 | 1.004 |
| 16-38 | 1.016 | 0.962 |

The first gameweeks are harder, because the form restarts every season and there is little of it yet. The model helps most there (0.12 better than the baseline, against 0.05 later). The first block has only about 10% of the rows, so even a perfect fix would improve the total by less than 0.01.

## Training again every gameweek

For every gameweek the model is trained on all earlier seasons and the finished gameweeks of the test season, instead of once on the earlier seasons. Gameweek 6 and later:

| Training | RMSE | MAE | Model squads | Most-owned squads | Recent form squads |
|---|---|---|---|---|---|
| Once | 1.948 | 0.974 | 55.6 | 53.9 | 49.4 |
| Every gameweek | 1.941 | 0.976 | **58.2** | 53.9 | 49.4 |

The error hardly changes (a difference of 0.007 is noise), but the picked squads score 2.6 points more per gameweek. This is one season, so the gain in squad points is not certain.

## The squad backtest

For every gameweek from 6 to 38, the model is trained on everything played before it, and one squad (15 players, starting 11, captain) is picked with each point column. The real points of the starting 11 plus the captain (counted double) are added up. The results of every gameweek are in [backtest_results.csv](backtest_results.csv).

| Squad picked by | Real points per gameweek |
|---|---|
| Model (predicted points) | 58.2 |
| Most-owned players | 53.9 |
| Recent form | 49.4 |
| Perfect (the real points, hindsight) | 155.1 |

The perfect squad is an upper limit: the model gets 38% of its points, the most-owned players 35%, recent form 32%.

### Bench weight

The bench counts with a weight in the optimizer, because a bench player can replace a starter who does not play. The weight is a setting, not a measured value, and 0.1 is the default. Points per gameweek, and how the model compares:

| Bench weight | Model | Most-owned | Recent form | Model against recent form | Model against most-owned |
|---|---|---|---|---|---|
| 0 | 58.1 | 54.0 | 49.4 | +8.7 (+18%), ahead in 23 of 33 | +4.1 (+8%), ahead in 19 of 33 |
| 0.1 | 58.2 | 53.9 | 49.4 | +8.9 (+18%), ahead in 23 of 33 | +4.3 (+8%), ahead in 20 of 33 |
| 0.3 | 56.4 | 53.3 | 48.9 | +7.5 (+15%), ahead in 25 of 33 | +3.1 (+6%), ahead in 16 of 33 |

(Ties are not counted as wins.) A rough 95% range for the difference in points per gameweek (mean ± 1.96 standard errors of the 33 gameweeks):

| Bench weight | Against recent form | Against most-owned |
|---|---|---|
| 0 | +3.7 to +13.7 | -0.7 to +9.0 |
| 0.1 | +4.1 to +13.7 | -0.9 to +9.5 |
| 0.3 | +2.3 to +12.6 | -2.1 to +8.4 |

- **Against recent form** the result is clear and stable: the whole range is above 0 for every weight.
- **Against the most-owned players** the model is ahead on average, but the range includes 0 for every weight, and at 0.3 the model wins fewer than half of the gameweeks. With one season, the model is not clearly better than them.
- The most-owned players do well, probably because managers react to news about injuries and line-ups. The backtest cannot use that news, because the data has no record of past injuries. For the next gameweek, injured and suspended players are left out.

### Limits

- Only the 11 starters and the captain count. There are no automatic substitutions, so the test cannot say how much a good bench is worth in the real game.
- Every gameweek is a new squad with £100m. There are no transfers, no transfer costs and no change in team value.
- One test season only.
- No injury information in the backtest, so it is harder for the model than for the managers behind the most-owned squad.
- The club of a player is the one in the saved player list (the end of the season for past seasons), so it can be wrong for players who moved during a season.
- The most-owned squad uses the number of owners saved with each match, which is known before the match.

## What did not help

These were tested in earlier runs and are not part of the current code:

- **Rolling xG, xA, ICT and BPS** (average of the last 5 matches): the MAE changed by about 0.005. The form of points, minutes and starts already contains most of it.
- **Relative ownership** (owners of the player divided by the owners of the most owned player in the same gameweek): again about 0.005, and the accuracy of "does he play?" stayed at about 86.5%.

Both are within noise.
