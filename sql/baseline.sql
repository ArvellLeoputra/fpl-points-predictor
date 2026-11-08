.headers on
.mode column

/* Baseline error: mean absolute error (MAE) of "average of last 5 matches".
   The first match of every player has no prediction, so it is skipped. */

SELECT
    ROUND(AVG(ABS(total_points - avg_last_5)), 3) AS mae,
    COUNT(*) AS n_predictions
FROM (
    SELECT
        gs.total_points,
        gs.minutes,
        AVG(gs.total_points) OVER (
            PARTITION BY gs.element
            ORDER BY gs.kickoff_time
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS avg_last_5
    FROM gameweek_stats AS gs
)
WHERE avg_last_5 IS NOT NULL
    AND minutes > 0;