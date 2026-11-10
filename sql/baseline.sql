.headers on
.mode column

/* Baseline error: mean absolute error (MAE) of "average of last 5 matches"
   in the same season, for the season 2025-26 and players who played.
   The first match of every player in a season has no prediction, so it is skipped. */

SELECT
    ROUND(AVG(ABS(total_points - avg_last_5)), 3) AS mae,
    COUNT(*) AS n_predictions
FROM (
    SELECT
        gs.total_points,
        gs.minutes,
        AVG(gs.total_points) OVER (
            PARTITION BY p.code, gs.season
            ORDER BY gs.kickoff_time
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS avg_last_5
    FROM gameweek_stats AS gs
    JOIN players AS p ON p.season = gs.season AND p.id = gs.element
    WHERE gs.team_h_score IS NOT NULL  -- keeping only finished matches
        AND gs.season = '2025-26'
)
WHERE avg_last_5 IS NOT NULL
    AND minutes > 0;
