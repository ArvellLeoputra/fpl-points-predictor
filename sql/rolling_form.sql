.headers on
.mode column

/* Rolling form: for each match, the average points of the player's
   previous 5 matches in the same season. The current match is excluded
   (no leakage). This is the baseline prediction that models must beat.
   The player and season filters are only an example (M.Salah, 2024-25),
   remove them to see everyone. */

SELECT
    p.web_name,  -- player name
    p.code,      -- player code (the same person in every season)
    gs.season,
    gs.round,    -- gameweek number
    gs.kickoff_time,
    gs.total_points,
    ROUND(AVG(gs.total_points) OVER (
        PARTITION BY p.code, gs.season
        ORDER BY gs.kickoff_time
        -- 1 PRECEDING excludes the current match, so the average cannot leak the answer
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ), 2) AS avg_last_5
FROM gameweek_stats AS gs
INNER JOIN players AS p ON p.season = gs.season AND p.id = gs.element
WHERE p.code = 118748 AND gs.season = '2024-25'  -- M.Salah
ORDER BY gs.kickoff_time
LIMIT 30;
