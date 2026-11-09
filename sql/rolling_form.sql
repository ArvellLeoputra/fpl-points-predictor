.headers on
.mode column

/* Rolling form: for each match, the average points of the player's
   previous 5 matches. The current match is excluded (no leakage).
   This is the baseline prediction that models must beat. */

SELECT
    p.web_name,  -- player name
    gs.element,  -- player id
    gs.round,    -- gameweek number
    gs.kickoff_time,
    gs.total_points,
    ROUND(AVG(gs.total_points) OVER (
        PARTITION BY gs.element
        ORDER BY gs.kickoff_time
        -- 1 PRECEDING excludes the current match, so the average cannot leak the answer
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ), 2) AS avg_last_5
FROM gameweek_stats AS gs
INNER JOIN players AS p ON p.id = gs.element
ORDER BY gs.element, gs.kickoff_time
LIMIT 30;
