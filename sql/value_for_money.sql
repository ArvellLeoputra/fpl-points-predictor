.headers on
.mode column

/* Value for money: season points per million of current price.
   Only players with at least 450 minutes (about 5 full matches),
   so a single lucky appearance cannot top the list. */

SELECT 
   p.web_name,
   p.now_cost,  -- price in tenths of a million (54 = 5.4m)
   SUM(gs.total_points) AS season_points,
   ROUND(SUM(gs.total_points) / (p.now_cost / 10.0), 2) AS points_per_million
FROM gameweek_stats AS gs
JOIN players AS p ON p.id = gs.element
GROUP BY p.id
HAVING SUM(gs.minutes) >= 450
ORDER BY points_per_million DESC
LIMIT 10;