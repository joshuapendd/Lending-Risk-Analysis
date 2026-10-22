-- Cumulative default curve by months on book, 36-month loans, 2010-2015
-- vintages. Months-on-book at default is approximated by the months from
-- issue to the last payment received (LendingClub charges off ~120+ days
-- after the last payment, so true charge-off is a few months later; the
-- shape of the curve is what matters here).
WITH base AS (
    SELECT
        issue_year,
        is_bad,
        CASE
            WHEN is_bad = 1 AND last_pymnt_d IS NULL THEN 0
            WHEN is_bad = 1 THEN
                (CAST(substr(last_pymnt_d,1,4) AS INT) * 12 + CAST(substr(last_pymnt_d,6,2) AS INT))
              - (CAST(substr(issue_d,1,4)       AS INT) * 12 + CAST(substr(issue_d,6,2)       AS INT))
        END AS mob_at_default
    FROM loans
    WHERE term_months = 36 AND issue_year BETWEEN 2010 AND 2015
),
cohort AS (
    SELECT issue_year, COUNT(*) AS n FROM base GROUP BY issue_year
),
defaults AS (
    SELECT issue_year, MIN(mob_at_default, 36) AS mob, COUNT(*) AS d
    FROM base WHERE is_bad = 1
    GROUP BY issue_year, MIN(mob_at_default, 36)
),
grid AS (  -- every (cohort, month 0..36) pair so curves have no gaps
    WITH RECURSIVE m(mob) AS (SELECT 0 UNION ALL SELECT mob + 1 FROM m WHERE mob < 36)
    SELECT c.issue_year, m.mob, c.n FROM cohort c CROSS JOIN m
)
SELECT
    g.issue_year,
    g.mob,
    ROUND(1.0 * SUM(COALESCE(d.d, 0)) OVER (PARTITION BY g.issue_year ORDER BY g.mob)
          / g.n, 5) AS cum_default_rate
FROM grid g
LEFT JOIN defaults d ON d.issue_year = g.issue_year AND d.mob = g.mob
ORDER BY g.issue_year, g.mob;
