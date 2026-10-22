-- Default (charged off / default) rate and net loss rate by issue year and
-- grade, 36-month loans only. Cohorts issued after 2015 have not reached
-- maturity by the snapshot, so their rates are understated (right-censored);
-- the `matured` flag marks the cohorts that can be compared fairly.
SELECT
    issue_year,
    grade,
    COUNT(*)                                              AS loans,
    ROUND(AVG(is_bad), 4)                                 AS default_rate,
    ROUND(SUM(net_loss) / SUM(funded_amnt), 4)            AS net_loss_rate,
    ROUND(AVG(int_rate), 2)                               AS avg_int_rate,
    CASE WHEN issue_year <= 2015 THEN 1 ELSE 0 END        AS matured
FROM loans
WHERE term_months = 36
GROUP BY issue_year, grade
ORDER BY issue_year, grade;
