-- Default rate by FICO band and DTI band for matured, resolved 36-month
-- loans (issued 2012-2015). Used to sanity-check that the model's
-- coefficients point in the direction the raw data does.
WITH m AS (
    SELECT *,
        CASE WHEN fico < 680 THEN '660-679'
             WHEN fico < 700 THEN '680-699'
             WHEN fico < 720 THEN '700-719'
             WHEN fico < 750 THEN '720-749'
             ELSE '750+' END AS fico_band,
        CASE WHEN dti < 10 THEN '0-10'
             WHEN dti < 20 THEN '10-20'
             WHEN dti < 30 THEN '20-30'
             ELSE '30+' END AS dti_band
    FROM loans
    WHERE term_months = 36 AND issue_year BETWEEN 2012 AND 2015 AND is_resolved = 1
)
SELECT 'fico' AS factor, fico_band AS band, COUNT(*) AS loans, ROUND(AVG(is_bad), 4) AS default_rate
FROM m GROUP BY fico_band
UNION ALL
SELECT 'dti', dti_band, COUNT(*), ROUND(AVG(is_bad), 4)
FROM m WHERE dti IS NOT NULL GROUP BY dti_band
ORDER BY factor, band;
