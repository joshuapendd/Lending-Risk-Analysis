-- Delinquency snapshot: share of each issue-year cohort in each status
-- as of the data snapshot (early 2019). Late buckets are only meaningful
-- for recent vintages that still have loans outstanding.
SELECT
    issue_year,
    COUNT(*)                                                          AS loans,
    ROUND(AVG(loan_status = 'Current'), 4)                            AS pct_current,
    ROUND(AVG(loan_status = 'In Grace Period'), 4)                    AS pct_grace,
    ROUND(AVG(loan_status = 'Late (16-30 days)'), 4)                  AS pct_late_16_30,
    ROUND(AVG(loan_status = 'Late (31-120 days)'), 4)                 AS pct_late_31_120,
    ROUND(AVG(loan_status IN ('Charged Off', 'Default')), 4)          AS pct_bad,
    ROUND(AVG(loan_status = 'Fully Paid'), 4)                         AS pct_paid,
    SUM(loan_status IN ('Current','In Grace Period',
        'Late (16-30 days)','Late (31-120 days)'))                    AS open_loans,
    -- 31+ days past due among loans still outstanding (noisy when open_loans is small)
    ROUND(1.0 * SUM(loan_status = 'Late (31-120 days)')
          / NULLIF(SUM(loan_status IN ('Current','In Grace Period',
                       'Late (16-30 days)','Late (31-120 days)')), 0), 4) AS dq30_of_open
FROM loans
GROUP BY issue_year
ORDER BY issue_year;
