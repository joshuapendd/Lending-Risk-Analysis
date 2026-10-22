# How the AI-generated code was checked

This project was built with Claude Code. I treated its output like code from a new analyst: plausible until proven otherwise. Four kinds of checking were used.

## 1. Automated checks (`src/verify.py`, 16/16 passing)
Each check recomputes a result a second, independent way, or tests an invariant that must hold if the code is correct. The latest run is in [`outputs/verification_results.txt`](outputs/verification_results.txt).

| # | Check | How it is independent |
|---|---|---|
| 1 | Row count, raw CSV vs SQLite (2,260,668 both) | Raw file counted with Python's `csv` module, not pandas |
| 2 | 2014 grade-B default rate (10.72%) | Recomputed from the **raw** CSV, bypassing all cleaning code, vs the SQL output |
| 3 | 2014 grade-B net loss ($33,542,922) | Same: raw file vs database |
| 4 | Net loss is between 0 and the funded amount, and 0 for good loans | Invariant over all 2.26M rows |
| 5 | Status shares sum to 100% for every vintage | Invariant |
| 6 | Vintage curve at month 36 equals the cohort default rate | Two different SQL queries must agree |
| 7 | No outcome fields, LendingClub grade/rate, or ZIP/state among model inputs | Explicit forbidden-list check |
| 8–9 | Train/test loan IDs disjoint; IDs unique | Set comparison |
| 10 | Test AUC: sklearn vs a hand-coded Mann-Whitney calculation vs the saved metric | Different formula |
| 11 | Raw default rate falls with FICO and rises with DTI | From SQL, not the model |
| 12 | Model signs: FICO −, DTI +, inquiries +, loan-to-income + | Credit common sense |
| 13 | Coefficients: sklearn vs a statsmodels refit (max diff 0.002) | Different library and solver |
| 14 | Mean predicted PD within 3 points of actual on the 2015 test set | Calibration |
| 15 | At 100% approval all three rankings give identical portfolio results | Invariant |
| 16 | Loss rate rises with approval rate | Invariant |

## 2. Problems found and fixed
These are the cases where the first version of the code or the output was wrong or misleading:

- **Reason codes could be false.** The first adverse-action code labelled each feature with a fixed phrase. Because bankruptcies, public records and employment length came out with counterintuitive signs (they overlap with other features), an applicant with *fewer* bankruptcies than average could have been told "bankruptcy on file". Fixed:
  - each label now depends on whether the applicant is above or below average;
  - only features that increased that applicant's risk can be cited;
  - the counterintuitive-sign features are barred from being cited.
  This would have been a Regulation B problem in production.
- **A misleading delinquency number.** The 2013 vintage showed "30% of open loans 31+ days late". The denominator was 10 loans. I added an `open_loans` column so the sample size is visible.
- **An outlier.** Revolving utilization reached 892% in a few records. It is now capped at 150%.
- **Unreadable chart.** The return chart's axis rounded every label to "7%". Changed to one decimal place.
- **A check bug, not a pipeline bug.** Check 15 first failed. I printed the values rather than loosening the test: the three rankings differed by 1×10⁻¹⁶, which is floating-point rounding from summing in a different order. Switched to a tolerance.
- **Spurious warnings.** sklearn printed "divide by zero / overflow in matmul". I confirmed the inputs had no inf/NaN values. The same warning also appears for a plain matrix multiply on small random arrays, which shows it is a quirk of numpy 2.0 with Apple's Accelerate library on this Mac. I suppressed it with a comment explaining why, and added check 13 so a real numerical problem would still be caught.

## 3. Design decisions checked against domain knowledge
- **Right-censoring.** Default rates for loans that haven't finished their term are understated. The model only uses 36-month vintages up to 2015, and I confirmed they are 99.95% resolved, so dropping the unresolved 0.05% adds no meaningful bias.
- **Leakage.** The raw file has payment, recovery and last-payment fields that are only known after the loan goes bad. These are kept out of the model, and check 7 enforces it.
- **Out-of-time test.** A random split would mix 2015 loans into training. Testing on a later year is how a lender would actually use the model, and it exposed calibration drift that a random split would have hidden.
- **Benchmark.** The model is compared against LendingClub's own sub-grade and against a random ordering, so its AUC has context.

## 4. Manual review
- I read every chart. That caught the axis bug, and it confirmed the grade F/G spike in 2012 is a small-sample artefact (24 grade-G loans), which the write-up says.
- Every number in `FINDINGS.md` was traced back to a file in `outputs/` or a query against the database, including the claims that one FICO standard deviation is about 30 points and that moving from 80% to 50% approval gives up 35% of volume.
