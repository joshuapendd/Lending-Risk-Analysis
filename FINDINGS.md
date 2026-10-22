# What 2.26 million LendingClub loans say about credit risk

**Data:** every loan LendingClub issued from 2007 to 2018 (2,260,668 loans), with performance as of March 2019. The cohort analysis and model use 36-month loans issued 2010–2015, which had their full term to play out (99.95% of them are resolved).

## 1. Cohorts: risk is priced, but later vintages got worse
- **Grade separates risk well.** In the 2015 vintage, 5.4% of grade A loans defaulted, against 11.9% for B, 19.4% for C, 26.2% for D and 45.6% for G.
- **Vintages after 2011 performed worse.** Lifetime default on 36-month loans was 10.9% for 2010 and 10.6% for 2011, then 13.7% for 2014 and 14.9% for 2015. The 2015 vintage defaulted more than 2013 in every grade from B to G, so this is not just a mix shift toward riskier grades. The vintage curves track each other for the first ~10 months and spread apart after that.
- **Early delinquency was steady.** Among loans still open at the snapshot, the share 31–120 days past due was about 3% for the 2014–2017 vintages. The 2018 figure (1.7%) is lower mainly because those loans are too new to have gone bad yet.

## 2. A logistic regression default model
- **Inputs:** only information known when the loan was applied for: FICO, DTI, income, loan-to-income, inquiries, utilization, credit history length, home ownership, purpose and verification status. LendingClub's grade and interest rate are left out (they are its own risk model), and so are ZIP code and state.
- **Validation:** trained on 2012–2014 loans (306k) and tested on 2015 loans (283k), so the test is out-of-time.
- **Results:** test AUC is 0.661 (KS 0.23), against 0.658 in training, so there is no overfitting. LendingClub's sub-grade does better at AUC 0.679, which is expected because LendingClub had full credit-bureau files and this model has a public subset.
- **Biggest drivers:**
  - Small-business purpose: 1.74× the odds of default.
  - Each standard deviation of FICO (about 30 points): 0.71× the odds.
  - Renting: 1.21×.
  - More recent credit inquiries: 1.18× per standard deviation.
  - Higher loan-to-income: 1.17× per standard deviation.
  - Higher DTI: 1.15× per standard deviation.
- **Calibration drift:** the model predicted 13.4% defaults for 2015 and the actual rate was 14.9%. It ranks borrowers well but under-predicts the level. In production it would need a recalibration step and drift monitoring.

## 3. Approval rate vs. losses (2015 test vintage)
| Approve the safest… | Default rate | Net loss / funded | Realized return* |
|---|---|---|---|
| 50% | 9.0% | 4.4% | 7.5% |
| 80% | 12.2% | 6.1% | 7.1% |
| 100% (everyone LendingClub approved) | 14.9% | 7.6% | 6.6% |

\*Total cash collected divided by principal, minus 1. Over a three-year term, before servicing fees and funding costs.

- **The riskiest loans barely pay for themselves.** The riskiest 10% of approved loans defaulted at 28.8%, lost 16.4% of principal and returned only 3.0% in cash over three years. After fees and funding costs that is roughly break-even or worse. Moving from 80% to 50% approval cuts losses by about 30% but gives up 35% of loan volume. Which point is right depends on funding costs and on whether those customers have value beyond this loan.
- **Fewer losses is not the same as more profit.** LendingClub's sub-grade gives lower losses than this model at every approval rate, but this model earns a higher return (7.07% vs 7.02% at 80%, 7.46% vs 7.14% at 50%). Ranking by sub-grade favors low-rate loans. This model never sees the rate, so it keeps some high-rate loans that turned out fine. The decision should be optimized on expected profit, not just on default rate.

## 4. Fair lending risks
- **No direct test is possible.** The data has no race, sex or age, so a direct disparate-impact test can't be run. A real lender would estimate these with BISG (a method that infers likely race and ethnicity from surname and address) or use HMDA-style data.
- **Proxies.** ZIP and state are excluded from the model, but income, DTI, home ownership and loan purpose are known to correlate with race and sex. Each one needs a documented business justification and a search for less discriminatory alternatives.
- **Geographic screen.** At an 80% approval cutoff, approval rates across states ranged from 73% to 85%. Hawaii has the lowest approval rate (73%) despite a below-average default rate (13.7% vs 14.9%). The likely cause is that higher cost of living pushes up DTI and loan-to-income. The model is penalizing borrowers there for something that doesn't predict default for them, and that pattern is what a disparate-impact review would chase.
- **Adverse-action reasons (ECOA/Regulation B).** The logistic model produces specific reasons for each declined applicant. Among applicants declined at 80% approval, the top primary reasons were credit score (43%), loan-to-income (16%) and recent inquiries (16%). Three features have signs that run the wrong way, such as bankruptcies appearing to *lower* risk because they overlap with other features, and a fourth (past delinquencies) has almost no effect. All four are barred from being cited as reasons so no applicant gets a reason that isn't true. In production they should be reviewed or dropped.
- **Selection bias.** The data contains approved loans only, so the model has never seen the applicants LendingClub declined. Using it to approve beyond LendingClub's own approval line would need reject inference and a controlled test.
