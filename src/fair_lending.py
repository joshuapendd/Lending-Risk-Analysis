"""Fair-lending checks that are possible on this data.

LendingClub's public file has no race, sex, age or ethnicity, so a direct
disparate-impact test is not possible. This script does the two things that
are possible and that a lender would need anyway:

1. Adverse-action reasons. ECOA / Regulation B requires lenders to give
   declined applicants the principal reasons. For a logistic model these
   come from each feature's contribution to the applicant's score relative
   to the average applicant.
2. A geographic screen. Approval rates by state at the chosen cutoff,
   computed even though state is not a model input, because geography is
   the classic channel for proxy discrimination (redlining).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
APPROVAL_RATE = 0.80  # the operating point discussed in the write-up

# Reason text depends on which side of average the applicant is on:
# (text if above average, text if below average).
READABLE = {
    "fico": ("Credit score", "Credit score too low"),
    "dti": ("Debt-to-income ratio too high", "Debt-to-income ratio"),
    "log_income": ("Income", "Income insufficient for amount requested"),
    "loan_to_income": ("Loan amount too high relative to income", "Loan-to-income"),
    "log_loan_amnt": ("Loan amount requested too high", "Loan amount"),
    "inq_last_6mths": ("Too many recent credit inquiries", "Credit inquiries"),
    "revol_util": ("Revolving credit utilization too high", "Revolving utilization"),
    "log_revol_bal": ("Revolving balance", "Limited revolving credit experience"),
    "credit_hist_yrs": ("Credit history", "Length of credit history too short"),
    "emp_length_missing": ("Length of employment not provided", "Employment"),
    "mort_acc": ("Mortgage accounts", "No or few mortgage accounts"),
    "open_acc": ("Too many open accounts", "Open accounts"),
    "total_acc": ("Credit accounts", "Limited number of credit accounts"),
}
# Features whose fitted sign is counterintuitive or ~0 (collinear with other
# features). They stay in the score but are never cited as a reason, since
# "fewer bankruptcies than average" cannot honestly be an adverse reason.
NOT_REASONS = {"pub_rec", "pub_rec_bankruptcies", "delinq_2yrs", "emp_length_yrs"}


def main() -> None:
    from model import NUMERIC  # same feature list as the model

    test = pd.read_parquet(OUT / "test_scored.parquet")
    coefs = pd.read_csv(OUT / "coefficients.csv").set_index("feature")["coef"]

    cutoff = test["pd"].quantile(APPROVAL_RATE)
    test["approved"] = test["pd"] <= cutoff
    declined = test[~test.approved]
    print(f"cutoff PD at {APPROVAL_RATE:.0%} approval: {cutoff:.4f}; "
          f"declined {len(declined):,} of {len(test):,}")

    # Contribution of each numeric feature, in log-odds, vs the average
    # applicant. Standardize with test-set statistics after median imputation
    # (the model's own scaler uses training statistics; the ranking of
    # reasons is what matters and is insensitive to this).
    X = test[NUMERIC].fillna(test[NUMERIC].median())
    Z = (X - X.mean()) / X.std(ddof=0)
    contrib = Z * coefs[[f"num__{c}" for c in NUMERIC]].values
    contrib = contrib.drop(columns=list(NOT_REASONS))

    def reasons(i: int) -> list:
        c = contrib.loc[i]
        c = c[c > 0].sort_values(ascending=False).index[:4]  # risk-increasing only
        return [READABLE[f][0] if Z.at[i, f] > 0 else READABLE[f][1] for f in c]

    top = {i: reasons(i) for i in declined.index}
    reason_counts = (pd.Series([r[0] for r in top.values() if r])
                     .value_counts(normalize=True).rename("share_primary_reason"))
    reason_counts.to_csv(OUT / "adverse_action_primary_reasons.csv")
    print("\nPrimary adverse-action reason among declined applicants:")
    print(reason_counts.round(3).to_string())

    print("\nExample reason codes:")
    for i in declined.sample(4, random_state=1).index:
        print(f"  loan {test.at[i, 'id']}  PD={test.at[i, 'pd']:.3f}: " + "; ".join(top[i]))

    # Geographic screen.
    by_state = (test.groupby("addr_state")
                .agg(applicants=("approved", "size"), approval_rate=("approved", "mean"),
                     actual_bad_rate=("is_bad", "mean"))
                .query("applicants >= 1000").sort_values("approval_rate"))
    by_state.to_csv(OUT / "approval_by_state.csv")
    overall = test.approved.mean()
    by_state["air_vs_overall"] = by_state.approval_rate / overall
    print(f"\nApproval rate by state (states with 1,000+ loans), overall {overall:.1%}:")
    print(by_state.head(5).round(3).to_string())
    print("...")
    print(by_state.tail(5).round(3).to_string())
    print(f"lowest/highest approval ratio: "
          f"{by_state.approval_rate.min() / by_state.approval_rate.max():.3f}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
