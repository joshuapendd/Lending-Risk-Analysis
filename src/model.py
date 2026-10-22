"""Logistic regression probability-of-default model and approval/loss tradeoff.

Sample: 36-month loans issued 2012-2015 that have resolved (Fully Paid,
Charged Off or Default). These vintages had the full 36-month term to play
out before the snapshot, so outcomes are not right-censored.

Validation is out-of-time: train on 2012-2014 issuance, test on 2015.
"""
import json
import sqlite3
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# numpy 2.0 with macOS Accelerate emits spurious "divide by zero / overflow
# in matmul" warnings even on small random finite arrays. Inputs were checked
# for inf/NaN and coefficients are cross-checked in verify.py.
warnings.filterwarnings("ignore", message=".*encountered in matmul")

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "lending.db"
OUT = ROOT / "outputs"
FIG = OUT / "figures"

# Application-time borrower attributes only. Deliberately excluded:
#  - grade, sub_grade, int_rate, installment: LendingClub's own risk pricing
#    (used below as a benchmark, not as inputs)
#  - zip_code, addr_state: geographic fields that can proxy for race
#  - anything recorded after origination (payments, recoveries, last_pymnt_d)
NUMERIC = [
    "fico", "dti", "log_income", "log_loan_amnt", "loan_to_income",
    "emp_length_yrs", "emp_length_missing", "revol_util", "log_revol_bal",
    "inq_last_6mths", "delinq_2yrs", "open_acc", "total_acc", "pub_rec",
    "pub_rec_bankruptcies", "mort_acc", "credit_hist_yrs",
]
CATEGORICAL = ["home_ownership", "purpose", "verification_status"]

QUERY = """
SELECT id, issue_year, is_bad, net_loss, funded_amnt, total_pymnt,
       grade, sub_grade, int_rate, addr_state,
       fico, dti, annual_inc, loan_amnt, emp_length_yrs, revol_util, revol_bal,
       inq_last_6mths, delinq_2yrs, open_acc, total_acc, pub_rec,
       pub_rec_bankruptcies, mort_acc, credit_hist_yrs,
       home_ownership, purpose, verification_status
FROM loans
WHERE term_months = 36
  AND issue_year BETWEEN 2012 AND 2015
  AND is_resolved = 1
  AND application_type = 'Individual'
"""


def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_income"] = np.log1p(df["annual_inc"])
    df["log_loan_amnt"] = np.log(df["loan_amnt"])
    df["loan_to_income"] = (df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)).clip(upper=2)
    df["emp_length_missing"] = df["emp_length_yrs"].isna().astype(int)
    df["log_revol_bal"] = np.log1p(df["revol_bal"])
    df["dti"] = df["dti"].clip(0, 60)
    df["revol_util"] = df["revol_util"].clip(0, 150)  # a few records exceed 800%
    # Collapse rare categories so the test year never sees an unseen level.
    df["home_ownership"] = df["home_ownership"].where(
        df["home_ownership"].isin(["RENT", "MORTGAGE", "OWN"]), "OTHER")
    return df


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def tradeoff(df: pd.DataFrame, score: str) -> pd.DataFrame:
    """Approve the lowest-risk X% by `score`; report realized outcomes."""
    d = df.sort_values(score, kind="mergesort").reset_index(drop=True)
    rows = []
    for rate in np.arange(0.10, 1.0001, 0.05):
        a = d.iloc[: int(round(rate * len(d)))]
        rows.append({
            "score": score,
            "approval_rate": round(rate, 2),
            "default_rate": a["is_bad"].mean(),
            "net_loss_rate": a["net_loss"].sum() / a["funded_amnt"].sum(),
            # Realized cash return: everything collected vs principal lent.
            "net_return": a["total_pymnt"].sum() / a["funded_amnt"].sum() - 1,
            "funded_volume": a["funded_amnt"].sum(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as con:
        df = features(pd.read_sql_query(QUERY, con))

    train = df[df.issue_year <= 2014]
    test = df[df.issue_year == 2015].copy()
    print(f"train {len(train):,} loans, bad rate {train.is_bad.mean():.3%}")
    print(f"test  {len(test):,} loans, bad rate {test.is_bad.mean():.3%}")

    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
    ])
    model = Pipeline([("pre", pre),
                      ("lr", LogisticRegression(max_iter=2000, C=1.0))])
    model.fit(train[NUMERIC + CATEGORICAL], train["is_bad"])

    test["pd"] = model.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]
    train_pd = model.predict_proba(train[NUMERIC + CATEGORICAL])[:, 1]
    # Benchmarks: LendingClub's own sub-grade (A1=1 ... G5=35) and interest rate.
    test["subgrade_rank"] = test["sub_grade"].map(
        {f"{g}{i}": n for n, (g, i) in enumerate(
            ((g, i) for g in "ABCDEFG" for i in range(1, 6)), start=1)})
    test["random"] = np.random.default_rng(0).random(len(test))

    metrics = {
        "train_loans": int(len(train)), "test_loans": int(len(test)),
        "train_bad_rate": float(train.is_bad.mean()), "test_bad_rate": float(test.is_bad.mean()),
        "train_auc": float(roc_auc_score(train.is_bad, train_pd)),
        "test_auc": float(roc_auc_score(test.is_bad, test["pd"])),
        "test_ks": ks_stat(test.is_bad, test["pd"]),
        "test_auc_lc_subgrade": float(roc_auc_score(test.is_bad, test["subgrade_rank"])),
        "test_auc_lc_int_rate": float(roc_auc_score(test.is_bad, test["int_rate"])),
        "test_mean_pd": float(test["pd"].mean()),
    }
    metrics["test_gini"] = 2 * metrics["test_auc"] - 1
    print(json.dumps(metrics, indent=2))
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Coefficients (numeric ones are per 1 standard deviation).
    names = model.named_steps["pre"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": names, "coef": model.named_steps["lr"].coef_[0]})
    coefs["odds_ratio"] = np.exp(coefs["coef"])
    coefs = coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index)
    coefs.to_csv(OUT / "coefficients.csv", index=False)
    print(coefs.head(15).to_string(index=False))

    # Calibration by decile of predicted PD on the out-of-time test set.
    test["decile"] = pd.qcut(test["pd"], 10, labels=False) + 1
    calib = test.groupby("decile").agg(loans=("is_bad", "size"),
                                       mean_pd=("pd", "mean"),
                                       actual_bad_rate=("is_bad", "mean")).reset_index()
    calib.to_csv(OUT / "calibration_by_decile.csv", index=False)
    print(calib.round(4).to_string(index=False))

    # Approval-rate vs loss tradeoff, three ways of ranking applicants.
    tr = pd.concat([tradeoff(test, s) for s in ["pd", "subgrade_rank", "random"]])
    tr.to_csv(OUT / "approval_tradeoff.csv", index=False)
    print(tr[tr.score == "pd"].round(4).to_string(index=False))

    labels = {"pd": "This model", "subgrade_rank": "LendingClub sub-grade",
              "random": "Random order"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for s, g in tr.groupby("score", sort=False):
        axes[0].plot(g.approval_rate, g.net_loss_rate, marker="o", ms=3, label=labels[s])
        axes[1].plot(g.approval_rate, g.net_return, marker="o", ms=3, label=labels[s])
    axes[0].set(title="Net loss rate vs approval rate (2015 test vintage)",
                xlabel="Approval rate", ylabel="Net principal loss / funded")
    axes[1].set(title="Realized net return vs approval rate",
                xlabel="Approval rate", ylabel="Total collected / funded - 1")
    for ax in axes:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "approval_tradeoff.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(calib.mean_pd, calib.actual_bad_rate, marker="o")
    lim = max(calib.mean_pd.max(), calib.actual_bad_rate.max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", c="grey")
    ax.set(title="Calibration, 2015 test vintage", xlabel="Mean predicted PD",
           ylabel="Actual default rate", xlim=(0, lim), ylim=(0, lim))
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)

    # Save scored test set for the fair-lending and verification scripts.
    test[["id", "issue_year", "addr_state", "is_bad", "pd", "subgrade_rank",
          "net_loss", "funded_amnt", "total_pymnt"] + NUMERIC + CATEGORICAL
         ].to_parquet(OUT / "test_scored.parquet", index=False)


if __name__ == "__main__":
    main()
