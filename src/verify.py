"""Independent checks on the pipeline's outputs.

Each check recomputes a number a second way (raw CSV instead of the cleaned
database, pandas instead of SQL, by hand instead of sklearn) or asserts an
invariant that must hold if the code is right. Results are written to
outputs/verification_results.txt and summarised in VERIFICATION.md.
"""
import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
DB = ROOT / "data" / "lending.db"
OUT = ROOT / "outputs"
sys.path.insert(0, str(Path(__file__).parent))
from model import NUMERIC, CATEGORICAL, QUERY  # noqa: E402

results = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(results[-1])


def main() -> None:
    con = sqlite3.connect(DB)

    # 1. Row count: raw CSV read with the stdlib csv module (no pandas) vs DB.
    with open(RAW, newline="") as f:
        reader = csv.DictReader(f)
        raw_n = sum(1 for r in reader if r["loan_amnt"].strip())
    db_n = con.execute("SELECT COUNT(*) FROM loans").fetchone()[0]
    check("row count raw CSV vs SQLite", raw_n == db_n, f"csv={raw_n:,} db={db_n:,}")

    # 2. A cohort cell from the SQL output recomputed from the *raw* file,
    #    bypassing all cleaning code: 36-month, 2014, grade B.
    raw = pd.read_csv(RAW, usecols=["term", "issue_d", "grade", "loan_status",
                                    "funded_amnt", "total_rec_prncp", "recoveries",
                                    "collection_recovery_fee"], low_memory=False)
    cell = raw[(raw.term.str.strip() == "36 months") & raw.issue_d.str.endswith("2014", na=False)
               & (raw.grade == "B")]
    raw_rate = cell.loan_status.isin(["Charged Off", "Default"]).mean()
    sql = pd.read_csv(OUT / "02_default_by_vintage_grade.csv")
    sql_rate = sql.query("issue_year == 2014 and grade == 'B'").default_rate.iloc[0]
    check("2014 grade-B default rate, raw CSV vs SQL", abs(raw_rate - sql_rate) < 1e-4,
          f"raw={raw_rate:.4f} sql={sql_rate:.4f} (n={len(cell):,})")

    # 3. Net loss: independent recompute of total 2014 36-month losses.
    bad = cell.loan_status.isin(["Charged Off", "Default"])
    raw_loss = (cell.funded_amnt - cell.total_rec_prncp - cell.recoveries
                + cell.collection_recovery_fee)[bad].clip(lower=0).sum()
    db_loss = con.execute("""SELECT SUM(net_loss) FROM loans WHERE term_months=36
                             AND issue_year=2014 AND grade='B'""").fetchone()[0]
    check("2014 grade-B net loss, raw vs DB", abs(raw_loss - db_loss) < 1,
          f"raw=${raw_loss:,.0f} db=${db_loss:,.0f}")

    # 4. Loss invariants.
    bad_rows = con.execute("""SELECT COUNT(*) FROM loans
                              WHERE net_loss < 0 OR net_loss > funded_amnt + 0.01
                                 OR (is_bad = 0 AND net_loss <> 0)""").fetchone()[0]
    check("0 <= net_loss <= funded, zero for good loans", bad_rows == 0, f"violations={bad_rows}")

    # 5. Status buckets in query 01 sum to 100% for every vintage.
    st = pd.read_csv(OUT / "01_status_mix_by_vintage.csv")
    total = st[["pct_current", "pct_grace", "pct_late_16_30", "pct_late_31_120",
                "pct_bad", "pct_paid"]].sum(axis=1)
    check("status shares sum to 1 per vintage", bool(np.allclose(total, 1, atol=2e-3)),
          f"range {total.min():.4f}-{total.max():.4f}")

    # 6. Cross-query consistency: vintage curve endpoint (query 03) equals the
    #    overall default rate for that vintage computed directly.
    curves = pd.read_csv(OUT / "03_vintage_curves.csv")
    ends = curves[curves.mob == 36].set_index("issue_year").cum_default_rate
    direct = pd.read_sql_query("""SELECT issue_year, AVG(is_bad) r FROM loans
                                  WHERE term_months=36 AND issue_year BETWEEN 2010 AND 2015
                                  GROUP BY issue_year""", con).set_index("issue_year").r
    check("vintage curve at month 36 == cohort default rate",
          bool(np.allclose(ends, direct, atol=1e-4)),
          "max diff " + f"{(ends - direct).abs().max():.6f}")

    # 7. Leakage: no outcome or LendingClub-pricing field among model inputs,
    #    and train/test are disjoint in time and in loan id.
    forbidden = {"loan_status", "total_pymnt", "total_rec_prncp", "recoveries",
                 "last_pymnt_d", "net_loss", "is_bad", "grade", "sub_grade",
                 "int_rate", "installment", "zip_code", "addr_state"}
    check("no leakage / excluded fields in features",
          not forbidden & set(NUMERIC + CATEGORICAL), str(sorted(set(NUMERIC + CATEGORICAL))))
    sample = pd.read_sql_query(QUERY, con)
    tr_ids = set(sample[sample.issue_year <= 2014].id)
    te_ids = set(sample[sample.issue_year == 2015].id)
    check("train/test loan ids disjoint", not tr_ids & te_ids,
          f"train={len(tr_ids):,} test={len(te_ids):,} overlap={len(tr_ids & te_ids)}")
    check("loan ids unique in sample", sample.id.is_unique, f"n={len(sample):,}")

    # 8. AUC recomputed by hand (Mann-Whitney U) vs sklearn and vs the saved metric.
    test = pd.read_parquet(OUT / "test_scored.parquet")
    y, p = test.is_bad.values, test.pd.values
    r = rankdata(p)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc_manual = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    saved = json.loads((OUT / "model_metrics.json").read_text())["test_auc"]
    check("test AUC: manual vs sklearn vs saved",
          abs(auc_manual - roc_auc_score(y, p)) < 1e-9 and abs(auc_manual - saved) < 1e-9,
          f"manual={auc_manual:.5f} saved={saved:.5f}")

    # 9. Coefficient directions agree with raw default rates (query 04) and
    #    with credit common sense.
    coefs = pd.read_csv(OUT / "coefficients.csv").set_index("feature").coef
    fac = pd.read_csv(OUT / "04_default_by_risk_factor.csv")
    fico_desc = fac[fac.factor == "fico"].default_rate.is_monotonic_decreasing
    dti_asc = fac[fac.factor == "dti"].default_rate.is_monotonic_increasing
    signs = {"num__fico": -1, "num__dti": 1, "num__inq_last_6mths": 1,
             "num__loan_to_income": 1}
    wrong = [f for f, s in signs.items() if np.sign(coefs[f]) != s]
    check("raw data: default falls with FICO, rises with DTI", fico_desc and dti_asc,
          f"fico monotone={fico_desc} dti monotone={dti_asc}")
    check("model coefficient signs as expected", not wrong,
          ", ".join(f"{f}={coefs[f]:+.3f}" for f in signs))

    # 9b. Refit the same design with statsmodels (unregularized MLE, a
    #     different solver and library) and compare coefficients to sklearn's.
    import statsmodels.api as sm
    from model import features
    import warnings
    warnings.filterwarnings("ignore", message=".*encountered in matmul")
    train = features(sample[sample.issue_year <= 2014])
    num = train[NUMERIC].fillna(train[NUMERIC].median())
    num = (num - num.mean()) / num.std(ddof=0)
    cat = pd.get_dummies(train[CATEGORICAL], drop_first=True, dtype=float)
    Xsm = sm.add_constant(pd.concat([num.add_prefix("num__"), cat], axis=1))
    sm_fit = sm.Logit(train.is_bad.values, Xsm).fit(disp=0, maxiter=200)
    shared = [f"num__{c}" for c in NUMERIC]
    diff = (sm_fit.params[shared] - coefs[shared]).abs().max()
    check("numeric coefficients: sklearn vs statsmodels refit", diff < 0.01,
          f"max abs diff {diff:.4f} (sklearn has light L2 penalty, C=1)")

    # 10. Calibration: out-of-time mean PD vs actual bad rate. Not a hard
    #     requirement (2015 was a worse vintage), so the gap is reported.
    gap = test.pd.mean() - test.is_bad.mean()
    check("mean PD vs actual bad rate within 3 pts on 2015 test",
          abs(gap) < 0.03, f"mean PD={test.pd.mean():.4f} actual={test.is_bad.mean():.4f}")

    # 11. Tradeoff sanity: at 100% approval every ranking gives identical
    #     results, equal to the test-set totals; loss rate rises with approval.
    #     (Compared with a tolerance: summing in a different order differs by ~1e-16.)
    tr = pd.read_csv(OUT / "approval_tradeoff.csv")
    full = tr[np.isclose(tr.approval_rate, 1.0)]
    whole_loss = test.net_loss.sum() / test.funded_amnt.sum()
    check("100% approval identical across rankings and == portfolio loss",
          np.ptp(full.net_loss_rate) < 1e-12 and abs(full.net_loss_rate.iloc[0] - whole_loss) < 1e-12,
          f"loss rate {whole_loss:.4f}")
    mono = tr[tr.score == "pd"].net_loss_rate.is_monotonic_increasing
    check("model loss rate increases with approval rate", mono, "")

    (OUT / "verification_results.txt").write_text("\n".join(results) + "\n")
    n_fail = sum(r.startswith("[FAIL]") for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
