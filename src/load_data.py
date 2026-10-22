"""Load the raw LendingClub accepted-loans CSV into a cleaned SQLite table.

Usage: python src/load_data.py [path/to/accepted_2007_to_2018Q4.csv]

Only columns known at application time, plus the outcome/payment fields
needed to measure performance, are kept. Everything downstream (SQL cohorts,
model, tradeoff) reads from data/lending.db.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
DB = ROOT / "data" / "lending.db"

# Application-time attributes (available when the credit decision is made).
APP_COLS = [
    "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "issue_d", "purpose", "zip_code", "addr_state",
    "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low",
    "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "application_type", "mort_acc",
    "pub_rec_bankruptcies",
]
# Outcome / post-origination fields. Never used as model features.
OUTCOME_COLS = [
    "loan_status", "total_pymnt", "total_rec_prncp", "recoveries",
    "collection_recovery_fee", "last_pymnt_d",
]

BAD_STATUSES = {"Charged Off", "Default"}


def to_month(s: pd.Series) -> pd.Series:
    """'Dec-2015' -> '2015-12-01' (ISO string, sorts correctly in SQLite)."""
    return pd.to_datetime(s, format="%b-%Y", errors="coerce").dt.strftime("%Y-%m-%d")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # The file ends with summary rows that have no loan amount; drop them.
    df = df[df["loan_amnt"].notna()].copy()

    # LendingClub prefixes loans issued outside its standard credit policy
    # (2007-2010) with this string. Keep the flag, normalize the status.
    prefix = "Does not meet the credit policy. Status:"
    df["policy_exception"] = df["loan_status"].str.startswith(prefix).astype(int)
    df["loan_status"] = df["loan_status"].str.replace(prefix, "", regex=False).str.strip()

    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float).astype("Int64")
    df = df.drop(columns="term")

    emp = df["emp_length"].replace({"< 1 year": "0", "10+ years": "10"})
    df["emp_length_yrs"] = pd.to_numeric(emp.str.extract(r"(\d+)")[0], errors="coerce")
    df = df.drop(columns="emp_length")

    for c in ["issue_d", "last_pymnt_d", "earliest_cr_line"]:
        df[c] = to_month(df[c])

    df["issue_year"] = df["issue_d"].str[:4].astype("Int64")
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    # Credit history length at application, in years.
    issue = pd.to_datetime(df["issue_d"])
    ecl = pd.to_datetime(df["earliest_cr_line"])
    df["credit_hist_yrs"] = ((issue - ecl).dt.days / 365.25).round(2)

    df["is_bad"] = df["loan_status"].isin(BAD_STATUSES).astype(int)
    df["is_resolved"] = df["loan_status"].isin(BAD_STATUSES | {"Fully Paid"}).astype(int)

    # Net principal loss on bad loans: principal not repaid, less net recoveries.
    net_rec = df["recoveries"].fillna(0) - df["collection_recovery_fee"].fillna(0)
    loss = df["funded_amnt"] - df["total_rec_prncp"].fillna(0) - net_rec
    df["net_loss"] = np.where(df["is_bad"] == 1, loss.clip(lower=0), 0.0)

    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    return df


def main(path: Path) -> None:
    print(f"Reading {path}")
    df = pd.read_csv(path, usecols=APP_COLS + OUTCOME_COLS, low_memory=False)
    raw_rows = len(df)
    df = clean(df)
    print(f"raw rows: {raw_rows:,}  cleaned rows: {len(df):,}")

    DB.unlink(missing_ok=True)
    with sqlite3.connect(DB) as con:
        df.to_sql("loans", con, index=False, chunksize=50_000)
        con.execute("CREATE INDEX ix_issue ON loans(issue_year, term_months)")
        con.execute("CREATE INDEX ix_grade ON loans(grade)")
        n = con.execute("SELECT COUNT(*) FROM loans").fetchone()[0]
    assert n == len(df), "row count mismatch between pandas and SQLite"
    print(f"wrote {n:,} rows to {DB}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RAW)
