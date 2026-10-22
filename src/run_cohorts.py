"""Run the cohort SQL in sql/ against data/lending.db and save tables + charts."""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "lending.db"
SQL = ROOT / "sql"
OUT = ROOT / "outputs"
FIG = OUT / "figures"


def run(con, name: str) -> pd.DataFrame:
    df = pd.read_sql_query((SQL / f"{name}.sql").read_text(), con)
    df.to_csv(OUT / f"{name}.csv", index=False)
    return df


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as con:
        status = run(con, "01_status_mix_by_vintage")
        grade = run(con, "02_default_by_vintage_grade")
        curves = run(con, "03_vintage_curves")
        factors = run(con, "04_default_by_risk_factor")

    print(status.to_string(index=False), "\n")
    print(factors.to_string(index=False), "\n")

    # Default rate by grade, matured 36-month vintages only.
    pivot = (grade[grade.matured == 1]
             .pivot(index="issue_year", columns="grade", values="default_rate"))
    print(pivot.round(3).to_string(), "\n")
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in pivot.columns:
        ax.plot(pivot.index, pivot[g], marker="o", label=g)
    ax.set(title="Default rate by grade and issue year (36-month, matured vintages)",
           xlabel="Issue year", ylabel="Share charged off / default")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(title="Grade", ncol=2)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "default_by_grade_vintage.png", dpi=150)

    # Vintage curves.
    fig, ax = plt.subplots(figsize=(8, 5))
    for yr, g in curves.groupby("issue_year"):
        ax.plot(g["mob"], g["cum_default_rate"], label=str(yr))
    ax.set(title="Cumulative default by months on book (36-month loans)",
           xlabel="Months on book (to last payment)", ylabel="Cumulative default rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(title="Vintage")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "vintage_curves.png", dpi=150)

    # 30+ DPD among open loans, recent vintages (the only ones with open loans).
    recent = status[status.issue_year >= 2015]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(recent["issue_year"].astype(str), recent["dq30_of_open"])
    ax.set(title="31-120 days past due as a share of open loans, by vintage (snapshot)",
           xlabel="Issue year", ylabel="Share of open loans")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    fig.tight_layout()
    fig.savefig(FIG / "delinquency_open_loans.png", dpi=150)
    print(f"saved tables to {OUT} and figures to {FIG}")


if __name__ == "__main__":
    main()
