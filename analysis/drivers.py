"""
Attrition driver analysis: who leaves, and is the difference real?

For each factor, computes the attrition rate per group and a chi-square
test of independence (does attrition depend on this factor, or could the
differences be chance?). Writes:
  outputs/driver_rates.csv     every group's headcount and attrition rate
  outputs/driver_tests.csv     chi-square statistic and p-value per factor
  outputs/drivers_chart.png    the strongest factors, side by side

Run from the repository root:  python analysis/drivers.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attrition_lib import OUTPUT_DIR, add_bands, load_data  # noqa: E402

FACTORS = [
    "OverTime", "JobLevel", "TenureBand", "AgeBand", "IncomeQuartile",
    "MaritalStatus", "StockOptionLevel", "BusinessTravel", "JobRole",
    "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
    "Department", "CommuteBand", "Gender", "Education", "PerformanceRating",
]


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = add_bands(load_data())
    overall = df["Left"].mean()
    print(f"Employees: {len(df)}  Left: {df['Left'].sum()}  Attrition rate: {overall:.1%}\n")

    rate_rows, test_rows = [], []
    for factor in FACTORS:
        table = pd.crosstab(df[factor], df["Left"])
        chi2, p, dof, _ = chi2_contingency(table)
        test_rows.append({"factor": factor, "chi2": round(chi2, 2), "dof": dof, "p_value": p,
                          "significant_at_5pct": p < 0.05})
        grouped = df.groupby(factor, observed=True)["Left"].agg(["count", "sum", "mean"])
        for group, row in grouped.iterrows():
            rate_rows.append({"factor": factor, "group": str(group), "employees": int(row["count"]),
                              "left": int(row["sum"]), "attrition_rate_pct": round(100 * row["mean"], 1)})

    rates = pd.DataFrame(rate_rows)
    tests = pd.DataFrame(test_rows).sort_values("p_value")
    rates.to_csv(OUTPUT_DIR / "driver_rates.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "driver_tests.csv", index=False)
    print(tests.to_string(index=False))

    # Combined segment called out in the README
    seg = df[(df["OverTime"] == "Yes") & (df["JobLevel"] == 1)]
    print(f"\nOvertime + job level 1: {len(seg)} employees, attrition {seg['Left'].mean():.1%}")
    ot_share = (df.loc[df["Left"] == 1, "OverTime"] == "Yes").mean()
    print(f"Share of all leavers who worked overtime: {ot_share:.1%} "
          f"(overtime staff are {(df['OverTime'] == 'Yes').mean():.1%} of headcount)")

    # Chart: six strongest factors
    top = ["OverTime", "TenureBand", "JobLevel", "IncomeQuartile", "MaritalStatus", "StockOptionLevel"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, factor in zip(axes.flat, top):
        sub = rates[rates["factor"] == factor]
        ax.bar(sub["group"], sub["attrition_rate_pct"], color="#2F6FB0")
        ax.axhline(100 * overall, color="#C0392B", linestyle="--", linewidth=1)
        ax.set_title(factor, fontsize=11)
        ax.set_ylabel("Attrition rate (%)")
        ax.set_ylim(0, sub["attrition_rate_pct"].max() * 1.18)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        for x, v in zip(sub["group"], sub["attrition_rate_pct"]):
            ax.text(x, v + 0.6, f"{v:.0f}%", ha="center", fontsize=8)
    fig.suptitle(f"Attrition rate by group (dashed line = company average, {overall:.1%})", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "drivers_chart.png", dpi=130)
    print(f"\nWrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
