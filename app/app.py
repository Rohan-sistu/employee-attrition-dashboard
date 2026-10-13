"""
Employee Attrition Analytics Dashboard (Streamlit).

1. Who leaves?  Attrition rate by any factor, filterable by department.
2. How good is the risk model?  Honest test-set metrics vs. a baseline.
3. Risk estimate for a hypothetical employee, starting from a typical profile.

Run from the repository root:  streamlit run app/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attrition_lib import (RANDOM_STATE, add_bands, build_pipeline,  # noqa: E402
                           features_and_target, load_data)

st.set_page_config(page_title="Employee Attrition Analytics", layout="wide")

FACTOR_LABELS = {
    "OverTime": "Works overtime",
    "TenureBand": "Years at company",
    "JobLevel": "Job level (1 = entry)",
    "IncomeQuartile": "Monthly income quartile",
    "AgeBand": "Age band",
    "MaritalStatus": "Marital status",
    "StockOptionLevel": "Stock option level",
    "BusinessTravel": "Business travel",
    "JobRole": "Job role",
    "JobSatisfaction": "Job satisfaction (1-4)",
    "EnvironmentSatisfaction": "Environment satisfaction (1-4)",
    "WorkLifeBalance": "Work-life balance (1-4)",
    "CommuteBand": "Distance from home",
    "Gender": "Gender",
    "PerformanceRating": "Performance rating",
}


@st.cache_data
def get_data() -> pd.DataFrame:
    return add_bands(load_data())


@st.cache_resource
def get_model_and_metrics():
    X, y = features_and_target(load_data())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    model = build_pipeline(X).fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "baseline_accuracy": 1 - y_test.mean(),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba),
        "tp": tp, "fn": fn, "fp": fp,
    }
    # Refit on all data for the what-if tool, using the evaluated settings.
    final_model = build_pipeline(X).fit(X, y)
    return final_model, metrics, X


df = get_data()
model, metrics, X_all = get_model_and_metrics()

st.title("Employee Attrition Analytics")
st.caption("IBM HR Analytics dataset (fictional data created by IBM for teaching): "
           "1,470 employees, one snapshot in time.")

# ---------------------------------------------------------------- filters
departments = ["All departments"] + sorted(df["Department"].unique())
dept = st.sidebar.selectbox("Department", departments)
view = df if dept == "All departments" else df[df["Department"] == dept]

# ---------------------------------------------------------------- KPIs
company_rate = df["Left"].mean()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Employees", f"{len(view):,}")
k2.metric("Left the company", f"{int(view['Left'].sum()):,}")
k3.metric("Attrition rate", f"{view['Left'].mean():.1%}",
          delta=f"{(view['Left'].mean() - company_rate) * 100:+.1f} pts vs company"
          if dept != "All departments" else None, delta_color="inverse")
ot = view[view["OverTime"] == "Yes"]["Left"].mean()
no_ot = view[view["OverTime"] == "No"]["Left"].mean()
k4.metric("Overtime vs. no overtime", f"{ot:.0%} vs {no_ot:.0%}")

# ---------------------------------------------------------------- 1. drivers
st.header("1. Who leaves?")
factor = st.selectbox("Compare attrition rate by", list(FACTOR_LABELS),
                      format_func=FACTOR_LABELS.get)
rates = (view.groupby(factor, observed=True)["Left"]
         .agg(employees="count", left="sum", rate="mean").reset_index())
rates["rate_pct"] = (100 * rates["rate"]).round(1)
rates[factor] = rates[factor].astype(str)
fig = px.bar(rates, x=factor, y="rate_pct", text="rate_pct",
             hover_data={"employees": True, "left": True, "rate_pct": False},
             labels={factor: FACTOR_LABELS[factor], "rate_pct": "Attrition rate (%)"})
fig.add_hline(y=100 * view["Left"].mean(), line_dash="dash", line_color="#C0392B",
              annotation_text="average", annotation_position="top left")
fig.update_traces(marker_color="#2F6FB0", texttemplate="%{text:.1f}%", textposition="outside")
fig.update_layout(yaxis_range=[0, rates["rate_pct"].max() * 1.2], margin=dict(t=30))
st.plotly_chart(fig, width="stretch")
small = rates[rates["employees"] < 30]
if not small.empty:
    st.caption("Groups with fewer than 30 employees: "
               + ", ".join(small[factor]) + " (treat their rates with caution).")

st.markdown(
    "**Key findings (whole company):** employees working overtime leave at 30.5% vs. 10.4%; "
    "entry-level staff on overtime at 52.6%; first-year employees at 34.9%; Sales Representatives "
    "at 39.8%. Gender, education and performance rating show **no** significant difference."
)

# ---------------------------------------------------------------- 2. model quality
st.header("2. How reliable is the risk model?")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Leavers correctly flagged", f"{metrics['tp']} of {metrics['tp'] + metrics['fn']}",
          f"recall {metrics['recall']:.0%}", delta_color="off")
m2.metric("Flagged who actually left", f"{metrics['precision']:.0%}", "precision", delta_color="off")
m3.metric("ROC-AUC", f"{metrics['auc']:.2f}", "0.5 = random guessing", delta_color="off")
m4.metric("'Nobody leaves' baseline accuracy", f"{metrics['baseline_accuracy']:.0%}",
          "why accuracy is not used", delta_color="off")
st.caption(f"Logistic regression with class balancing, evaluated on a held-out 25% test set "
           f"(stratified). It produces {metrics['fp']} false alarms, so it is a screening aid "
           f"for HR conversations, not a verdict on individuals.")

# ---------------------------------------------------------------- 3. what-if
st.header("3. What-if: estimated risk for an employee profile")
st.caption("Starts from a typical employee (median for numbers, most common value for "
           "categories). Change the factors below to see how the estimate moves.")

typical = {c: (X_all[c].median() if pd.api.types.is_numeric_dtype(X_all[c]) else X_all[c].mode()[0])
           for c in X_all.columns}
c1, c2, c3 = st.columns(3)
with c1:
    typical["OverTime"] = st.radio("Works overtime", ["No", "Yes"], horizontal=True)
    typical["JobLevel"] = st.slider("Job level", 1, 5, int(typical["JobLevel"]))
    typical["YearsAtCompany"] = st.slider("Years at company", 0, 40, int(typical["YearsAtCompany"]))
with c2:
    typical["MonthlyIncome"] = st.slider("Monthly income", 1000, 20000, int(typical["MonthlyIncome"]), step=250)
    typical["Age"] = st.slider("Age", 18, 60, int(typical["Age"]))
    typical["MaritalStatus"] = st.selectbox("Marital status", ["Single", "Married", "Divorced"],
                                            index=["Single", "Married", "Divorced"].index(typical["MaritalStatus"]))
with c3:
    typical["JobSatisfaction"] = st.slider("Job satisfaction (1-4)", 1, 4, int(typical["JobSatisfaction"]))
    typical["StockOptionLevel"] = st.slider("Stock option level", 0, 3, int(typical["StockOptionLevel"]))
    roles = sorted(X_all["JobRole"].unique())
    typical["JobRole"] = st.selectbox("Job role", roles, index=roles.index(typical["JobRole"]))

profile = pd.DataFrame([typical])[X_all.columns]
risk = model.predict_proba(profile)[0, 1]
st.metric("Estimated attrition risk score", f"{risk:.0%}")
st.caption("Because the model is trained with class balancing, scores are useful for ranking "
           "employees by relative risk; they overstate the true probability of leaving "
           f"(the company-wide rate is {company_rate:.0%}).")
