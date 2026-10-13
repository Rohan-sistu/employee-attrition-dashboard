"""
Shared data loading, feature preparation and model pipeline.

Used by analysis/drivers.py, analysis/model_evaluation.py and app/app.py,
so the dashboard and the written results always use the same logic.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_DIR = ROOT / "outputs"

# Columns with one value for every employee, or a pure ID: no information.
DROP_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

# Text categories with no natural order -> one-hot encoded.
# (Label-encoding these would invent an order, e.g. Sales > R&D > HR.)
NOMINAL_COLUMNS = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime",
]

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load the raw IBM HR dataset and add a 0/1 target column `Left`."""
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=DROP_COLUMNS)
    df["Left"] = (df["Attrition"] == "Yes").astype(int)
    return df


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add readable groupings used in the driver analysis and dashboard."""
    df = df.copy()
    df["TenureBand"] = pd.cut(df["YearsAtCompany"], [-1, 1, 3, 5, 10, 40],
                              labels=["0-1 yrs", "2-3 yrs", "4-5 yrs", "6-10 yrs", "11+ yrs"])
    df["AgeBand"] = pd.cut(df["Age"], [17, 25, 30, 35, 45, 60],
                           labels=["18-25", "26-30", "31-35", "36-45", "46-60"])
    df["IncomeQuartile"] = pd.qcut(df["MonthlyIncome"], 4,
                                   labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    df["CommuteBand"] = pd.cut(df["DistanceFromHome"], [0, 5, 10, 30],
                               labels=["0-5 km", "6-10 km", "11+ km"])
    return df


def features_and_target(df: pd.DataFrame):
    X = df.drop(columns=["Attrition", "Left"])
    y = df["Left"]
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """One-hot nominal columns, scale numeric ones, then logistic regression.

    class_weight="balanced" makes the model treat a missed leaver as costly
    as a false alarm; without it, the model learns that predicting "stays"
    is almost always right (84% of employees stay) and finds few leavers.
    """
    numeric = [c for c in X.columns if c not in NOMINAL_COLUMNS]
    preprocess = ColumnTransformer([
        ("nominal", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), NOMINAL_COLUMNS),
        ("numeric", StandardScaler(), numeric),
    ])
    model = LogisticRegression(class_weight="balanced", max_iter=5000)
    return Pipeline([("prep", preprocess), ("model", model)])
