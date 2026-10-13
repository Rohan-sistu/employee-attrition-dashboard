"""
Honest evaluation of an attrition-risk model.

Only 16% of employees leave, so a "model" that predicts "everyone stays" is
already ~84% accurate. Accuracy is therefore the wrong headline metric.
This script reports, on a held-out test set:
  - the do-nothing baseline
  - recall (share of actual leavers the model flags)
  - precision (share of flagged employees who actually leave)
  - ROC-AUC (how well the model ranks leavers above stayers; 0.5 = random)
plus 5-fold cross-validated ROC-AUC to show the result is stable.

Writes outputs/model_metrics.csv and outputs/model_coefficients.csv.
Run from the repository root:  python analysis/model_evaluation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attrition_lib import (OUTPUT_DIR, RANDOM_STATE, build_pipeline,  # noqa: E402
                           features_and_target, load_data)


def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "recall_leavers": recall_score(y_test, pred, zero_division=0),
        "precision_leavers": precision_score(y_test, pred, zero_division=0),
        "f1_leavers": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba) if len(np.unique(proba)) > 1 else 0.5,
        "leavers_flagged": f"{tp} of {tp + fn}",
        "false_alarms": int(fp),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    X, y = features_and_target(load_data())

    # stratify=y keeps the 16% leaver share identical in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    pipeline = build_pipeline(X)
    rf = build_pipeline(X)
    rf.set_params(model=RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               min_samples_leaf=5, random_state=RANDOM_STATE))
    results = [
        evaluate("Baseline: predict everyone stays", DummyClassifier(strategy="most_frequent"),
                 X_train, X_test, y_train, y_test),
        evaluate("Logistic regression (balanced)", pipeline, X_train, X_test, y_train, y_test),
        evaluate("Random forest (balanced)", rf, X_train, X_test, y_train, y_test),
    ]
    metrics = pd.DataFrame(results)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_lr = cross_val_score(build_pipeline(X), X, y, cv=cv, scoring="roc_auc")
    cv_rf = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    metrics["cv_roc_auc_mean"] = [np.nan, cv_lr.mean(), cv_rf.mean()]
    metrics["cv_roc_auc_std"] = [np.nan, cv_lr.std(), cv_rf.std()]

    pd.set_option("display.width", 200)
    print(f"Test set: {len(y_test)} employees, {int(y_test.sum())} leavers\n")
    print(metrics.round(3).to_string(index=False))
    metrics.round(4).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    # Coefficients of the logistic regression (fitted on the training set).
    # Features are scaled, so sizes are comparable: positive = higher leave risk.
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coefs = pd.DataFrame({
        "feature": [n.split("__", 1)[1] for n in names],
        "coefficient": pipeline.named_steps["model"].coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)
    coefs["odds_multiplier"] = np.exp(coefs["coefficient"]).round(2)
    coefs.round(3).to_csv(OUTPUT_DIR / "model_coefficients.csv", index=False)
    print("\nTop 10 logistic regression coefficients (scaled features):")
    print(coefs.head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
