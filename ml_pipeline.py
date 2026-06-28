"""
Predictive Modeling Pipeline
=============================
Models trained:
  Classification  → Logistic Regression | Decision Tree | Random Forest | Gradient Boosting
  Regression      → Linear Regression   | Decision Tree Regressor | Random Forest Regressor

Outputs:
  • model_results.json   — all metrics
  • feature_importance.csv
  • predictions_test.csv
"""

import json, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
warnings.filterwarnings("ignore")

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("/home/claude/churn_dataset.csv")
print(f"Loaded: {df.shape}  |  Churn rate: {df['churn'].mean()*100:.1f}%\n")

# ── Preprocessing ──────────────────────────────────────────────────────────────
NUMERIC_COLS  = ["age","tenure_months","num_products","monthly_charges",
                 "avg_monthly_gb","support_calls","has_tech_support","paperless_billing"]
CATEGORY_COLS = ["contract_type","internet_service","payment_method"]
TARGET_CLF    = "churn"
TARGET_REG    = "monthly_spend"

X = df[NUMERIC_COLS + CATEGORY_COLS]
y_clf = df[TARGET_CLF]
y_reg = df[TARGET_REG]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_COLS),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORY_COLS),
])

# ── Train / Test split ─────────────────────────────────────────────────────────
X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
    X, y_clf, y_reg, test_size=0.20, random_state=42, stratify=y_clf
)
print(f"Train: {X_tr.shape[0]}  |  Test: {X_te.shape[0]}\n")

results = {"classification": {}, "regression": {}}
all_preds = {}

# ══════════════════════════════════════════════════════════════════════════════
#   CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
clf_models = {
    "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":          DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
}

print("=" * 62)
print("  CLASSIFICATION RESULTS")
print("=" * 62)

for name, model in clf_models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    pipe.fit(X_tr, yc_tr)

    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]

    cv = cross_val_score(pipe, X_tr, yc_tr, cv=StratifiedKFold(5), scoring="roc_auc")

    acc  = accuracy_score(yc_te, y_pred)
    prec = precision_score(yc_te, y_pred)
    rec  = recall_score(yc_te, y_pred)
    f1   = f1_score(yc_te, y_pred)
    auc  = roc_auc_score(yc_te, y_proba)
    cm   = confusion_matrix(yc_te, y_pred).tolist()

    results["classification"][name] = {
        "accuracy":   round(acc,  4),
        "precision":  round(prec, 4),
        "recall":     round(rec,  4),
        "f1_score":   round(f1,   4),
        "roc_auc":    round(auc,  4),
        "cv_auc_mean": round(cv.mean(), 4),
        "cv_auc_std":  round(cv.std(),  4),
        "confusion_matrix": cm,
    }

    # Store proba for ROC curves
    all_preds[name] = {"y_pred": y_pred.tolist(), "y_proba": y_proba.tolist()}

    # Feature importance for tree-based
    if hasattr(model, "feature_importances_"):
        ohe_cats = pipe.named_steps["pre"] \
                       .named_transformers_["cat"] \
                       .get_feature_names_out(CATEGORY_COLS).tolist()
        feat_names = NUMERIC_COLS + ohe_cats
        fi = pd.DataFrame({"feature": feat_names,
                            "importance": model.feature_importances_,
                            "model": name})
        fi_path = f"/home/claude/fi_{name.replace(' ','_').lower()}.csv"
        fi.to_csv(fi_path, index=False)

    print(f"\n  {name}")
    print(f"    Accuracy : {acc:.4f}   Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f}   F1-Score : {f1:.4f}")
    print(f"    ROC-AUC  : {auc:.4f}   CV-AUC   : {cv.mean():.4f} ± {cv.std():.4f}")

# Best classifier
best_clf_name = max(results["classification"],
                    key=lambda k: results["classification"][k]["roc_auc"])
print(f"\n  ✅ Best Classifier (ROC-AUC): {best_clf_name} "
      f"→ {results['classification'][best_clf_name]['roc_auc']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#   REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
reg_models = {
    "Linear Regression":      LinearRegression(),
    "Decision Tree Reg":      DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest Reg":      RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42),
}

print("\n" + "=" * 62)
print("  REGRESSION RESULTS  (target: monthly_spend)")
print("=" * 62)

for name, model in reg_models.items():
    pipe = Pipeline([("pre", preprocessor), ("reg", model)])
    pipe.fit(X_tr, yr_tr)

    y_pred = pipe.predict(X_te)
    rmse   = np.sqrt(mean_squared_error(yr_te, y_pred))
    mae    = mean_absolute_error(yr_te, y_pred)
    r2     = r2_score(yr_te, y_pred)
    cv_r2  = cross_val_score(pipe, X_tr, yr_tr, cv=5, scoring="r2")

    results["regression"][name] = {
        "rmse":       round(rmse, 4),
        "mae":        round(mae,  4),
        "r2_score":   round(r2,   4),
        "cv_r2_mean": round(cv_r2.mean(), 4),
        "cv_r2_std":  round(cv_r2.std(),  4),
    }
    all_preds[name] = {"y_true": yr_te.tolist(), "y_pred": y_pred.tolist()}

    print(f"\n  {name}")
    print(f"    RMSE     : {rmse:.4f}   MAE     : {mae:.4f}")
    print(f"    R²       : {r2:.4f}   CV-R²   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# ── Save results ───────────────────────────────────────────────────────────────
# Attach y_true for classification charts
results["y_test_clf"] = yc_te.tolist()
results["y_test_reg"] = yr_te.tolist()
results["all_preds"]  = all_preds

with open("/home/claude/model_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅  model_results.json saved")
print("✅  Feature importance CSVs saved")
