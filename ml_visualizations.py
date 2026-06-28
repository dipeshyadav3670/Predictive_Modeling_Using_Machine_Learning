"""
ML Visualization Dashboard
===========================
Produces:
  ml_dashboard.png          — 12-panel master dashboard
  confusion_matrices.png    — 2×2 grid of confusion matrices
  roc_curves.png            — all 4 classifiers on one plot
  regression_analysis.png   — regression diagnostics
  feature_importance.png    — RF & GB importance comparison
"""

import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

# ── Theme ──────────────────────────────────────────────────────────────────────
BG      = "#0D1117"
CARD    = "#161B22"
GRID    = "#21262D"
TEXT    = "#E6EDF3"
MUTED   = "#8B949E"
P       = ["#4361EE","#F72585","#06D6A0","#FFB703","#4CC9F0","#7209B7","#FB8500","#3A0CA3"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "grid.color": GRID, "text.color": TEXT,
    "legend.facecolor": CARD, "legend.edgecolor": GRID,
    "font.family": "DejaVu Sans",
})

# ── Load results ───────────────────────────────────────────────────────────────
with open("/home/claude/model_results.json") as f:
    res = json.load(f)

clf_names  = list(res["classification"].keys())
reg_names  = list(res["regression"].keys())
y_true_clf = np.array(res["y_test_clf"])
y_true_reg = np.array(res["y_test_reg"])
preds      = res["all_preds"]

# ── Feature importance dataframes ──────────────────────────────────────────────
fi_rf = pd.read_csv("/home/claude/fi_random_forest.csv")
fi_gb = pd.read_csv("/home/claude/fi_gradient_boosting.csv")

# ══════════════════════════════════════════════════════════════════════════════
#  1. MASTER DASHBOARD  (4 × 3)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(24, 28), facecolor=BG)
fig.suptitle("🤖  Predictive ML Dashboard — Customer Churn Analysis",
             fontsize=26, fontweight="bold", color=TEXT, y=0.99)
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38,
                        top=0.95, bottom=0.03, left=0.06, right=0.97)

# ── Row 0 : KPI Cards ─────────────────────────────────────────────────────────
kpi_ax = fig.add_subplot(gs[0, :])
kpi_ax.set_facecolor(BG); kpi_ax.axis("off")

best_name = max(res["classification"], key=lambda k: res["classification"][k]["roc_auc"])
best      = res["classification"][best_name]
kpis = [
    ("Best Model",      best_name.replace(" ","\n"),        P[0]),
    ("ROC-AUC",         f"{best['roc_auc']:.4f}",           P[1]),
    ("F1-Score",        f"{best['f1_score']:.4f}",          P[2]),
    ("CV-AUC (5-fold)", f"{best['cv_auc_mean']:.4f}±{best['cv_auc_std']:.4f}", P[3]),
    ("Reg. Best R²",    f"{max(v['r2_score'] for v in res['regression'].values()):.4f}", P[4]),
]
for i, (label, value, color) in enumerate(kpis):
    x = 0.01 + i * 0.198
    rect = FancyBboxPatch((x, 0.08), 0.185, 0.82,
                          boxstyle="round,pad=0.015",
                          facecolor=CARD, edgecolor=color,
                          linewidth=2.5, transform=kpi_ax.transAxes, zorder=2)
    kpi_ax.add_patch(rect)
    kpi_ax.text(x+0.0925, 0.72, label, ha="center", va="center",
                fontsize=9.5, color=MUTED, transform=kpi_ax.transAxes)
    kpi_ax.text(x+0.0925, 0.32, value, ha="center", va="center",
                fontsize=13.5, fontweight="bold", color=color,
                transform=kpi_ax.transAxes)

# ── Row 1, Col 0 : Model Accuracy Comparison ──────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
metrics_data = {
    n: [res["classification"][n][m]
        for m in ["accuracy","precision","recall","f1_score","roc_auc"]]
    for n in clf_names
}
x = np.arange(5)
metric_labels = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
width = 0.20
for i, (name, vals) in enumerate(metrics_data.items()):
    ax.bar(x + i*width, vals, width, label=name.split()[0], color=P[i], alpha=0.88)
ax.set_xticks(x + width*1.5); ax.set_xticklabels(metric_labels, fontsize=8)
ax.set_ylim(0.5, 1.02); ax.set_ylabel("Score")
ax.set_title("Classifier Metrics Comparison", fontweight="bold", pad=10)
ax.legend(fontsize=7, ncol=2)
ax.axhline(0.8, color=MUTED, linestyle="--", linewidth=0.7, alpha=0.6)

# ── Row 1, Col 1 : ROC Curves ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
for i, name in enumerate(clf_names):
    y_proba = np.array(preds[name]["y_proba"])
    fpr, tpr, _ = roc_curve(y_true_clf, y_proba)
    roc_auc_val = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color=P[i], lw=2,
             label=f"{name.split()[0]} (AUC={roc_auc_val:.3f})")
ax2.plot([0,1],[0,1], color=MUTED, lw=1, linestyle="--", label="Random")
ax2.fill_between([0,1],[0,1], alpha=0.05, color=MUTED)
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curves — All Classifiers", fontweight="bold", pad=10)
ax2.legend(fontsize=7.5, loc="lower right")

# ── Row 1, Col 2 : CV Score Distribution ─────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
cv_means = [res["classification"][n]["cv_auc_mean"] for n in clf_names]
cv_stds  = [res["classification"][n]["cv_auc_std"]  for n in clf_names]
short = [n.split()[0] for n in clf_names]
bars = ax3.bar(short, cv_means, color=P[:4], alpha=0.85, width=0.5)
ax3.errorbar(short, cv_means, yerr=cv_stds, fmt="none",
             color=TEXT, capsize=5, linewidth=2)
for bar, m, s in zip(bars, cv_means, cv_stds):
    ax3.text(bar.get_x()+bar.get_width()/2, m+s+0.005,
             f"{m:.3f}", ha="center", fontsize=8, color=TEXT)
ax3.set_ylim(0.6, 1.0)
ax3.set_ylabel("CV ROC-AUC (5-fold)")
ax3.set_title("Cross-Validation Stability", fontweight="bold", pad=10)
ax3.tick_params(axis="x", labelsize=8)

# ── Row 2, Cols 0-1 : Confusion Matrices ─────────────────────────────────────
for col, name in enumerate(clf_names[:2]):
    ax_cm = fig.add_subplot(gs[2, col])
    y_pred = np.array(preds[name]["y_pred"])
    cm     = confusion_matrix(y_true_clf, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm, annot=False, ax=ax_cm, cmap="Blues",
                linewidths=2, linecolor=BG, cbar=False)
    for r in range(2):
        for c in range(2):
            clr = TEXT if cm[r,c] < cm.max()*0.6 else BG
            ax_cm.text(c+0.5, r+0.38, str(cm[r,c]),
                       ha="center", va="center", fontsize=16, fontweight="bold", color=clr)
            ax_cm.text(c+0.5, r+0.65, f"({cm_pct[r,c]:.1f}%)",
                       ha="center", va="center", fontsize=9, color=clr)
    ax_cm.set_xticklabels(["Stayed","Churned"], fontsize=9)
    ax_cm.set_yticklabels(["Stayed","Churned"], fontsize=9, rotation=0)
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"Confusion Matrix\n{name}", fontweight="bold", pad=8)

# ── Row 2, Col 2 : Confusion Matrices for RF & GB ────────────────────────────
for col, name in enumerate(clf_names[2:], start=0):
    ax_cm2 = fig.add_subplot(gs[2, 2]) if col == 0 else None
    if col == 0:
        y_pred = np.array(preds[name]["y_pred"])
        cm     = confusion_matrix(y_true_clf, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm, annot=False, ax=ax_cm2, cmap="Purples",
                    linewidths=2, linecolor=BG, cbar=False)
        for r in range(2):
            for c in range(2):
                clr = TEXT if cm[r,c] < cm.max()*0.6 else BG
                ax_cm2.text(c+0.5, r+0.38, str(cm[r,c]),
                            ha="center", va="center", fontsize=16, fontweight="bold", color=clr)
                ax_cm2.text(c+0.5, r+0.65, f"({cm_pct[r,c]:.1f}%)",
                            ha="center", va="center", fontsize=9, color=clr)
        ax_cm2.set_xticklabels(["Stayed","Churned"], fontsize=9)
        ax_cm2.set_yticklabels(["Stayed","Churned"], fontsize=9, rotation=0)
        ax_cm2.set_xlabel("Predicted"); ax_cm2.set_ylabel("Actual")
        ax_cm2.set_title(f"Confusion Matrix\n{name}", fontweight="bold", pad=8)

# ── Row 3, Col 0 : Feature Importance (RF) ────────────────────────────────────
ax_fi = fig.add_subplot(gs[3, 0])
fi_top = fi_rf.sort_values("importance", ascending=True).tail(10)
colors_fi = [P[1] if i >= 7 else P[0] for i in range(len(fi_top))]
ax_fi.barh(fi_top["feature"], fi_top["importance"], color=colors_fi, height=0.6)
ax_fi.set_title("Feature Importance (Random Forest)", fontweight="bold", pad=10)
ax_fi.set_xlabel("Importance")
ax_fi.tick_params(axis="y", labelsize=7)

# ── Row 3, Col 1 : Regression R² Comparison ───────────────────────────────────
ax_reg = fig.add_subplot(gs[3, 1])
r2_vals  = [res["regression"][n]["r2_score"]  for n in reg_names]
rmse_vals= [res["regression"][n]["rmse"]       for n in reg_names]
short_reg = ["Linear\nReg", "Decision\nTree", "Random\nForest"]
x_pos = np.arange(len(reg_names))
ax_reg.bar(x_pos - 0.2, r2_vals, 0.35, color=P[2], label="R²", alpha=0.88)
ax_reg2 = ax_reg.twinx()
ax_reg2.bar(x_pos + 0.2, rmse_vals, 0.35, color=P[3], label="RMSE", alpha=0.88)
ax_reg2.set_ylabel("RMSE", color=P[3])
ax_reg2.tick_params(axis="y", colors=P[3])
ax_reg.set_xticks(x_pos); ax_reg.set_xticklabels(short_reg, fontsize=8)
ax_reg.set_ylabel("R² Score", color=P[2])
ax_reg.tick_params(axis="y", colors=P[2])
ax_reg.set_title("Regression Model Comparison", fontweight="bold", pad=10)
lines1, labs1 = ax_reg.get_legend_handles_labels()
lines2, labs2 = ax_reg2.get_legend_handles_labels()
ax_reg.legend(lines1+lines2, labs1+labs2, fontsize=8, loc="upper left")

# ── Row 3, Col 2 : Actual vs Predicted (RF Regressor) ────────────────────────
ax_avp = fig.add_subplot(gs[3, 2])
y_true_r = np.array(res["all_preds"]["Random Forest Reg"]["y_true"])
y_pred_r = np.array(res["all_preds"]["Random Forest Reg"]["y_pred"])
sc = ax_avp.scatter(y_true_r, y_pred_r, alpha=0.35, s=15,
                    c=np.abs(y_true_r - y_pred_r), cmap="plasma", edgecolors="none")
mn = min(y_true_r.min(), y_pred_r.min())
mx = max(y_true_r.max(), y_pred_r.max())
ax_avp.plot([mn, mx], [mn, mx], color=P[2], lw=1.5, linestyle="--", label="Perfect fit")
plt.colorbar(sc, ax=ax_avp, label="|Error|", fraction=0.04)
ax_avp.set_xlabel("Actual Spend (₹)"); ax_avp.set_ylabel("Predicted Spend (₹)")
ax_avp.set_title("Actual vs Predicted\n(RF Regressor)", fontweight="bold", pad=8)
ax_avp.legend(fontsize=8)

fig.savefig("/mnt/user-data/outputs/ml_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("✅  ml_dashboard.png")

# ══════════════════════════════════════════════════════════════════════════════
#  2. STANDALONE ROC CURVES (high-res)
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
for i, name in enumerate(clf_names):
    y_proba = np.array(preds[name]["y_proba"])
    fpr, tpr, _ = roc_curve(y_true_clf, y_proba)
    roc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=P[i], lw=2.5,
            label=f"{name}  (AUC = {roc_val:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.06, color=P[i])
ax.plot([0,1],[0,1], color=MUTED, lw=1.5, linestyle="--", label="No-skill baseline")
ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
ax.set_title("ROC Curves — All Classifiers", fontsize=15, fontweight="bold", pad=14)
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig("/mnt/user-data/outputs/roc_curves.png",
             dpi=150, bbox_inches="tight", facecolor=BG)
print("✅  roc_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
#  3. CONFUSION MATRICES — 2×2 grid
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10), facecolor=BG)
fig3.suptitle("Confusion Matrices — All Classifiers",
              fontsize=16, fontweight="bold", color=TEXT, y=1.01)
cmaps = ["Blues","Reds","Greens","Purples"]
for ax_c, name, cmap in zip(axes3.flatten(), clf_names, cmaps):
    y_pred = np.array(preds[name]["y_pred"])
    cm     = confusion_matrix(y_true_clf, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm, annot=False, ax=ax_c, cmap=cmap,
                linewidths=2, linecolor=BG, cbar=True,
                cbar_kws={"shrink": 0.75})
    for r in range(2):
        for c in range(2):
            clr = TEXT if cm[r,c] < cm.max()*0.6 else BG
            ax_c.text(c+0.5, r+0.35, str(cm[r,c]),
                      ha="center", va="center", fontsize=20, fontweight="bold", color=clr)
            ax_c.text(c+0.5, r+0.65, f"({cm_pct[r,c]:.1f}%)",
                      ha="center", va="center", fontsize=11, color=clr)
    ax_c.set_xticklabels(["Stayed","Churned"], fontsize=10)
    ax_c.set_yticklabels(["Stayed","Churned"], fontsize=10, rotation=0)
    ax_c.set_xlabel("Predicted Label"); ax_c.set_ylabel("True Label")
    ax_c.set_title(name, fontsize=12, fontweight="bold", pad=8)

plt.tight_layout()
fig3.savefig("/mnt/user-data/outputs/confusion_matrices.png",
             dpi=150, bbox_inches="tight", facecolor=BG)
print("✅  confusion_matrices.png")

# ══════════════════════════════════════════════════════════════════════════════
#  4. FEATURE IMPORTANCE — RF vs Gradient Boosting
# ══════════════════════════════════════════════════════════════════════════════
fig4, (axL, axR) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig4.suptitle("Feature Importance Comparison",
              fontsize=16, fontweight="bold", color=TEXT, y=1.01)

for ax_fi2, fi_df, name, color in [
        (axL, fi_rf, "Random Forest", P[0]),
        (axR, fi_gb, "Gradient Boosting", P[1])]:
    top = fi_df.sort_values("importance", ascending=True).tail(12)
    bar_colors = [color if imp >= top["importance"].quantile(0.7) else P[6]
                  for imp in top["importance"]]
    ax_fi2.barh(top["feature"], top["importance"] * 100,
                color=bar_colors, height=0.65)
    for i, (feat, imp) in enumerate(zip(top["feature"], top["importance"])):
        ax_fi2.text(imp * 100 + 0.1, i, f"{imp*100:.2f}%",
                    va="center", fontsize=8, color=TEXT)
    ax_fi2.set_title(name, fontsize=13, fontweight="bold", pad=10)
    ax_fi2.set_xlabel("Importance (%)")
    ax_fi2.tick_params(axis="y", labelsize=8)

plt.tight_layout()
fig4.savefig("/mnt/user-data/outputs/feature_importance.png",
             dpi=150, bbox_inches="tight", facecolor=BG)
print("✅  feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
#  5. REGRESSION DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(2, 2, figsize=(14, 11), facecolor=BG)
fig5.suptitle("Regression Diagnostics — Random Forest Regressor",
              fontsize=16, fontweight="bold", color=TEXT, y=1.01)

residuals = y_true_r - y_pred_r

# A — Actual vs Predicted
a = axes5[0, 0]
sc2 = a.scatter(y_true_r, y_pred_r, alpha=0.4, s=18,
                c=np.abs(residuals), cmap="RdYlGn_r", edgecolors="none")
a.plot([mn,mx],[mn,mx], color=P[2], lw=2, linestyle="--", label="y=x")
plt.colorbar(sc2, ax=a, label="|Residual|", fraction=0.04)
a.set_xlabel("Actual"); a.set_ylabel("Predicted")
a.set_title("Actual vs Predicted", fontweight="bold", pad=8)
a.legend(fontsize=8)

# B — Residual Distribution
b = axes5[0, 1]
b.hist(residuals, bins=40, color=P[0], alpha=0.8, edgecolor="none")
b.axvline(0, color=P[1], lw=2, linestyle="--", label="Zero error")
b.axvline(residuals.mean(), color=P[2], lw=2, linestyle="-.", label=f"Mean={residuals.mean():.2f}")
b.set_xlabel("Residual"); b.set_ylabel("Count")
b.set_title("Residual Distribution", fontweight="bold", pad=8)
b.legend(fontsize=8)

# C — Residuals vs Fitted
c = axes5[1, 0]
c.scatter(y_pred_r, residuals, alpha=0.35, s=15, color=P[4], edgecolors="none")
c.axhline(0, color=P[1], lw=1.5, linestyle="--")
z = np.polyfit(y_pred_r, residuals, 1)
p_fn = np.poly1d(z)
x_line = np.linspace(y_pred_r.min(), y_pred_r.max(), 200)
c.plot(x_line, p_fn(x_line), color=P[3], lw=1.5, label="Trend")
c.set_xlabel("Fitted Values"); c.set_ylabel("Residuals")
c.set_title("Residuals vs Fitted", fontweight="bold", pad=8)
c.legend(fontsize=8)

# D — Model R² Comparison bar chart
d = axes5[1, 1]
reg_data = {n: res["regression"][n] for n in reg_names}
r2s  = [v["r2_score"]  for v in reg_data.values()]
maes = [v["mae"]       for v in reg_data.values()]
short_names = ["Linear\nReg", "Decision\nTree", "Random\nForest"]
x_d = np.arange(len(reg_names))
d.bar(x_d - 0.18, r2s,  0.32, color=P[2], label="R²",   alpha=0.88)
d2  = d.twinx()
d2.bar(x_d + 0.18, maes, 0.32, color=P[3], label="MAE",  alpha=0.88)
d.set_xticks(x_d); d.set_xticklabels(short_names, fontsize=9)
d.set_ylabel("R² Score", color=P[2]); d2.set_ylabel("MAE (₹)", color=P[3])
d.tick_params(axis="y", colors=P[2]); d2.tick_params(axis="y", colors=P[3])
d.set_title("Regression Model R² vs MAE", fontweight="bold", pad=8)
h1,l1 = d.get_legend_handles_labels(); h2,l2 = d2.get_legend_handles_labels()
d.legend(h1+h2, l1+l2, fontsize=8)

plt.tight_layout()
fig5.savefig("/mnt/user-data/outputs/regression_diagnostics.png",
             dpi=150, bbox_inches="tight", facecolor=BG)
print("✅  regression_diagnostics.png")

print("\n🎉  All ML visualizations complete!")
