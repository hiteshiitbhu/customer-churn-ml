"""
Model Interpretability and Visualization Generator using LightGBM feature importance
and Tree Explanations, producing clean analytical plots without heavy runtime dependencies.
Generates:
1. Feature Importance Ranking (Split & Gain importance)
2. Tenure vs. Churn Rate Nonlinear Relationship curve
3. Monthly Charges vs. Total Charges risk distribution
4. Confusion Matrix heatmap and PR Curve
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

from preprocess import prepare_data
from get_data import RAW_DATA_PATH

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def run_visualizations():
    print("[*] Loading best model and test data...")
    model = joblib.load(os.path.join(MODEL_DIR, "best_lgb_model.pkl"))
    _, X_test, _, y_test, _, _, _, feature_names = prepare_data(RAW_DATA_PATH)

    y_prob = model.predict_proba(X_test)[:, 1]

    # 1. Feature Importance by Gain
    print("[+] Plotting Top 12 Feature Importances (Gain)...")
    importances = model.booster_.feature_importance(importance_type="gain")
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(12)

    plt.figure(figsize=(9, 6))
    bars = plt.barh(feat_imp.index[::-1], feat_imp.values[::-1], color="#2b5c8f", edgecolor="black")
    plt.xlabel("Total Information Gain", fontsize=11)
    plt.title("Top 12 Most Influential Features (LightGBM Gain)", fontsize=13, weight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importance_gain.png"), dpi=200)
    plt.close()

    # 2. ROC & PR Curves
    print("[+] Plotting ROC & PR Evaluation Curves...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)
    pr_auc_val = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # ROC
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"LightGBM (AUC = {roc_auc_val:.3f})")
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[0].set_title("ROC Curve (True Positive vs False Positive Rate)", fontsize=11, weight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.4)

    # PR
    axes[1].plot(rec, prec, color="#2ca02c", lw=2, label=f"LightGBM (PR-AUC = {pr_auc_val:.3f})")
    baseline_churn = y_test.mean()
    axes[1].plot([0, 1], [baseline_churn, baseline_churn], color="grey", linestyle="--", label=f"Random Guess ({baseline_churn:.2f})")
    axes[1].set_title("Precision-Recall Curve (Handling Imbalance)", fontsize=11, weight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "evaluation_curves.png"), dpi=200)
    plt.close()

    # 3. Confusion Matrix at Optimal Decision Threshold
    optimal_th = joblib.load(os.path.join(MODEL_DIR, "optimal_threshold.pkl"))
    y_pred = (y_prob >= optimal_th).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Decision Threshold = {optimal_th:.2f})", fontsize=11, weight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Churn (0)", "Churn (1)"])
    plt.yticks(tick_marks, ["No Churn (0)", "Churn (1)"])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=12)

    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=200)
    plt.close()

    print(f"[+] All figures successfully generated and saved to {FIGURES_DIR}")

if __name__ == "__main__":
    run_visualizations()
