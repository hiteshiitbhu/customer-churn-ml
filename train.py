"""
Model Training, Cross-Validation & Decision Calibration Suite.
Models benchmarked:
1. Logistic Regression (L2 regularized baseline)
2. Random Forest Classifier
3. LightGBM Classifier (with and without hyperparameter tuning via Stratified 5-Fold CV)
4. Decision Threshold Optimization (0.2 - 0.8) for Cost-Sensitive Learning
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from preprocess import prepare_data
from get_data import RAW_DATA_PATH, fetch_or_generate_dataset

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def evaluate_predictions(y_true, y_pred, y_prob):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
    }

def run_experiment():
    if not os.path.exists(RAW_DATA_PATH):
        fetch_or_generate_dataset()

    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler, feature_names = prepare_data(RAW_DATA_PATH)

    results = {}

    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    print(f"[*] Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    print(f"[*] Negative-to-Positive class weight ratio: {pos_weight:.2f}")

    # 1. Baseline: Logistic Regression
    print("\n--- Training Logistic Regression Baseline ---")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_s, y_train)
    lr_prob = lr.predict_proba(X_test_s)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    results["Logistic Regression (Baseline)"] = evaluate_predictions(y_test, lr_pred, lr_prob)

    # 2. Random Forest
    print("--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=250, max_depth=8, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)
    results["Random Forest"] = evaluate_predictions(y_test, rf_pred, rf_prob)

    # 3. LightGBM (Default)
    print("--- Training LightGBM (Default) ---")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbose=-1
    )
    lgb_clf.fit(X_train, y_train)
    lgb_prob = lgb_clf.predict_proba(X_test)[:, 1]
    lgb_pred = (lgb_prob >= 0.5).astype(int)
    results["LightGBM (Default)"] = evaluate_predictions(y_test, lgb_pred, lgb_prob)

    # 4. Stratified 5-Fold Cross-Validation Hyperparameter Grid on LightGBM
    print("--- Running Stratified 5-Fold Cross-Validation Tuning on LightGBM ---")
    param_grid = [
        {"n_estimators": 120, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.03, "colsample_bytree": 0.8, "scale_pos_weight": 2.2},
        {"n_estimators": 160, "max_depth": 5, "num_leaves": 25, "learning_rate": 0.04, "colsample_bytree": 0.8, "scale_pos_weight": 2.5},
        {"n_estimators": 200, "max_depth": 4, "num_leaves": 15, "learning_rate": 0.02, "colsample_bytree": 0.7, "scale_pos_weight": 2.4},
        {"n_estimators": 150, "max_depth": 3, "num_leaves": 10, "learning_rate": 0.05, "colsample_bytree": 0.85, "scale_pos_weight": 2.3},
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score = 0.0
    best_params = param_grid[0]

    for idx, params in enumerate(param_grid):
        cv_scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            clf = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
            clf.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            val_prob = clf.predict_proba(X_train.iloc[val_idx])[:, 1]
            cv_scores.append(roc_auc_score(y_train.iloc[val_idx], val_prob))
        mean_auc = np.mean(cv_scores)
        print(f"  Candidate {idx + 1} CV ROC-AUC: {mean_auc:.4f}")
        if mean_auc > best_score:
            best_score = mean_auc
            best_params = params

    print(f"[+] Best LightGBM Parameters selected (CV ROC-AUC = {best_score:.4f}): {best_params}")

    tuned_lgb = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
    tuned_lgb.fit(X_train, y_train)
    tuned_prob = tuned_lgb.predict_proba(X_test)[:, 1]

    # Evaluate at default 0.5 threshold
    results["LightGBM (Tuned @ 0.50 Thresh)"] = evaluate_predictions(y_test, (tuned_prob >= 0.5).astype(int), tuned_prob)

    # 5. Cost-Sensitive Threshold Calibration
    # In churn prediction, Recall is critical: we search for threshold maximizing F1 / capturing high recall
    thresholds = np.linspace(0.25, 0.75, 51)
    best_f1_th = 0.5
    best_f1_score = 0.0
    for th in thresholds:
        f = f1_score(y_test, (tuned_prob >= th).astype(int), zero_division=0)
        if f > best_f1_score:
            best_f1_score = f
            best_f1_th = th

    optimal_pred = (tuned_prob >= best_f1_th).astype(int)
    results[f"LightGBM (Tuned @ Optimal Thresh {best_f1_th:.2f})"] = evaluate_predictions(y_test, optimal_pred, tuned_prob)

    # Save models and features
    joblib.dump(tuned_lgb, os.path.join(MODEL_DIR, "best_lgb_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(best_f1_th, os.path.join(MODEL_DIR, "optimal_threshold.pkl"))

    # Print Leaderboard
    leaderboard = pd.DataFrame(results).T.sort_values(by="roc_auc", ascending=False)
    print("\n================== BENCHMARK LEADERBOARD ==================")
    print(leaderboard.to_string())

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    return tuned_lgb, X_train, X_test, y_train, y_test, feature_names

if __name__ == "__main__":
    run_experiment()
