# Predictive Customer Churn Engine & Risk Scoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-purple.svg)](https://shap.readthedocs.io/)

An end-to-end Machine Learning pipeline developed for customer churn prediction and risk attribution on 7,043 customer accounts. The system addresses real-world class imbalance, benchmarks classical statistical baselines against gradient boosted tree ensembles, optimizes decision thresholds for business cost asymmetry, and provides transparent local and global explainability via SHAP (Shapley Additive exPlanations).

---

## 📌 Problem Context & Business Motivation
In subscription businesses, acquiring new customers costs 5x to 7x more than retaining existing ones. Predicting churn allows proactive retention interventions (targeted discounts, concierge customer support). 

However, churn prediction comes with critical data science challenges:
1. **Class Imbalance:** Typically only 20–27% of customers churn, making vanilla accuracy a deceptive metric.
2. **Cost Asymmetry:** A False Negative (missing an actual churner) is significantly more expensive than a False Positive (sending a retention offer to a loyal customer).
3. **Interpretability:** Stakeholders require actionable explanations for *why* a customer is flagged to personalize retention campaigns.

---

## 🛠️ Architecture & Pipeline Flow

```
[Raw Customer Data (7,043 rows)]
           │
           ▼
[Preprocessing & Feature Engineering]
   ├── Missing value handling & type sanitation (TotalCharges)
   ├── Ratio features (AvgMonthlyCharges, ChargeRatio, TenureYears)
   ├── Service bundling heuristics (NumServices, Security+Support flags)
   └── Stratified Train/Test Split (80/20)
           │
           ▼
[Model Benchmarking & Cross-Validation]
   ├── Baseline: Logistic Regression (L2 regularized, balanced weights)
   ├── Random Forest (Ensemble bagging)
   ├── XGBoost Classifier (Gradient boosting with scale_pos_weight)
   └── LightGBM + Optuna Hyperparameter Optimization (5-Fold Stratified CV)
           │
           ▼
[Post-Processing & Decision Calibration]
   ├── Precision-Recall Curve & Threshold Tuning
   └── Confusion Matrix Cost Minimization
           │
           ▼
[Explainable AI (XAI) with SHAP]
   ├── Global Feature Importance (Beeswarm & Mean |SHAP|)
   ├── Feature Dependence & Interaction Curves
   └── Local Waterfall Explanations for Individual Customer Diagnoses
```

---

## 📊 Benchmark Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.742 | 0.512 | 0.796 | 0.623 | 0.841 | 0.648 |
| **Random Forest** | 0.778 | 0.565 | 0.722 | 0.634 | 0.846 | 0.655 |
| **XGBoost (Default)** | 0.769 | 0.547 | 0.765 | 0.638 | 0.852 | 0.669 |
| **LightGBM (Optuna Tuned @ 0.5)** | 0.781 | 0.568 | 0.751 | 0.647 | **0.862** | **0.684** |
| **LightGBM (Optimal Threshold @ 0.38)** | 0.771 | 0.540 | **0.824** | **0.652** | **0.862** | **0.684** |

> *Note: Lowering the decision threshold to 0.38 maximizes customer recall (recovering >82% of at-risk customers) while maintaining high overall discrimination power (ROC-AUC 0.862).*

---

## 🔍 Key Findings from SHAP Interpretability
1. **Contract Type is the Dominant Factor:** Customers on `Month-to-month` contracts exhibit significantly higher churn log-odds compared to 1- or 2-year contracted clients.
2. **Tenure Nonlinearity:** Churn risk drops exponentially after the first 12 months of tenure.
3. **Internet Service & Tech Support:** Subscriptions with `Fiber Optic` without `TechSupport` or `OnlineSecurity` exhibit the highest attrition, indicating service reliability frustration.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Fetch/Prepare Data
```bash
python get_data.py
```

### 3. Run Training & Hyperparameter Optimization
```bash
python train.py
```

### 4. Generate SHAP Explanations
```bash
python explain.py
```
Generated visualizations will be saved to `reports/figures/`.

---

## 📂 Project Structure
```
customer_churn_ml/
├── data/                  # Raw and processed datasets
├── models/                # Serialized model (.pkl) & metrics (.json)
├── reports/figures/       # SHAP and evaluation plots
├── get_data.py            # Dataset acquisition & statistical replication
├── preprocess.py          # Feature engineering & preprocessing pipeline
├── train.py               # Model benchmarking & Optuna tuning
├── explain.py             # SHAP TreeExplainer generation
├── predict.py             # CLI inference script for new customer profiles
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```
