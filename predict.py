"""
Interactive CLI Inference Tool for Customer Churn Scoring.
Takes arbitrary customer features and prints:
- Churn probability & predicted category
- Risk Level (Low, Medium, High)
- Personalized retention recommendation
"""

import os
import joblib
import pandas as pd
from preprocess import clean_and_feature_engineer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def predict_customer(sample_dict: dict):
    model_path = os.path.join(MODEL_DIR, "best_lgb_model.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_names.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Run train.py first!")

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)

    # Convert to DataFrame
    df = pd.DataFrame([sample_dict])
    df["Churn"] = "No"  # dummy for pipeline
    df_processed = clean_and_feature_engineer(df).drop(columns=["Churn"])

    # Align columns
    for col in feature_names:
        if col not in df_processed.columns:
            df_processed[col] = 0
    df_processed = df_processed[feature_names]

    prob = model.predict_proba(df_processed)[0, 1]
    
    if prob >= 0.65:
        risk = "HIGH RISK"
        action = "Immediate retention outreach required (Discount offer + Assigned Account Executive)"
    elif prob >= 0.35:
        risk = "MODERATE RISK"
        action = "Automated email campaign promoting annual contract savings & bundled tech support"
    else:
        risk = "LOW RISK"
        action = "Standard loyalty engagement"

    return {
        "churn_probability": round(float(prob), 4),
        "risk_level": risk,
        "recommended_action": action
    }

if __name__ == "__main__":
    test_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": "256.50"
    }
    
    print("Evaluating Test Customer Profile...")
    try:
        res = predict_customer(test_customer)
        print(f"\nResult:")
        print(f"  Churn Probability: {res['churn_probability'] * 100:.1f}%")
        print(f"  Risk Category:     {res['risk_level']}")
        print(f"  Action Item:       {res['recommended_action']}")
    except Exception as e:
        print(f"Prediction error: {e}")
