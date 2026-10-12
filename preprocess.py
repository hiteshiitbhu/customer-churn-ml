"""
Feature Engineering and Preprocessing Pipeline.
Performs:
1. Missing value handling (imputing TotalCharges where tenure=0 or spaces).
2. Domain-driven feature transformations.
3. Categorical encoding (One-Hot & Binary).
4. Outlier analysis and Feature scaling.
5. Train/Test split preserving stratification.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_and_feature_engineer(df: pd.DataFrame):
    df = df.copy()

    # Drop identifier
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # 1. Handle TotalCharges (convert spaces to NaN and impute)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    # For tenure == 0, TotalCharges is typically 0 or equal to MonthlyCharges
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"].replace(0, 1))

    # 2. Encode Binary Target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 3. Domain Feature Engineering
    # Tenure in years
    df["TenureYears"] = df["tenure"] / 12.0

    # Charge ratios
    df["AvgMonthlyChargesOverTenure"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # Service bundling count
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["NumServicesSubscribed"] = 0
    for col in service_cols:
        df["NumServicesSubscribed"] += (df[col] == "Yes").astype(int)

    # Security/Support bundle flag
    df["HasSecurityAndSupport"] = (
        (df["OnlineSecurity"] == "Yes") & (df["TechSupport"] == "Yes")
    ).astype(int)

    # Streaming lover flag
    df["HasStreamingBundle"] = (
        (df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes")
    ).astype(int)

    # Tenure segment (Cohort)
    df["TenureCohort"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
    )

    # Normalize categorical values ("No internet service" / "No phone service" -> "No")
    replace_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in replace_cols:
        df[col] = df[col].replace({"No phone service": "No", "No internet service": "No"})

    # Categorical One-Hot Encoding
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df_encoded

def prepare_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    df_raw = pd.read_csv(data_path)
    df_processed = clean_and_feature_engineer(df_raw)

    X = df_processed.drop(columns=["Churn"])
    y = df_processed["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize numerical features for linear models
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "TenureYears", "AvgMonthlyChargesOverTenure", "ChargeRatio", "NumServicesSubscribed"]
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, list(X.columns)

if __name__ == "__main__":
    import os
    from get_data import RAW_DATA_PATH, fetch_or_generate_dataset
    if not os.path.exists(RAW_DATA_PATH):
        fetch_or_generate_dataset()
    X_tr, X_te, y_tr, y_te, _, _, _, cols = prepare_data(RAW_DATA_PATH)
    print(f"[+] Data prepared successfully! Features: {len(cols)}, Train shape: {X_tr.shape}, Churn rate: {y_tr.mean():.3f}")
