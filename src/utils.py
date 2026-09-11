# src/utils.py
# Shared utilities for the Credit Risk Scorer API.
# Handles model loading, scaler loading, feature definitions,
# and Groq client setup. Imported by predict.py and agent.py.

import os
import joblib
import pandas as pd
import numpy  as np

from pathlib    import Path
from dotenv     import load_dotenv
from groq       import Groq

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# File paths — all relative to project root
MODEL_PATH  = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

# Feature definitions
# These are the exact 23 raw input features a user provides.
# They match the original dataset columns before engineering.

RAW_FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0",     "PAY_2", "PAY_3",   "PAY_4",   "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",
    "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6"
]

# Columns that get scaled by StandardScaler
SCALE_COLS = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",
    "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6",
    "credit_utilization_rate",
    "debt_to_income_ratio",
    "avg_payment_delay",
    "total_bill_amt",
    "total_pay_amt",
    "payment_to_bill_ratio"
]

# Groq model selection
# Tries each preferred model in order.
# Falls back to first available if none match.

PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "groq/compound",
    "groq/compound-mini"
]


def load_model():
    """
    Loads the trained Random Forest model from disk.
    Returns the model object ready for prediction.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Run Phase 3 notebook first."
        )
    return joblib.load(MODEL_PATH)


def load_scaler():
    """
    Loads the fitted StandardScaler from disk.
    Returns the scaler object ready for transforming input.
    """
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. "
            f"Run Phase 2 notebook first."
        )
    return joblib.load(SCALER_PATH)


def get_groq_client():
    """
    Creates and returns a Groq API client.
    Reads API key from .env file.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env file. "
            "Add your key to the .env file."
        )
    return Groq(api_key=api_key)


def get_best_groq_model(client: Groq) -> str:
    """
    Fetches available Groq models and returns
    the best one from our preferred list.
    Falls back to first available model.
    """
    try:
        import requests as req
        api_key  = os.getenv("GROQ_API_KEY")
        headers  = {"Authorization": f"Bearer {api_key}"}
        response = req.get(
            "https://api.groq.com/openai/v1/models",
            headers=headers,
            timeout=10
        )
        model_ids = [m["id"] for m in response.json()["data"]]

        for preferred in PREFERRED_MODELS:
            if preferred in model_ids:
                return preferred

        return model_ids[0] if model_ids else "llama3-8b-8192"

    except Exception:
        return "llama3-8b-8192"


def engineer_features(raw_input: dict) -> pd.DataFrame:
    """
    Takes raw customer input dictionary and applies
    the same feature engineering we did in Phase 2.

    Steps:
    1. Fix invalid category codes
    2. Create 6 derived features
    3. Return as a single-row DataFrame

    Args:
        raw_input: dict with 23 raw feature values

    Returns:
        DataFrame with 29 features (23 raw + 6 derived)
    """
    df = pd.DataFrame([raw_input])

    # Step 1: Fix invalid EDUCATION codes (0,5,6 → 4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # Step 2: Fix invalid MARRIAGE code (0 → 3)
    df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})

    # Step 3: Label encode categorical columns
    # SEX: 1→0, 2→1 | EDUCATION: 1→0,2→1,3→2,4→3 | MARRIAGE: 1→0,2→1,3→2
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        df[col] = le.fit_transform(df[col])

    # Step 4: Cap outliers at fixed 99th percentile thresholds
    # (same values used during Phase 2 training)
    bill_caps = {
        "BILL_AMT1": 426529, "BILL_AMT2": 420862,
        "BILL_AMT3": 409604, "BILL_AMT4": 368682,
        "BILL_AMT5": 348608, "BILL_AMT6": 332333
    }
    pay_caps  = {
        "PAY_AMT1": 168000, "PAY_AMT2": 168000,
        "PAY_AMT3": 120000, "PAY_AMT4": 110000,
        "PAY_AMT5": 85524,  "PAY_AMT6": 85000
    }
    for col, cap in {**bill_caps, **pay_caps}.items():
        df[col] = df[col].clip(upper=cap)

    # Step 5: Create 6 derived features (same as Phase 2)
    total_bill = df[[
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
    ]].sum(axis=1)

    total_pay  = df[[
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
        "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]].sum(axis=1)

    df["credit_utilization_rate"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)
    df["debt_to_income_ratio"]    = total_bill / (total_pay + 1)
    df["avg_payment_delay"]       = df[[
        "PAY_0", "PAY_2", "PAY_3",
        "PAY_4", "PAY_5", "PAY_6"
    ]].mean(axis=1)
    df["total_bill_amt"]          = total_bill
    df["total_pay_amt"]           = total_pay
    df["payment_to_bill_ratio"]   = total_pay / (total_bill + 1)

    return df


def scale_features(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Applies StandardScaler to the columns that were
    scaled during Phase 2 training.

    Args:
        df     : DataFrame with 29 features
        scaler : Fitted StandardScaler from Phase 2

    Returns:
        DataFrame with scaled numerical features
    """
    df_scaled = df.copy()
    cols_present = [c for c in SCALE_COLS if c in df_scaled.columns]
    df_scaled[cols_present] = scaler.transform(df_scaled[cols_present])
    return df_scaled