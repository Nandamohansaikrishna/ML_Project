# Prediction logic for the Credit Risk Scorer API.
# Takes raw customer input, applies feature engineering,
# scaling, and returns model prediction with probabilities.

import pandas as pd
import numpy  as np

from src.utils import (
    load_model,
    load_scaler,
    engineer_features,
    scale_features
)

# Load model and scaler once at module import.
# They stay in memory for the entire server lifetime.
# This is much faster than loading on every request.
model  = load_model()
scaler = load_scaler()


def predict_customer(raw_input: dict) -> dict:
    """
    Full prediction pipeline for one customer.

    Steps:
    1. Engineer features from raw input
    2. Scale numerical features
    3. Run Random Forest prediction
    4. Return prediction + probabilities

    Args:
        raw_input: dict with 23 raw feature values

    Returns:
        dict containing:
            prediction       : "DEFAULT" or "NO DEFAULT"
            prediction_code  : 1 or 0
            confidence       : float (highest probability)
            probability      : dict with no_default and default %
            processed_df     : scaled DataFrame for SHAP
            raw_df           : unscaled DataFrame for display
    """

    # Step 1: Feature engineering
    raw_df     = engineer_features(raw_input)

    # Step 2: Feature scaling
    scaled_df  = scale_features(raw_df, scaler)

    # Step 3: Prediction
    prediction      = model.predict(scaled_df)[0]
    probability     = model.predict_proba(scaled_df)[0]

    # Step 4: Format results
    prediction_label = "DEFAULT" if prediction == 1 else "NO DEFAULT"
    confidence       = round(float(max(probability)) * 100, 2)

    return {
        "prediction"      : prediction_label,
        "prediction_code" : int(prediction),
        "confidence"      : confidence,
        "probability"     : {
            "no_default"  : round(float(probability[0]) * 100, 2),
            "default"     : round(float(probability[1]) * 100, 2)
        },
        "processed_df"    : scaled_df,
        "raw_df"          : raw_df
    }


def get_model_info() -> dict:
    """
    Returns metadata about the loaded model.
    Used by the GET /model-info endpoint.
    """
    return {
        "model_type"       : type(model).__name__,
        "n_estimators"     : model.n_estimators,
        "n_features"       : model.n_features_in_,
        "feature_names"    : list(model.feature_names_in_)
                            if hasattr(model, "feature_names_in_")
                            else [],
        "classes"          : ["NO DEFAULT", "DEFAULT"],
        "model_file"       : "models/best_model.pkl"
    }