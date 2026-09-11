# AI Explanation Agent for the Credit Risk Scorer API.
# Uses SHAP to explain WHY the model predicted default/no default.
# Uses Groq LLM to convert SHAP values into plain English.

import shap
import numpy  as np
import pandas as pd

from src.utils  import (
    load_model,
    get_groq_client,
    get_best_groq_model
)

# Load model and setup SHAP explainer once at startup.
# TreeExplainer is the fastest SHAP method for Random Forest.
model         = load_model()
groq_client   = get_groq_client()
groq_model    = get_best_groq_model(groq_client)
shap_explainer= shap.TreeExplainer(model)

print(f"Agent initialized with model : {groq_model}")


def calculate_shap(scaled_df: pd.DataFrame) -> list:
    """
    Calculates SHAP values for one customer row.
    Returns top 8 features sorted by absolute impact.

    Args:
        scaled_df: Single-row scaled DataFrame from predict.py

    Returns:
        List of dicts with feature, shap_score, direction, impact
    """
    shap_values = shap_explainer.shap_values(scaled_df)

    # Get SHAP values for class 1 (DEFAULT)
    if isinstance(shap_values, list):
        shap_for_default = shap_values[1][0]
    else:
        shap_for_default = shap_values[0]

    # Pair feature names with SHAP values
    feature_names = scaled_df.columns.tolist()
    shap_pairs    = list(zip(feature_names, shap_for_default))

    # Sort by absolute value — most impactful first
    shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    # Format top 8 results
    top_factors = []
    for feature, value in shap_pairs[:8]:
        top_factors.append({
            "feature"   : feature,
            "shap_score": round(float(value), 4),
            "direction" : "toward_default"    if value > 0
                        else "away_from_default",
            "impact"    : "HIGH"   if abs(value) > 0.15
                        else "MEDIUM" if abs(value) > 0.05
                        else "LOW"
        })

    return top_factors


def build_explanation_prompt(
    prediction       : str,
    confidence       : float,
    top_factors      : list,
    customer_profile : dict
) -> str:
    """
    Builds a structured prompt for the Groq LLM.
    Formats SHAP values and customer data into
    a clear question the LLM can answer well.

    Args:
        prediction      : "DEFAULT" or "NO DEFAULT"
        confidence      : float (e.g. 78.5)
        top_factors     : list from calculate_shap()
        customer_profile: dict of key raw feature values

    Returns:
        Formatted prompt string
    """
    shap_lines = ""
    for f in top_factors:
        direction   = "increases" if f["direction"] == "toward_default" \
                    else "decreases"
        shap_lines += (
            f"\n  - {f['feature']}: "
            f"{direction} default risk "
            f"(Impact: {f['impact']}, "
            f"Score: {f['shap_score']:+.4f})"
        )

    profile_lines = ""
    key_fields    = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2",
        "BILL_AMT1", "PAY_AMT1", "EDUCATION", "MARRIAGE"
    ]
    for field in key_fields:
        if field in customer_profile:
            profile_lines += f"\n  - {field}: {customer_profile[field]}"

    return f"""
A machine learning model analyzed a bank customer
and made the following credit risk prediction:

PREDICTION : {prediction}
CONFIDENCE : {confidence}%

Top factors that influenced this decision (SHAP Analysis):
{shap_lines}

Customer Key Profile:
{profile_lines}

Instructions:
1. Explain WHY this prediction was made in exactly 3 bullet points.
Use simple language a non-technical bank manager can understand.
2. Suggest exactly 2 specific actions this customer can take
to improve their credit risk profile.

Use this exact format:

**Why this prediction was made:**
- [Explanation 1]
- [Explanation 2]
- [Explanation 3]

**Actions the customer can take:**
1. [Action 1]
2. [Action 2]
"""


def ask_groq(prompt: str) -> str:
    """
    Sends the explanation prompt to Groq LLM.
    Returns the plain text response.

    Args:
        prompt: Formatted prompt from build_explanation_prompt()

    Returns:
        LLM response as plain text string
    """
    response = groq_client.chat.completions.create(
        model    = groq_model,
        messages = [
            {
                "role"    : "system",
                "content" : (
                    "You are a senior credit risk analyst "
                    "at a bank with 15 years of experience. "
                    "You explain machine learning predictions "
                    "in clear, professional, non-technical language."
                )
            },
            {
                "role"    : "user",
                "content" : prompt
            }
        ],
        temperature = 0.1,
        max_tokens  = 1024
    )
    return response.choices[0].message.content


def explain_customer(
    prediction   : str,
    confidence   : float,
    scaled_df    : pd.DataFrame,
    raw_input    : dict
) -> dict:
    """
    Full AI explanation pipeline for one customer.

    Steps:
    1. Calculate SHAP values
    2. Build explanation prompt
    3. Send to Groq LLM
    4. Return structured result

    Args:
        prediction  : "DEFAULT" or "NO DEFAULT"
        confidence  : float percentage
        scaled_df   : scaled features for SHAP
        raw_input   : original raw input dict

    Returns:
        dict with top_shap_factors and ai_explanation
    """

    # Step 1: Calculate SHAP values
    top_factors = calculate_shap(scaled_df)

    # Step 2: Build prompt
    prompt      = build_explanation_prompt(
        prediction       = prediction,
        confidence       = confidence,
        top_factors      = top_factors,
        customer_profile = raw_input
    )

    # Step 3: Get AI explanation
    ai_explanation = ask_groq(prompt)

    return {
        "top_shap_factors" : top_factors,
        "ai_explanation"   : ai_explanation,
        "model_used"       : groq_model
    }