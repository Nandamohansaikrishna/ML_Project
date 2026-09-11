# FastAPI Backend for Credit Risk Scorer 
# Endpoints:
#   GET  /             → Health check
#   GET  /model-info   → Model metadata
#   POST /predict      → Prediction only (fast)
#   POST /explain      → Prediction + SHAP + AI explanation
# Run with:
#   uvicorn app:app --reload --port 8000
# Docs available at:
#   http://localhost:8000/docs

from fastapi             import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic            import BaseModel, Field
from typing              import Optional
import traceback

from src.predict import predict_customer, get_model_info
from src.agent   import explain_customer

# Initializing FastAPI application

app = FastAPI(
    title          = "Credit Risk Scorer API",
    description    = (
        "Production API for predicting credit default risk. "
        "Uses Random Forest model trained on UCI Credit Card Dataset. "
        "Provides SHAP-based explanations via Groq AI."
    ),
    version        = "1.0.0",
    docs_url       = "/docs",
    redoc_url      = "/redoc"
)

# CORS Middleware
# Allows Streamlit frontend (localhost:8501) to call this API.
# In production update origins to your actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)

# Request Schema — what the user must send
# All 23 original features with validation rules.
# Field() adds documentation visible in /docs

class CustomerInput(BaseModel):
    LIMIT_BAL  : float = Field(..., ge=10000,  le=1000000, description="Credit limit in NT dollars")
    SEX        : int   = Field(..., ge=1,      le=2,       description="1=Male, 2=Female")
    EDUCATION  : int   = Field(..., ge=0,      le=6,       description="1=Grad, 2=Uni, 3=HS, 4=Other")
    MARRIAGE   : int   = Field(..., ge=0,      le=3,       description="1=Married, 2=Single, 3=Other")
    AGE        : int   = Field(..., ge=18,     le=100,     description="Age in years")
    PAY_0      : int   = Field(..., ge=-2,     le=8,       description="Repayment status Sep (-1=on time)")
    PAY_2      : int   = Field(..., ge=-2,     le=8,       description="Repayment status Aug")
    PAY_3      : int   = Field(..., ge=-2,     le=8,       description="Repayment status Jul")
    PAY_4      : int   = Field(..., ge=-2,     le=8,       description="Repayment status Jun")
    PAY_5      : int   = Field(..., ge=-2,     le=8,       description="Repayment status May")
    PAY_6      : int   = Field(..., ge=-2,     le=8,       description="Repayment status Apr")
    BILL_AMT1  : float = Field(..., description="Bill amount Sep (NT dollars)")
    BILL_AMT2  : float = Field(..., description="Bill amount Aug")
    BILL_AMT3  : float = Field(..., description="Bill amount Jul")
    BILL_AMT4  : float = Field(..., description="Bill amount Jun")
    BILL_AMT5  : float = Field(..., description="Bill amount May")
    BILL_AMT6  : float = Field(..., description="Bill amount Apr")
    PAY_AMT1   : float = Field(..., ge=0,     description="Payment amount Sep")
    PAY_AMT2   : float = Field(..., ge=0,     description="Payment amount Aug")
    PAY_AMT3   : float = Field(..., ge=0,     description="Payment amount Jul")
    PAY_AMT4   : float = Field(..., ge=0,     description="Payment amount Jun")
    PAY_AMT5   : float = Field(..., ge=0,     description="Payment amount May")
    PAY_AMT6   : float = Field(..., ge=0,     description="Payment amount Apr")

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL" : 50000,
                "SEX"       : 2,
                "EDUCATION" : 2,
                "MARRIAGE"  : 1,
                "AGE"       : 30,
                "PAY_0"     : 0,
                "PAY_2"     : 0,
                "PAY_3"     : 0,
                "PAY_4"     : 0,
                "PAY_5"     : 0,
                "PAY_6"     : 0,
                "BILL_AMT1" : 15000,
                "BILL_AMT2" : 14000,
                "BILL_AMT3" : 13000,
                "BILL_AMT4" : 12000,
                "BILL_AMT5" : 11000,
                "BILL_AMT6" : 10000,
                "PAY_AMT1"  : 2000,
                "PAY_AMT2"  : 2000,
                "PAY_AMT3"  : 2000,
                "PAY_AMT4"  : 2000,
                "PAY_AMT5"  : 2000,
                "PAY_AMT6"  : 2000
            }
        }

# Response Schemas — what the API sends back
class PredictionResponse(BaseModel):
    prediction      : str
    prediction_code : int
    confidence      : float
    probability     : dict

class SHAPFactor(BaseModel):
    feature    : str
    shap_score : float
    direction  : str
    impact     : str

class ExplainResponse(BaseModel):
    prediction       : str
    prediction_code  : int
    confidence       : float
    probability      : dict
    top_shap_factors : list
    ai_explanation   : str
    model_used       : str

# ENDPOINT 1 — Health Check
# GET /
@app.get(
    "/",
    summary     = "Health Check",
    description = "Returns API status. Use this to verify the server is running."
)
def health_check():
    return {
        "status"  : " Credit Risk Scorer API is running",
        "version" : "1.0.0",
        "docs"    : "/docs",
        "endpoints": {
            "health"      : "GET  /",
            "model_info"  : "GET  /model-info",
            "predict"     : "POST /predict",
            "explain"     : "POST /explain"
        }
    }

# Model Info
# GET /model-info
@app.get(
    "/model-info",
    summary     = "Model Metadata",
    description = "Returns information about the loaded ML model."
)
def model_info():
    try:
        info = get_model_info()
        return {
            "status" : "success",
            "model"  : info
        }
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Failed to load model info: {str(e)}"
        )

# Predict Only
# POST /predict
# Fast endpoint — no SHAP, no AI explanation
@app.post(
    "/predict",
    response_model = PredictionResponse,
    summary        = "Predict Credit Default Risk",
    description    = (
        "Takes customer financial data and returns "
        "a default risk prediction with probability scores. "
        "Fast endpoint — no AI explanation included."
    )
)
def predict(customer: CustomerInput):
    try:
        raw_input = customer.model_dump()
        result    = predict_customer(raw_input)

        return PredictionResponse(
            prediction      = result["prediction"],
            prediction_code = result["prediction_code"],
            confidence      = result["confidence"],
            probability     = result["probability"]
        )

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Prediction failed: {str(e)}"
        )

# Predict + Explain
# POST /explain
# Full pipeline — prediction + SHAP + AI explanation
# Slower (2-5 seconds) due to LLM call

class ExplainResponse(BaseModel):

    model_config = {"protected_namespaces": ()}

    prediction       : str
    prediction_code  : int
    confidence       : float
    probability      : dict
    top_shap_factors : list
    ai_explanation   : str
    model_used       : str
@app.post(
    "/explain",
    response_model = ExplainResponse,
    summary        = "Predict + AI Explanation",
    description    = (
        "Takes customer financial data and returns "
        "a prediction WITH SHAP-based feature importance "
        "AND an AI-generated plain English explanation. "
        "Takes 2-5 seconds due to LLM processing."
    )
)
def explain(customer: CustomerInput):
    try:
        raw_input = customer.model_dump()

        # Step 1: Get prediction
        pred_result = predict_customer(raw_input)

        # Step 2: Get AI explanation with SHAP
        expl_result = explain_customer(
            prediction  = pred_result["prediction"],
            confidence  = pred_result["confidence"],
            scaled_df   = pred_result["processed_df"],
            raw_input   = raw_input
        )

        return ExplainResponse(
            prediction       = pred_result["prediction"],
            prediction_code  = pred_result["prediction_code"],
            confidence       = pred_result["confidence"],
            probability      = pred_result["probability"],
            top_shap_factors = expl_result["top_shap_factors"],
            ai_explanation   = expl_result["ai_explanation"],
            model_used       = expl_result["model_used"]
        )

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Explanation failed: {str(e)}\n{traceback.format_exc()}"
        )