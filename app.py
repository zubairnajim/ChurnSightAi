"""
Customer Churn Prediction API
FastAPI backend wrapping the Azure AutoML trained model (model.pkl).
Serves the prediction REST endpoint and the static frontend dashboard.
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import logging

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ChurnSight AI",
    description="Telecom customer churn prediction powered by Azure ML AutoML",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")

model = None


@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅  Model loaded successfully from: {MODEL_PATH}")
        logger.info(f"    Classes detected: {list(model.classes_)}")
    except FileNotFoundError:
        logger.error(f"❌  model.pkl not found at: {MODEL_PATH}")
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Make sure model.pkl is in the same directory as app.py.")
    except Exception as exc:
        logger.error(f"❌  Failed to load model: {exc}")
        raise


# ── Request / response schemas ─────────────────────────────────────────────────
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: bool
    Partner: bool
    Dependents: bool
    tenure: int
    PhoneService: bool
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: bool
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Azure App Service uses this to confirm the app is running."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/predict", tags=["Inference"])
async def predict(customer: CustomerData):
    """
    Run churn prediction on a single customer record.
    Returns:
      - churn (bool): Whether the model predicts churn
      - churn_probability (float): Probability of churn as a percentage (0–100)
      - retain_probability (float): Probability of retention as a percentage
      - risk_level (str): 'High' | 'Medium' | 'Low'
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please try again shortly.")

    try:
        # Build DataFrame that exactly matches the schema from scoring_file_v_2_0_0.py
        data = pd.DataFrame([{
            "gender":           customer.gender,
            "SeniorCitizen":    customer.SeniorCitizen,
            "Partner":          customer.Partner,
            "Dependents":       customer.Dependents,
            "tenure":           customer.tenure,
            "PhoneService":     customer.PhoneService,
            "MultipleLines":    customer.MultipleLines,
            "InternetService":  customer.InternetService,
            "OnlineSecurity":   customer.OnlineSecurity,
            "OnlineBackup":     customer.OnlineBackup,
            "DeviceProtection": customer.DeviceProtection,
            "TechSupport":      customer.TechSupport,
            "StreamingTV":      customer.StreamingTV,
            "StreamingMovies":  customer.StreamingMovies,
            "Contract":         customer.Contract,
            "PaperlessBilling": customer.PaperlessBilling,
            "PaymentMethod":    customer.PaymentMethod,
            "MonthlyCharges":   customer.MonthlyCharges,
            "TotalCharges":     customer.TotalCharges,
        }])

        # Run inference
        pred_result  = model.predict(data)
        prob_result  = model.predict_proba(data)

        if isinstance(pred_result, pd.DataFrame):
            prediction_raw = pred_result.iloc[0, 0]
        else:
            prediction_raw = pred_result[0]

        if isinstance(prob_result, pd.DataFrame):
            probabilities = prob_result.iloc[0].values
        else:
            probabilities = prob_result[0]

        # Find churn class index robustly.
        # AutoML may encode the target as: True/False (bool), 1/0 (int), or "Yes"/"No" (str)
        classes = list(model.classes_)
        churn_idx = 1  # safe default (second class is typically the positive class)
        for i, cls in enumerate(classes):
            if cls in (True, 1, "Yes", "yes", "1"):
                churn_idx = i
                break

        churn_prob     = float(probabilities[churn_idx])
        churn_prob_pct = round(churn_prob * 100, 1)
        retain_prob_pct = round((1.0 - churn_prob) * 100, 1)

        # Determine boolean churn result
        is_churn = prediction_raw in (True, 1, "Yes", "yes")

        # Risk level thresholds (tunable)
        if churn_prob >= 0.70:
            risk = "High"
        elif churn_prob >= 0.40:
            risk = "Medium"
        else:
            risk = "Low"

        logger.info(
            f"Prediction → {'CHURN' if is_churn else 'RETAIN'} | "
            f"Probability: {churn_prob_pct}% | Risk: {risk}"
        )

        return {
            "churn":              is_churn,
            "churn_probability":  churn_prob_pct,
            "retain_probability": retain_prob_pct,
            "risk_level":         risk,
        }

    except Exception as exc:
        logger.error(f"Prediction error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")


# ── Static files & root ────────────────────────────────────────────────────────
# IMPORTANT: mount /static AFTER defining all API routes so /predict isn't shadowed
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
