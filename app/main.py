from fastapi import FastAPI
from app.schemas import CommitData, PredictionResponse
from app.model_loader import load_model
from app.services.predictor import run_prediction
from app.config import MODEL_VERSION

import logging

# ---------------------------------------------------
# Logging Configuration
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------
app = FastAPI(
    title="CDPR AI - Engineering Risk Intelligence Service",
    version=MODEL_VERSION
)

# Load model once at startup
model_data = load_model()


# ---------------------------------------------------
# Health Endpoint
# ---------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_data is not None,
        "version": MODEL_VERSION
    }


# ---------------------------------------------------
# Risk Prediction Endpoint
# ---------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: CommitData):

    logger.info(f"Prediction requested | Input: {data.dict()}")

    prediction, probability, latency = run_prediction(model_data, data)

    # ---------------------------------------------------
    # Convert Probability → Professional Risk Levels
    # ---------------------------------------------------
    if probability >= 0.75:
        risk_level = "HIGH"
        severity = "Critical"
    elif probability >= 0.50:
        risk_level = "MEDIUM"
        severity = "Elevated"
    else:
        risk_level = "LOW"
        severity = "Stable"

    logger.info(
        f"Prediction completed | "
        f"Risk Level: {risk_level} | "
        f"Score: {probability:.4f} | "
        f"Latency: {latency:.2f} ms"
    )

    return {
        "risk_level": risk_level,
        "risk_score": float(probability),
        "severity": severity,
        "model_version": MODEL_VERSION,
        "latency_ms": latency
    }