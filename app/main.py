from fastapi import FastAPI
from app.schemas import CommitData, PredictionResponse
from app.model_loader import load_model
from app.services.predictor import run_prediction
from app.config import MODEL_VERSION

import pandas as pd
from app.services.risk_aggregator import generate_developer_risk_report

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

    # Convert Probability → Professional Risk Levels
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


# ---------------------------------------------------
# Manager Alerts Endpoint
# ---------------------------------------------------
@app.get("/manager/alerts")
def manager_alerts():

    logger.info("Manager alert report requested")

    try:
        df = pd.read_csv("data/processed_commits.csv")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {"error": "Unable to load processed dataset"}

    report = generate_developer_risk_report(model_data, df)

    high_risk = report[report["risk_level"] == "HIGH"]

    alert_message = None

    if len(high_risk) > 0:
        alert_message = (
            f"⚠ {len(high_risk)} developer(s) show sustained high behavioral deviation patterns."
        )

    logger.info(
        f"Manager report generated | "
        f"Total Developers: {len(report)} | "
        f"High Risk Count: {len(high_risk)}"
    )

    return {
        "total_developers": int(len(report)),
        "high_risk_count": int(len(high_risk)),
        "alerts": alert_message,
        "high_risk_developers": high_risk.to_dict(orient="records")
    }