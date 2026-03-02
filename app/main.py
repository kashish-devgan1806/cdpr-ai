from fastapi import FastAPI
from app.schemas import CommitData, PredictionResponse
from app.model_loader import load_model
from app.services.predictor import run_prediction
from app.config import MODEL_VERSION

app = FastAPI(
    title="CDPR AI Service",
    version=MODEL_VERSION
)

model_data = load_model()

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": MODEL_VERSION
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CommitData):

    prediction, probability, latency = run_prediction(model_data, data)

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "model_version": MODEL_VERSION,
        "latency_ms": latency
    }