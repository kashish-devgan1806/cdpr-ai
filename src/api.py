import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

#load model
model_data = joblib.load("model/risk_model.pkl")
model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]

#init app
app = FastAPI(
    title= "CDPR AI Risk Prediction API",
    description= "Predict developers productivity risk",
    version= "1.0"
)

#input schema

class CommitData(BaseModel):
    commit_hour: int
    day_of_week: int
    developer_mean_hour: float
    developer_std_hour: float
    message_length: int
    
#root endpoint

@app.get("/")
def root():
    return {"message": "CDPR AI API is running."}


#prediction endpoint
@app.post("/predict")
def predict(data: CommitData):

    is_weekend = 1 if data.day_of_week >= 5 else 0

    hour_deviation = abs(data.commit_hour - data.developer_mean_hour)

    input_array = np.array([[
        data.commit_hour,
        data.day_of_week,
        is_weekend,
        hour_deviation,
        data.message_length
    ]])

    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }