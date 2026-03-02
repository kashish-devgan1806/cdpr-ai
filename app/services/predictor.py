import pandas as pd
import time


def run_prediction(model_data, input_data):

    model = model_data["model"]
    scaler = model_data["scaler"]
    features = model_data["features"]  # stored during training

    start = time.time()

    # Feature engineering
    is_weekend = 1 if input_data.day_of_week >= 5 else 0
    hour_deviation = abs(
        input_data.commit_hour - input_data.developer_mean_hour
    )

    # Create DataFrame with correct feature names
    input_df = pd.DataFrame([{
        "commit_hour": input_data.commit_hour,
        "day_of_week": input_data.day_of_week,
        "is_weekend": is_weekend,
        "hour_deviation": hour_deviation,
        "message_length": input_data.message_length
    }])

    # Enforce correct feature order
    input_df = input_df[features]

    # Scale using same feature names
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    latency = (time.time() - start) * 1000

    return prediction, probability, latency