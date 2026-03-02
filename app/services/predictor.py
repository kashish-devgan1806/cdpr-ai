import numpy as np
import time

def run_prediction(model_data, input_data):

    model = model_data["model"]
    scaler = model_data["scaler"]

    start = time.time()

    is_weekend = 1 if input_data.day_of_week >= 5 else 0
    hour_deviation = abs(
        input_data.commit_hour - input_data.developer_mean_hour
    )

    input_array = np.array([[
        input_data.commit_hour,
        input_data.day_of_week,
        is_weekend,
        hour_deviation,
        input_data.message_length
    ]])

    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    latency = (time.time() - start) * 1000

    return prediction, probability, latency