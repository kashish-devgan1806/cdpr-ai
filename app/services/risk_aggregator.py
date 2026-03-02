import pandas as pd
import numpy as np


def generate_developer_risk_report(model_data, df):

    model = model_data["model"]
    scaler = model_data["scaler"]

    # Compute engineered features
    df = df.copy()

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_deviation"] = abs(df["commit_hour"] - df["dev_mean_hour"])

    feature_matrix = df[[
        "commit_hour",
        "day_of_week",
        "is_weekend",
        "hour_deviation",
        "message_length"
    ]].values

    # Scale once (vectorized)
    scaled = scaler.transform(feature_matrix)

    # Predict once (vectorized)
    probabilities = model.predict_proba(scaled)[:, 1]

    df["risk_score"] = probabilities

    # Aggregate per developer
    summary = (
        df.groupby("developer")["risk_score"]
        .mean()
        .reset_index()
    )

    def classify(score):
        if score >= 0.75:
            return "HIGH"
        elif score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    summary["risk_level"] = summary["risk_score"].apply(classify)

    return summary.sort_values("risk_score", ascending=False)