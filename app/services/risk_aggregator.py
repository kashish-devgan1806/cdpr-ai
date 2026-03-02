import pandas as pd
import numpy as np


def generate_developer_risk_report(model_data, df):

    model = model_data["model"]
    scaler = model_data["scaler"]

    risk_records = []

    for _, row in df.iterrows():

        is_weekend = 1 if row["day_of_week"] >= 5 else 0
        hour_deviation = abs(row["commit_hour"] - row["dev_mean_hour"])

        input_array = np.array([[
            row["commit_hour"],
            row["day_of_week"],
            is_weekend,
            hour_deviation,
            row["message_length"]
        ]])

        input_scaled = scaler.transform(input_array)

        probability = model.predict_proba(input_scaled)[0][1]

        risk_records.append({
            "developer": row["developer"],
            "risk_score": probability
        })

    risk_df = pd.DataFrame(risk_records)

    # Aggregate by developer
    summary = (
        risk_df
        .groupby("developer")["risk_score"]
        .mean()
        .reset_index()
    )

    # Professional risk classification
    def classify(score):
        if score >= 0.75:
            return "HIGH"
        elif score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    summary["risk_level"] = summary["risk_score"].apply(classify)

    return summary.sort_values("risk_score", ascending=False)