import pandas as pd


def generate_developer_risk_report(model_data, df):

    model = model_data["model"]
    scaler = model_data["scaler"]
    features = model_data["features"]

    df = df.copy()

    # Feature engineering
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_deviation"] = abs(df["commit_hour"] - df["dev_mean_hour"])

    # Keep DataFrame (DO NOT convert to .values)
    feature_df = df[[
        "commit_hour",
        "day_of_week",
        "is_weekend",
        "hour_deviation",
        "message_length"
    ]]

    # Enforce correct training feature order
    feature_df = feature_df[features]

    # Scale with column names preserved
    scaled = scaler.transform(feature_df)

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