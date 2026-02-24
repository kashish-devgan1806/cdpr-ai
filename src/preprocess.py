import pandas as pd
import os

# CONFIG
INPUT_PATH = "data/github_commits_large.csv"
OUTPUT_PATH = "data/processed_commits.csv"


# LOAD DATA

def load_data():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Original dataset size: {len(df)}")

    return df


# CLEAN DATA

def clean_data(df):

    print("Cleaning data...")

    # Remove duplicates
    df = df.drop_duplicates(subset=["commit_id"])

    # Remove missing values
    df = df.dropna()

    print(f"After cleaning: {len(df)}")

    return df


# FEATURE VALIDATION

def validate_features(df):

    print("Validating features...")

    required_columns = [
        "message_length",
        "commit_hour",
        "day_of_week",
        "is_weekend",
        "risk_label"
    ]

    for col in required_columns:

        if col not in df.columns:
            raise Exception(f"Missing required column: {col}")

    print("All required features present.")

    return df


# SAVE PROCESSED DATA

def save_data(df):

    os.makedirs("data", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Processed dataset saved to {OUTPUT_PATH}")


# =========================
# MAIN PIPELINE
# =========================

def main():

    df = load_data()

    df = clean_data(df)

    df = validate_features(df)

    save_data(df)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()