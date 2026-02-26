import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

#config block

DATA_PATH= "data/processed_commits.csv"
MODEL_PATH= "model/risk_model.pkl"

#load data block

print("Loading processed dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset size: {len(df)}")

#features block

features = [
    "commit_hour",
    "day_of_week",
    "is_weekend",
    "hour_deviation",
    "message_length"
]

X= df[features]
y= df["risk_label"]

#scaling block

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train-test split block

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y, 
    test_size=0.2, 
    random_state=42
    )

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

#model training block

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

print("Training model...")

model.fit(X_train, y_train)


#cross validation block

cv_scores = cross_val_score(model, X_scaled, y, cv=5, n_jobs=-1)

print("\nCross-validation scores:", cv_scores)
print("Average CV score:", np.mean(cv_scores))

#evaluation block

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#save model block

os.makedirs("model", exist_ok=True)

joblib.dump({
    "model": model,
    "scaler": scaler,
    "features": features
}, MODEL_PATH)

print(f"\nModel saved to {MODEL_PATH}")

