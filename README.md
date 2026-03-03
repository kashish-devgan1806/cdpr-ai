# CDPR AI

## Engineering Risk Intelligence Platform

CDPR AI is a behavioral risk intelligence system designed to proactively
identify developer productivity deviation patterns using commit behavior
analytics.

This platform combines machine learning, statistical deviation modeling,
and a production-ready microservice architecture to provide
leadership-level visibility into engineering risk exposure.

----------------------------------------------------------------------

## 🚀 Overview

Modern engineering teams lack proactive intelligence systems to detect
sustained behavioral deviations that may indicate delivery instability,
burnout signals, or workflow anomalies.

CDPR AI addresses this gap by:

-   Ingesting real-world GitHub commit data
-   Engineering behavioral deviation features
-   Classifying risk severity (LOW / MEDIUM / HIGH)
-   Aggregating developer-level risk metrics
-   Exposing predictions through a FastAPI microservice
-   Providing an executive dashboard for monitoring

----------------------------------------------------------------------

## 🏗 System Architecture

GitHub API\
→ Data Collection Pipeline\
→ Feature Engineering\
→ Risk Classification Model (RandomForest)\
→ FastAPI Inference Service\
→ Developer Aggregation Layer\
→ Executive Dashboard (Streamlit)

------------------------------------------------------------------------

## 📊 Core Features

### 1. Behavioral Risk Modeling

-   Statistical deviation-based labeling
-   Risk probability scoring
-   Professional severity classification
-   Schema-consistent inference pipeline

### 2. Manager Intelligence Layer

-   Developer-level risk aggregation
-   High-risk alert detection
-   Risk tier segmentation
-   Severity distribution reporting

### 3. Executive Dashboard

-   High-risk employee pie chart
-   Multi-tab severity breakdown (HIGH / MEDIUM / LOW)
-   Per-severity analytics visualization
-   CSV export functionality
-   Clean, light SaaS-style UI

----------------------------------------------------------------------

## 🔌 API Endpoints

### Health Check

GET /health

Returns system health and model status.

### Risk Prediction

POST /predict

Returns: - risk_level - risk_score - severity - model_version -
latency_ms

### Manager Alerts

GET /manager/alerts

Returns: - Total developers - High-risk count - Alert message - Full
developer severity breakdown

----------------------------------------------------------------------

## 📈 Model Performance

-   Cross-validation score: \~0.86
-   Test accuracy: \~0.91
-   Feature schema consistency enforced
-   Vectorized batch inference
-   Production-safe scaling pipeline

----------------------------------------------------------------------

## 🖥 Running the Application

### Install Dependencies

pip install -r requirements.txt

### Start Backend

uvicorn app.main:app --reload

### Launch Dashboard

streamlit run dashboard.py

----------------------------------------------------------------------

## 🐳 Docker Deployment (Optional)

docker build -t cdpr-ai .\
docker run -p 8000:8000 cdpr-ai

----------------------------------------------------------------------
## 📦 Project Structure

    CDPR-AI/
    ├── app/
    │   ├── main.py
    │   ├── schemas.py
    │   ├── config.py
    │   ├── model_loader.py
    │   └── services/
    ├── data/
    ├── model/
    ├── src/
    ├── dashboard.py
    ├── requirements.txt
    └── Dockerfile

----------------------------------------------------------------------

## 🔐 Disclaimer

This system is a behavioral analytics research platform and should not
be used as a direct performance evaluation mechanism. It is intended for
early signal monitoring and team-level operational insights.

----------------------------------------------------------------------

## 🧠 Author

Developed as an AI Engineering R&D initiative.

Last Updated: 03 March 2026
