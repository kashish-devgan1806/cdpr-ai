from pydantic import BaseModel, Field

class CommitData(BaseModel):
    commit_hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    developer_mean_hour: float = Field(..., ge=0, le=23)
    developer_std_hour: float = Field(..., ge=0)
    message_length: int = Field(..., ge=0)

class PredictionResponse(BaseModel):
    risk_level: str
    risk_score: float
    severity: str
    model_version: str
    latency_ms: float