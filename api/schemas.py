from pydantic import BaseModel, Field
from typing import List

class ScoringRequest(BaseModel):
    customer_id: str = Field(..., json_schema_extra={"example": "CUST_00123"})
    initial_credit_score: int = Field(..., ge=300, le=850, json_schema_extra={"example": 720})
    account_age_days: int = Field(..., ge=0, json_schema_extra={"example": 365})
    sessions_last_90d: int = Field(..., ge=0, json_schema_extra={"example": 15})
    total_duration_last_90d: float = Field(..., ge=0.0, json_schema_extra={"example": 120.5})
    tx_count_last_90d: int = Field(..., ge=0, json_schema_extra={"example": 8})
    tx_volume_last_90d: float = Field(..., ge=0.0, json_schema_extra={"example": 450.0})
    avg_tx_size_last_90d: float = Field(..., ge=0.0, json_schema_extra={"example": 56.25})
    recency_days: int = Field(..., ge=0, json_schema_extra={"example": 12})
    
    # Categorical fields required for one-hot encoding in ModelInferenceService
    segment: str = Field(default="Standard", json_schema_extra={"example": "Standard"})
    acquisition_channel: str = Field(default="Organic", json_schema_extra={"example": "Organic"})

class ScoringResponse(BaseModel):
    customer_id: str
    propensity_score: float = Field(..., ge=0.0, le=1.0)
    risk_decile: int = Field(..., ge=1, le=10)
    risk_category: str
    recommended_action: str

class BatchScoringRequest(BaseModel):
    customers: List[ScoringRequest]

class BatchScoringResponse(BaseModel):
    total_processed: int
    predictions: List[ScoringResponse]

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool