from fastapi import FastAPI, HTTPException, status
from api.schemas import (
    ScoringRequest, 
    ScoringResponse, 
    BatchScoringRequest, 
    BatchScoringResponse, 
    HealthCheckResponse
)
from api.service import ModelInferenceService

app = FastAPI(
    title="JPMorgan Churn Propensity Scoring API", 
    version="1.0.0",
    description="Microservice providing real-time and batch customer churn prediction."
)

service = ModelInferenceService()

@app.get("/health", response_model=HealthCheckResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint to monitor service readiness and artifact loading."""
    return HealthCheckResponse(
        status="healthy" if service.is_ready else "degraded",
        model_loaded=service.model is not None,
        scaler_loaded=service.scaler is not None
    )

@app.post("/predict", response_model=ScoringResponse, status_code=status.HTTP_200_OK)
async def predict_churn(request: ScoringRequest):
    """Real-time churn propensity scoring for a single customer."""
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model artifacts uninitialized or service degraded."
        )
    try:
        return service.predict_single(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchScoringResponse, status_code=status.HTTP_200_OK)
async def predict_churn_batch(request: BatchScoringRequest):
    """Batch churn propensity scoring for multiple customers."""
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model artifacts uninitialized or service degraded."
        )
    try:
        preds = [service.predict_single(c) for c in request.customers]
        return BatchScoringResponse(total_processed=len(preds), predictions=preds)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )