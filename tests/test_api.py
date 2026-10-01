import pytest
from fastapi.testclient import TestClient
from api.app import app, service

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200

def test_validation_error():
    payload = {
        "customer_id": "CUST_TEST",
        "initial_credit_score": 100,  # Fails constraint ge=300
        "account_age_days": 100,
        "sessions_last_90d": 5,
        "total_duration_last_90d": 50.0,
        "tx_count_last_90d": 2,
        "tx_volume_last_90d": 100.0,
        "avg_tx_size_last_90d": 50.0,
        "recency_days": 10
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422