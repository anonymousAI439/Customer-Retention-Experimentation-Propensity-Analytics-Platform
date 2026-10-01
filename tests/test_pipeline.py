import os
import pytest
from src.pipeline import PointInTimeFeatureExtractor

def test_pipeline_leakage():
    db_path = "data/retention_analytics.db"
    assert os.path.exists(db_path), "Database missing. Run driver first."
    extractor = PointInTimeFeatureExtractor(db_path)
    df = extractor.extract_dataset("2025-06-30", outcome_days=90)
    assert len(df) > 0
    assert df["recency_days"].min() >= 0, "Data leakage detected: negative recency days!"