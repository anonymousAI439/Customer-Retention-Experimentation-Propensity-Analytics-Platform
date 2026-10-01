import os
import joblib
import numpy as np
import pandas as pd
from api.schemas import ScoringRequest, ScoringResponse

class ModelInferenceService:
    def __init__(self, model_path="model.joblib", scaler_path="scaler.joblib"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        
        self.num_cols = [
            'initial_credit_score', 'account_age_days', 'sessions_last_90d',
            'total_duration_last_90d', 'tx_count_last_90d', 'tx_volume_last_90d',
            'avg_tx_size_last_90d', 'recency_days'
        ]
        self.cat_cols = ['segment', 'acquisition_channel']
        
        self.load_artifacts()

    def load_artifacts(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    @property
    def expected_features(self) -> list:
        """Retrieves the exact feature names and order expected by the trained scaler/model."""
        if hasattr(self.scaler, "feature_names_in_"):
            return list(self.scaler.feature_names_in_)
        elif hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)
        else:
            # Fallback if feature_names_in_ is not attached to the loaded artifacts
            return self.num_cols + [
                'segment_SMB', 'segment_Wealth',
                'acquisition_channel_Organic', 'acquisition_channel_Paid Search', 'acquisition_channel_Referral'
            ]

    def _prepare_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """One-hot encodes categorical fields and aligns columns to match training fit."""
        # 1. One-hot encode categoricals
        df_encoded = pd.get_dummies(df_raw[self.cat_cols], drop_first=True)
        
        # 2. Combine numerical and dummy columns
        df_full = pd.concat([df_raw[self.num_cols], df_encoded], axis=1)
        
        # 3. Align columns to match exact fit order, filling absent dummy columns with 0
        df_aligned = df_full.reindex(columns=self.expected_features, fill_value=0)
        return df_aligned

    def predict_single(self, request: ScoringRequest) -> ScoringResponse:
        if not self.is_ready:
            raise RuntimeError("Model artifacts not loaded.")

        # Convert Pydantic request object to DataFrame
        request_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        df_raw = pd.DataFrame([request_dict])

        # Preprocess features & align schema
        X_aligned = self._prepare_features(df_raw)

        # Scale features and generate inference
        scaled = self.scaler.transform(X_aligned)
        propensity = float(self.model.predict_proba(scaled)[0, 1])
        risk_decile = int(np.clip(11 - np.ceil(propensity * 10), 1, 10))

        if propensity >= 0.70:
            category, action = "High Risk", "Dispatch high-value retention offer ($50 statement credit)"
        elif propensity >= 0.35:
            category, action = "Medium Risk", "Enroll in automated re-engagement campaign"
        else:
            category, action = "Low Risk", "Standard engagement (No intervention)"

        return ScoringResponse(
            customer_id=request.customer_id,
            propensity_score=round(propensity, 4),
            risk_decile=risk_decile,
            risk_category=category,
            recommended_action=action
        )