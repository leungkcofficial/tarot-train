"""
Temporal prediction schemas for TAROT CKD Risk Prediction API
Handles 11 features x 10 timepoints format
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator


class PatientInfo(BaseModel):
    """Patient information"""
    age_at_obs: int = Field(..., ge=18, le=120, description="Age at observation time")
    gender: str = Field(..., pattern=r"^(male|female)$", description="Patient gender")
    observation_period: int = Field(..., ge=0, description="Days since CKD diagnosis")


class TemporalPredictionRequest(BaseModel):
    """
    Temporal prediction request with feature matrix format
    
    Feature matrix: 11 features x 10 timepoints
    Features: age_at_obs, albumin, uacr, bicarbonate, cci_score_total, 
             creatinine, gender, hemoglobin, ht, observation_period, phosphate
    """
    feature_matrix: Dict[str, List[Union[float, None]]] = Field(
        ..., 
        description="Feature matrix: 11 features x 10 timepoints"
    )
    timepoint_dates: List[str] = Field(
        ..., 
        max_items=10, 
        description="Dates for each timepoint (YYYY-MM-DD format)"
    )
    patient_info: PatientInfo = Field(..., description="Patient demographic information")
    session_id: Optional[str] = Field(None, description="Session identifier")

    @validator('feature_matrix')
    def validate_feature_matrix(cls, v):
        """Validate feature matrix structure"""
        required_features = {
            'age_at_obs', 'albumin', 'uacr', 'bicarbonate', 'cci_score_total',
            'creatinine', 'gender', 'hemoglobin', 'ht', 'observation_period', 'phosphate'
        }
        
        # Check all required features are present
        missing_features = required_features - set(v.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Check each feature has exactly 10 timepoints
        for feature, values in v.items():
            if len(values) != 10:
                raise ValueError(f"Feature {feature} must have exactly 10 timepoints, got {len(values)}")
        
        return v
    
    @validator('timepoint_dates')
    def validate_timepoint_dates(cls, v):
        """Validate timepoint dates"""
        if len(v) != 10:
            raise ValueError(f"Must provide exactly 10 timepoint dates, got {len(v)}")
        
        # Validate date format for non-empty dates
        for date_str in v:
            if date_str:  # Allow empty strings for missing dates
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")
        
        return v


class TemporalPredictionResponse(BaseModel):
    """Response for temporal prediction"""
    success: bool = Field(True, description="Request success status")
    predictions: Dict[str, List[float]] = Field(..., description="Risk predictions over time")
    confidence_intervals: Optional[Dict[str, List[float]]] = Field(None, description="95% confidence intervals")
    shap_values: Optional[Dict[str, Dict[str, float]]] = Field(None, description="SHAP feature importance")
    patient_context: Dict[str, Any] = Field(..., description="Patient context information")
    clinical_benchmarks: Dict[str, float] = Field(..., description="Clinical decision thresholds")
    model_info: Dict[str, Any] = Field(..., description="Model ensemble information")
    session_id: str = Field(..., description="Session identifier")
    timestamp: str = Field(..., description="Response timestamp")
    validation_warnings: Optional[List[str]] = Field(default_factory=list, description="Input validation warnings")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "predictions": {
                    "dialysis_risk": [0.05, 0.12, 0.18, 0.25, 0.32],
                    "mortality_risk": [0.03, 0.08, 0.14, 0.21, 0.28]
                },
                "confidence_intervals": {
                    "dialysis_lower": [0.03, 0.08, 0.12, 0.17, 0.23],
                    "dialysis_upper": [0.07, 0.16, 0.24, 0.33, 0.41],
                    "mortality_lower": [0.02, 0.05, 0.09, 0.15, 0.20],
                    "mortality_upper": [0.04, 0.11, 0.19, 0.27, 0.36]
                },
                "shap_values": {
                    "dialysis": {
                        "creatinine": 0.15,
                        "age_at_obs": 0.12,
                        "uacr": 0.08
                    },
                    "mortality": {
                        "age_at_obs": 0.18,
                        "cci_score_total": 0.14,
                        "creatinine": 0.10
                    }
                },
                "patient_context": {
                    "age": 68,
                    "gender": "female",
                    "egfr": 25.4,
                    "egfr_stage": "CKD Stage 4"
                },
                "clinical_benchmarks": {
                    "nephrology_referral_threshold": 0.05,
                    "multidisciplinary_care_threshold": 0.10,
                    "krt_preparation_threshold": 0.40
                },
                "model_info": {
                    "ensemble_size": 36,
                    "model_types": {
                        "deepsurv": 24,
                        "deephit": 12
                    },
                    "inference_time_ms": 45.2
                },
                "session_id": "abc123",
                "timestamp": "2025-01-15T12:00:00Z"
            }
        }