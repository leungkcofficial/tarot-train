"""
Simplified Pydantic schemas for prediction API (temporary for testing)
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
from app.core.validation import GenderEnum


class DemographicsRequest(BaseModel):
    """Patient demographics"""
    age: Optional[float] = None
    date_of_birth: Optional[str] = None
    gender: GenderEnum


class LabValueRequest(BaseModel):
    """Laboratory value with unit and date"""
    parameter: str
    value: float
    unit: str
    date: str


class ComorbidityRequest(BaseModel):
    """Medical comorbidity"""
    condition: str
    diagnosed: bool = False
    date: Optional[str] = None


class PredictionRequest(BaseModel):
    """Complete prediction request"""
    demographics: DemographicsRequest
    laboratory_values: List[LabValueRequest]
    medical_history: Optional[List[ComorbidityRequest]] = None
    session_id: Optional[str] = None


class ValidationWarning(BaseModel):
    """Validation warning"""
    field: str
    message: str
    severity: str = "warning"


class ProcessedLabValue(BaseModel):
    """Processed laboratory value after validation and conversion"""
    parameter: str
    original_value: float
    original_unit: str
    converted_value: float
    converted_unit: str
    date: str


class ShapValues(BaseModel):
    """SHAP values for feature importance"""
    dialysis: Dict[str, float]
    mortality: Dict[str, float]


class ConfidenceIntervals(BaseModel):
    """95% confidence intervals for predictions"""
    dialysis_lower: List[float]
    dialysis_upper: List[float] 
    mortality_lower: List[float]
    mortality_upper: List[float]


class ClinicalBenchmarks(BaseModel):
    """Clinical benchmark information"""
    nephrology_referral_threshold: float = 0.05
    multidisciplinary_care_threshold: float = 0.10
    krt_preparation_threshold: float = 0.40


class ModelInfo(BaseModel):
    """Model information and performance metrics"""
    ensemble_size: int
    model_types: Dict[str, int]
    inference_time_ms: float
    preprocessing_time_ms: float
    sequence_length: int


class PatientContext(BaseModel):
    """Patient clinical context derived from input"""
    age: float
    gender: str
    egfr: float
    egfr_stage: str
    cci_score: int
    observation_period: float


class PredictionResponse(BaseModel):
    """Successful prediction response"""
    success: bool = True
    predictions: Dict[str, List[float]]
    confidence_intervals: Optional[ConfidenceIntervals] = None
    shap_values: Optional[ShapValues] = None
    patient_context: PatientContext
    clinical_benchmarks: ClinicalBenchmarks
    processed_values: List[ProcessedLabValue]
    validation_warnings: List[ValidationWarning] = []
    model_info: ModelInfo
    session_id: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: float
    version: str
    models_loaded: int
    system_info: Optional[Dict[str, Any]] = None


class ValidationResponse(BaseModel):
    """Input validation response"""
    valid: bool
    warnings: List[ValidationWarning] = []
    egfr_info: Optional[Dict[str, Any]] = None
    processed_values: Optional[List[ProcessedLabValue]] = None
    errors: Optional[List[str]] = None