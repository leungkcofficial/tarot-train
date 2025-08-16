"""
Pydantic schemas for prediction API
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel

from app.core.validation import GenderEnum


class DemographicsRequest(BaseModel):
    """Patient demographics"""
    age: Optional[float] = None
    date_of_birth: Optional[str] = None
    gender: GenderEnum
    
    # @validator('age', always=True)
    # def validate_age_or_dob(cls, v, values):
    #     """Ensure either age or date_of_birth is provided"""
    #     if v is None and values.get('date_of_birth') is None:
    #         raise ValueError("Either age or date_of_birth must be provided")
    #     return v


class LabValueRequest(BaseModel):
    """Laboratory value with unit and date"""
    parameter: str
    value: float
    unit: str
    date: str
    
    # @validator('parameter')
    # def validate_parameter(cls, v):
    #     """Validate parameter name"""
    #     allowed_parameters = {
    #         'creatinine', 'hemoglobin', 'phosphate', 'bicarbonate', 
    #         'uacr', 'upcr'
    #     }
    #     if v.lower() not in allowed_parameters:
    #         raise ValueError(f"Parameter must be one of: {allowed_parameters}")
    #     return v.lower()


class ComorbidityRequest(BaseModel):
    """Medical comorbidity"""
    condition: str
    diagnosed: bool = False
    date: Optional[str] = None
    
    # @validator('condition')
    # def validate_condition(cls, v):
    #     """Validate condition name"""
    #     allowed_conditions = {
    #         'hypertension', 'diabetes', 'myocardial_infarction',
    #         'congestive_heart_failure', 'peripheral_vascular_disease',
    #         'cerebrovascular_disease', 'dementia', 'chronic_pulmonary_disease',
    #         'rheumatic_disease', 'peptic_ulcer_disease', 'mild_liver_disease',
    #         'renal_mild_moderate', 'diabetes_w_complication',
    #         'hemiplegia_paraplegia', 'any_malignancy', 'liver_severe',
    #         'renal_severe', 'metastatic_cancer', 'hiv', 'aids'
    #     }
    #     if v.lower() not in allowed_conditions:
    #         raise ValueError(f"Condition must be one of: {allowed_conditions}")
    #     return v.lower()


class PredictionRequest(BaseModel):
    """Complete prediction request"""
    demographics: DemographicsRequest
    laboratory_values: List[LabValueRequest]
    medical_history: Optional[List[ComorbidityRequest]] = None
    session_id: Optional[str] = None
    
    # @validator('laboratory_values')
    # def validate_required_labs(cls, v):
    #     """Ensure required lab values are present"""
    #     parameters = {lab.parameter for lab in v}
    #     required = {'creatinine', 'hemoglobin', 'phosphate', 'bicarbonate'}
    #     
    #     # Must have either uacr or upcr
    #     urine_params = {'uacr', 'upcr'} & parameters
    #     
    #     missing_required = required - parameters
    #     if missing_required:
    #         raise ValueError(f"Missing required lab values: {missing_required}")
    #     
    #     if not urine_params:
    #         raise ValueError("Must provide either UACR or UPCR")
    #     
    #     return v


class ValidationWarning(BaseModel):
    """Validation warning"""
    field: str = Field(..., description="Field with warning")
    message: str = Field(..., description="Warning message")
    severity: str = Field("warning", description="Severity level")


class ProcessedLabValue(BaseModel):
    """Processed laboratory value after validation and conversion"""
    parameter: str
    original_value: float
    original_unit: str
    converted_value: float
    converted_unit: str
    date: date
    warnings: List[str] = []


class ShapValues(BaseModel):
    """SHAP values for feature importance"""
    dialysis: Dict[str, float] = Field(..., description="SHAP values for dialysis prediction")
    mortality: Dict[str, float] = Field(..., description="SHAP values for mortality prediction")


class ConfidenceIntervals(BaseModel):
    """95% confidence intervals for predictions"""
    dialysis_lower: List[float] = Field(..., description="Lower CI for dialysis risk")
    dialysis_upper: List[float] = Field(..., description="Upper CI for dialysis risk") 
    mortality_lower: List[float] = Field(..., description="Lower CI for mortality risk")
    mortality_upper: List[float] = Field(..., description="Upper CI for mortality risk")


class ClinicalBenchmarks(BaseModel):
    """Clinical benchmark information"""
    nephrology_referral_threshold: float = Field(0.05, description="5-year risk threshold for nephrology referral")
    multidisciplinary_care_threshold: float = Field(0.10, description="2-year risk threshold for multidisciplinary care")
    krt_preparation_threshold: float = Field(0.40, description="2-year risk threshold for KRT preparation")


class ModelInfo(BaseModel):
    """Model inference information"""
    ensemble_size: int = Field(..., description="Number of models in ensemble")
    model_types: Dict[str, int] = Field(..., description="Count of each model type")
    inference_time_ms: float = Field(..., description="Total inference time in milliseconds")
    preprocessing_time_ms: float = Field(..., description="Preprocessing time in milliseconds")
    sequence_length: int = Field(..., description="Length of input sequence used")


class PatientContext(BaseModel):
    """Patient context information"""
    age: float
    gender: str
    egfr: float = Field(..., description="Calculated eGFR")
    egfr_stage: str = Field(..., description="CKD stage based on eGFR")
    cci_score: int = Field(..., description="Charlson Comorbidity Index score")
    observation_period: float = Field(..., description="Period covered by data")


class PredictionResponse(BaseModel):
    """Prediction API response"""
    success: bool = Field(True, description="Whether prediction was successful")
    
    # Core predictions
    predictions: Dict[str, List[float]] = Field(..., description="Risk predictions")
    confidence_intervals: Optional[ConfidenceIntervals] = Field(None, description="95% confidence intervals")
    
    # Feature importance
    shap_values: Optional[ShapValues] = Field(None, description="SHAP feature importance values")
    
    # Clinical context
    patient_context: PatientContext = Field(..., description="Patient clinical context")
    clinical_benchmarks: ClinicalBenchmarks = Field(..., description="Clinical guideline thresholds")
    
    # Processing information
    processed_values: List[ProcessedLabValue] = Field(..., description="Processed laboratory values")
    validation_warnings: List[ValidationWarning] = Field([], description="Validation warnings")
    
    # Model information
    model_info: ModelInfo = Field(..., description="Model inference information")
    
    # Session tracking
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(..., description="Prediction timestamp")


class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = Field(False, description="Success flag")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    suggestions: Optional[List[str]] = Field(None, description="Suggested corrections")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    timestamp: float = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    models_loaded: int = Field(..., description="Number of models loaded")
    system_info: Optional[Dict[str, Any]] = Field(None, description="System information")


class ValidationResponse(BaseModel):
    """Validation-only response for form validation"""
    valid: bool = Field(..., description="Whether input is valid")
    warnings: List[ValidationWarning] = Field([], description="Validation warnings")
    egfr_info: Optional[Dict[str, Any]] = Field(None, description="eGFR calculation info")
    processed_values: Optional[List[ProcessedLabValue]] = Field(None, description="Processed values")
    errors: Optional[List[str]] = Field(None, description="Validation errors")