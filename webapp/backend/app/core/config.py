"""
Configuration settings for TAROT CKD Risk Prediction API
"""

import os
from typing import List, Optional
from pydantic import Field

try:
    # Pydantic v2
    from pydantic_settings import BaseSettings
except ImportError:
    # Pydantic v1
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "TAROT CKD Risk Prediction API"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    
    # API
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for session management"
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins"
    )
    
    # Models
    MODEL_PATH: str = Field(
        default="/mnt/dump/yard/projects/tarot2/foundation_models",
        description="Path to model files"
    )
    PREPROCESSOR_PATH: str = Field(
        default="/mnt/dump/yard/projects/tarot2/foundation_models/ckd_preprocessor.pkl",
        description="Path to preprocessor file"
    )
    
    # Session Management
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis URL for session storage (optional)"
    )
    SESSION_EXPIRE_SECONDS: int = Field(
        default=3600,
        description="Session expiration time in seconds"
    )
    
    # Validation Parameters
    MIN_AGE: int = Field(default=18, description="Minimum age for predictions")
    MIN_EGFR: float = Field(default=10.0, description="Minimum eGFR (mL/min/1.73m²)")
    MAX_EGFR: float = Field(default=60.0, description="Maximum eGFR (mL/min/1.73m²)")
    
    # Laboratory Value Ranges (based on default_data_validation_rules.yml)
    CREATININE_MIN: float = Field(default=10.0, description="Min creatinine (μmol/L)")
    CREATININE_MAX: float = Field(default=3000.0, description="Max creatinine (μmol/L)")
    
    HEMOGLOBIN_MIN: float = Field(default=3.0, description="Min hemoglobin (g/dL)")
    HEMOGLOBIN_MAX: float = Field(default=25.0, description="Max hemoglobin (g/dL)")
    
    PHOSPHATE_MIN: float = Field(default=0.1, description="Min phosphate (mmol/L)")
    PHOSPHATE_MAX: float = Field(default=5.0, description="Max phosphate (mmol/L)")
    
    BICARBONATE_MIN: float = Field(default=1.0, description="Min bicarbonate (mmol/L)")
    BICARBONATE_MAX: float = Field(default=50.0, description="Max bicarbonate (mmol/L)")
    
    UACR_MIN: float = Field(default=0.0, description="Min UACR (mg/mmol)")
    UACR_MAX: float = Field(default=10000.0, description="Max UACR (mg/mmol)")
    
    UPCR_MIN: float = Field(default=0.0, description="Min UPCR (mg/mmol)")
    UPCR_MAX: float = Field(default=10000.0, description="Max UPCR (mg/mmol)")
    
    # Unit Conversion Factors
    # Creatinine: mg/dL to μmol/L
    CREATININE_MGDL_TO_UMOL_FACTOR: float = Field(default=88.4)
    
    # Hemoglobin: g/L to g/dL
    HEMOGLOBIN_GL_TO_GDL_FACTOR: float = Field(default=0.1)
    
    # Phosphate: mg/dL to mmol/L
    PHOSPHATE_MGDL_TO_MMOL_FACTOR: float = Field(default=0.3229)
    
    # UACR/UPCR: mg/g to mg/mmol
    URINE_MGG_TO_MGMMOL_FACTOR: float = Field(default=0.113)
    
    # UPCR to UACR conversion parameters (from data_cleaning.py)
    URINE_MGMMOL_TO_MGG_FACTOR: float = Field(default=8.84)
    ACR_PREDICTION_INTERCEPT: float = Field(default=5.3920)
    ACR_PREDICTION_COEF1: float = Field(default=0.3072)
    ACR_PREDICTION_COEF2: float = Field(default=1.5793)
    ACR_PREDICTION_COEF3: float = Field(default=1.1266)
    
    # Performance
    MAX_WORKERS: int = Field(default=4, description="Max worker threads")
    INFERENCE_TIMEOUT_SECONDS: float = Field(default=30.0, description="Inference timeout")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Log level")
    LOG_FORMAT: str = Field(default="json", description="Log format (json/text)")
    
    # Clinical Benchmarks (KDIGO guidelines)
    NEPHROLOGY_REFERRAL_THRESHOLD_5Y: float = Field(default=0.05, description="5-year risk for nephrology referral (5%)")
    MULTIDISCIPLINARY_CARE_THRESHOLD_2Y: float = Field(default=0.10, description="2-year risk for multidisciplinary care (10%)")
    KRT_PREPARATION_THRESHOLD_2Y: float = Field(default=0.40, description="2-year risk for KRT preparation (40%)")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()


# Validation ranges for easy access
VALIDATION_RANGES = {
    "creatinine": {
        "min": settings.CREATININE_MIN,
        "max": settings.CREATININE_MAX,
        "unit": "μmol/L",
        "alt_unit": "mg/dL",
        "conversion_factor": settings.CREATININE_MGDL_TO_UMOL_FACTOR
    },
    "hemoglobin": {
        "min": settings.HEMOGLOBIN_MIN,
        "max": settings.HEMOGLOBIN_MAX,
        "unit": "g/dL",
        "alt_unit": "g/L",
        "conversion_factor": settings.HEMOGLOBIN_GL_TO_GDL_FACTOR
    },
    "phosphate": {
        "min": settings.PHOSPHATE_MIN,
        "max": settings.PHOSPHATE_MAX,
        "unit": "mmol/L",
        "alt_unit": "mg/dL",
        "conversion_factor": settings.PHOSPHATE_MGDL_TO_MMOL_FACTOR
    },
    "bicarbonate": {
        "min": settings.BICARBONATE_MIN,
        "max": settings.BICARBONATE_MAX,
        "unit": "mmol/L",
        "alt_unit": "mEq/L",
        "conversion_factor": 1.0  # Same values
    },
    "uacr": {
        "min": settings.UACR_MIN,
        "max": settings.UACR_MAX,
        "unit": "mg/mmol",
        "alt_unit": "mg/g",
        "conversion_factor": settings.URINE_MGG_TO_MGMMOL_FACTOR
    },
    "upcr": {
        "min": settings.UPCR_MIN,
        "max": settings.UPCR_MAX,
        "unit": "mg/mmol",
        "alt_unit": "mg/g", 
        "conversion_factor": settings.URINE_MGG_TO_MGMMOL_FACTOR
    }
}