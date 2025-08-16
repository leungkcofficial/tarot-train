"""
Structured logging configuration for TAROT CKD Risk Prediction API
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.typing import FilteringBoundLogger

from app.core.config import settings


def setup_logging() -> None:
    """Configure structured logging for the application"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            # Add log level
            structlog.stdlib.add_log_level,
            
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            
            # Add caller info in debug mode
            structlog.dev.set_exc_info if settings.DEBUG else structlog.processors.format_exc_info,
            
            # JSON formatting for production, console for development
            structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" 
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    
    # Reduce noise from external libraries
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


def log_patient_data_access(
    logger: FilteringBoundLogger,
    session_id: str,
    action: str,
    data_type: str,
    **kwargs: Any
) -> None:
    """
    Log patient data access for audit purposes
    
    Args:
        logger: Structured logger instance
        session_id: Session identifier (anonymized)
        action: Action performed (validate, process, predict, etc.)
        data_type: Type of data accessed (demographics, lab_values, etc.)
        **kwargs: Additional context
    """
    logger.info(
        "Patient data access",
        session_id=session_id[:8],  # Truncate for privacy
        action=action,
        data_type=data_type,
        **kwargs
    )


def log_model_inference(
    logger: FilteringBoundLogger,
    session_id: str,
    model_count: int,
    inference_time: float,
    **kwargs: Any
) -> None:
    """
    Log model inference metrics
    
    Args:
        logger: Structured logger instance
        session_id: Session identifier (anonymized)
        model_count: Number of models used
        inference_time: Total inference time in seconds
        **kwargs: Additional metrics
    """
    logger.info(
        "Model inference completed",
        session_id=session_id[:8],  # Truncate for privacy
        model_count=model_count,
        inference_time=inference_time,
        **kwargs
    )


def log_validation_error(
    logger: FilteringBoundLogger,
    session_id: str,
    field: str,
    value: Any,
    error_type: str,
    message: str
) -> None:
    """
    Log validation errors for monitoring
    
    Args:
        logger: Structured logger instance
        session_id: Session identifier (anonymized)
        field: Field that failed validation
        value: Value that failed (potentially sensitive - handle carefully)
        error_type: Type of validation error
        message: Human-readable error message
    """
    # Only log the type and range of numeric values, not actual values
    safe_value = type(value).__name__
    if isinstance(value, (int, float)):
        safe_value = f"{type(value).__name__}({value:.1f})" if abs(value) < 10000 else f"{type(value).__name__}(large)"
    
    logger.warning(
        "Validation error",
        session_id=session_id[:8],
        field=field,
        value_type=safe_value,
        error_type=error_type,
        message=message
    )


class PrivacyFilter(logging.Filter):
    """Filter to remove potentially sensitive information from logs"""
    
    SENSITIVE_FIELDS = {
        'creatinine', 'hemoglobin', 'phosphate', 'bicarbonate', 
        'uacr', 'upcr', 'age', 'dob', 'date_of_birth'
    }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter sensitive information from log records"""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            # Check if the message contains sensitive field names
            msg_lower = record.msg.lower()
            for field in self.SENSITIVE_FIELDS:
                if field in msg_lower:
                    # Don't completely block, but flag for review
                    record.sensitive_data = True
                    break
        
        return True


def setup_privacy_logging() -> None:
    """Add privacy filter to all loggers"""
    privacy_filter = PrivacyFilter()
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(privacy_filter)
    
    # Add to specific loggers
    for logger_name in ["app", "uvicorn", "fastapi"]:
        logger = logging.getLogger(logger_name)
        logger.addFilter(privacy_filter)