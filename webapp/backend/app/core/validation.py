"""
Input validation and unit conversion for TAROT CKD Risk Prediction API
"""

import math
import numpy as np
from datetime import datetime, date
from typing import Dict, Tuple, Optional, Any, List, Union
from enum import Enum

from pydantic import BaseModel, Field, validator
from app.core.config import settings, VALIDATION_RANGES


class GenderEnum(str, Enum):
    """Gender enumeration"""
    MALE = "male"
    FEMALE = "female"


class UnitEnum(str, Enum):
    """Laboratory unit enumeration"""
    # Creatinine
    CREATININE_UMOL_L = "umol/L"
    CREATININE_MGDL = "mg/dL"
    
    # Hemoglobin
    HEMOGLOBIN_GDL = "g/dL"
    HEMOGLOBIN_GL = "g/L"
    
    # Phosphate
    PHOSPHATE_MMOL_L = "mmol/L"
    PHOSPHATE_MGDL = "mg/dL"
    
    # Bicarbonate
    BICARBONATE_MMOL_L = "mmol/L"
    BICARBONATE_MEQ_L = "mEq/L"
    
    # UACR/UPCR
    URINE_MGMMOL = "mg/mmol"
    URINE_MGG = "mg/g"


class ValidationError(ValueError):
    """Custom validation error with detailed information"""
    
    def __init__(self, field: str, value: Any, message: str, suggestion: Optional[str] = None):
        self.field = field
        self.value = value
        self.message = message
        self.suggestion = suggestion
        super().__init__(f"{field}: {message}")


def validate_age(age: float, date_of_birth: Optional[Union[date, str]] = None) -> float:
    """
    Validate age with minimum age requirement
    
    Args:
        age: Age in years
        date_of_birth: Optional date of birth for verification (date object or ISO string)
        
    Returns:
        Validated age
        
    Raises:
        ValidationError: If age is invalid
    """
    # Calculate age from DOB if provided
    if date_of_birth:
        today = date.today()
        
        # Handle string date
        if isinstance(date_of_birth, str):
            try:
                date_of_birth = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            except ValueError:
                # If we can't parse the date, just use the provided age
                pass
            except:
                pass
        
        # Only calculate if we have a valid date object
        if hasattr(date_of_birth, 'year'):
            calculated_age = today.year - date_of_birth.year - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))
        
        # Use calculated age if significantly different
        if abs(calculated_age - age) > 1:
            age = calculated_age
    
    # Validate minimum age
    if age < settings.MIN_AGE:
        raise ValidationError(
            field="age",
            value=age,
            message=f"Age must be ≥{settings.MIN_AGE} years. This tool is designed for adults.",
            suggestion="Please consult a pediatric nephrologist for patients under 18."
        )
    
    # Reasonable maximum age check
    if age > 120:
        raise ValidationError(
            field="age",
            value=age,
            message="Age seems unrealistic. Please verify the date of birth.",
            suggestion="Check that the date of birth is entered correctly."
        )
    
    return float(age)


def calculate_egfr(creatinine_mgdl: float, age: float, gender: str) -> float:
    """
    Calculate eGFR using CKD-EPI 2021 equation
    
    Based on: /mnt/dump/yard/projects/tarot2/src/data_cleaning.py:325-351
    
    Args:
        creatinine_mgdl: Serum creatinine in mg/dL
        age: Age in years
        gender: Gender ("male" or "female")
        
    Returns:
        eGFR in mL/min/1.73m²
    """
    # CKD-EPI 2021 parameters
    if gender.lower() == "female":
        kappa = 0.7
        alpha = -0.329
        gender_factor = 1.018
    else:  # male
        kappa = 0.9
        alpha = -0.411
        gender_factor = 1.0
    
    # Calculate eGFR
    egfr = (141 * 
            (min(creatinine_mgdl / kappa, 1) ** alpha) * 
            (max(creatinine_mgdl / kappa, 1) ** -1.209) * 
            (0.993 ** age) * 
            gender_factor)
    
    return egfr


def validate_egfr(creatinine: float, creatinine_unit: str, age: float, gender: str) -> Tuple[float, str]:
    """
    Validate eGFR is within acceptable range for CKD predictions
    
    Args:
        creatinine: Creatinine value
        creatinine_unit: Unit of creatinine measurement
        age: Age in years
        gender: Gender
        
    Returns:
        Tuple of (egfr, validation_message)
        
    Raises:
        ValidationError: If eGFR is outside acceptable range
    """
    # Convert creatinine to mg/dL if needed
    if creatinine_unit == "umol/L":
        creatinine_mgdl = creatinine / settings.CREATININE_MGDL_TO_UMOL_FACTOR
    else:
        creatinine_mgdl = creatinine
    
    # Calculate eGFR
    egfr = calculate_egfr(creatinine_mgdl, age, gender)
    
    # Validate eGFR range
    if egfr > settings.MAX_EGFR:
        raise ValidationError(
            field="egfr",
            value=egfr,
            message=f"eGFR {egfr:.1f} mL/min/1.73m² indicates normal/mild kidney function. This tool is for CKD patients with eGFR ≤{settings.MAX_EGFR}.",
            suggestion="This tool is designed for patients with moderate to severe CKD. Consider using KFRE calculator for higher eGFR values."
        )
    
    if egfr < settings.MIN_EGFR:
        raise ValidationError(
            field="egfr", 
            value=egfr,
            message=f"eGFR {egfr:.1f} mL/min/1.73m² indicates very advanced kidney disease. Please consult nephrology immediately for urgent care planning.",
            suggestion="Patients with eGFR <10 require immediate nephrology consultation and may need urgent dialysis preparation."
        )
    
    # Generate informational message
    if egfr < 15:
        message = f"eGFR {egfr:.1f} mL/min/1.73m² (CKD Stage 5) - Very high risk, urgent nephrology care recommended"
    elif egfr < 30:
        message = f"eGFR {egfr:.1f} mL/min/1.73m² (CKD Stage 4) - High risk, close monitoring recommended"
    elif egfr < 45:
        message = f"eGFR {egfr:.1f} mL/min/1.73m² (CKD Stage 3b) - Moderate-high risk"
    else:
        message = f"eGFR {egfr:.1f} mL/min/1.73m² (CKD Stage 3a) - Moderate risk"
    
    return egfr, message


def convert_units(value: float, from_unit: str, parameter: str) -> float:
    """
    Convert laboratory values between different units
    
    Args:
        value: Value to convert
        from_unit: Current unit
        parameter: Parameter name (creatinine, hemoglobin, etc.)
        
    Returns:
        Converted value in expected unit
    """
    expected_ranges = VALIDATION_RANGES.get(parameter)
    if not expected_ranges:
        return value
    
    # Normalize expected units to handle unicode symbols
    expected_unit = expected_ranges["unit"]
    expected_unit = expected_unit.replace('µ', 'u').replace('μ', 'u')
    
    alt_unit = expected_ranges["alt_unit"]
    alt_unit = alt_unit.replace('µ', 'u').replace('μ', 'u')
    
    conversion_factor = expected_ranges["conversion_factor"]
    
    # No conversion needed
    if from_unit == expected_unit:
        return value
    
    # Convert from alternative unit to expected unit
    if from_unit == alt_unit:
        if parameter == "hemoglobin" and from_unit == "g/L":
            return value * conversion_factor  # g/L to g/dL
        elif parameter == "creatinine" and from_unit == "mg/dL":
            return value * conversion_factor  # mg/dL to μmol/L
        elif parameter == "phosphate" and from_unit == "mg/dL":
            return value * conversion_factor  # mg/dL to mmol/L
        elif parameter in ["uacr", "upcr"] and from_unit == "mg/g":
            return value * conversion_factor  # mg/g to mg/mmol
        elif parameter == "bicarbonate":
            return value  # mEq/L = mmol/L
    
    raise ValueError(f"Unknown unit conversion: {from_unit} for {parameter}")


def validate_lab_value(parameter: str, value: float, unit: str) -> Tuple[float, List[str]]:
    """
    Validate laboratory value against expected ranges
    
    Args:
        parameter: Parameter name
        value: Value to validate
        unit: Unit of measurement
        
    Returns:
        Tuple of (converted_value, warnings)
        
    Raises:
        ValidationError: If value is outside acceptable range
    """
    warnings = []
    
    # Normalize unit formats (handle different unicode symbols)
    unit = unit.replace('µ', 'u')  # Convert µmol/L to umol/L
    unit = unit.replace('μ', 'u')  # Convert μmol/L to umol/L
    
    # Convert to expected unit
    try:
        converted_value = convert_units(value, unit, parameter)
    except ValueError as e:
        raise ValidationError(
            field=parameter,
            value=f"{value} {unit}",
            message=str(e),
            suggestion=f"Please use {VALIDATION_RANGES[parameter]['unit']} or {VALIDATION_RANGES[parameter]['alt_unit']}"
        )
    
    # Get validation range
    ranges = VALIDATION_RANGES.get(parameter)
    if not ranges:
        return converted_value, warnings
    
    min_val = ranges["min"]
    max_val = ranges["max"]
    expected_unit = ranges["unit"]
    
    # Validate range
    if converted_value < min_val:
        raise ValidationError(
            field=parameter,
            value=converted_value,
            message=f"{parameter.title()} {converted_value:.2f} {expected_unit} is below minimum expected value ({min_val} {expected_unit})",
            suggestion="Please verify the test result and unit. Very low values may indicate measurement error."
        )
    
    if converted_value > max_val:
        raise ValidationError(
            field=parameter,
            value=converted_value,
            message=f"{parameter.title()} {converted_value:.2f} {expected_unit} is above maximum expected value ({max_val} {expected_unit})",
            suggestion="Please verify the test result and unit. Very high values may indicate measurement error."
        )
    
    # Add warnings for borderline values
    if converted_value < min_val * 1.5:
        warnings.append(f"Low {parameter}: {converted_value:.2f} {expected_unit} - please verify accuracy")
    elif converted_value > max_val * 0.8:
        warnings.append(f"High {parameter}: {converted_value:.2f} {expected_unit} - please verify accuracy")
    
    return converted_value, warnings


def convert_upcr_to_uacr(upcr_mgmmol: float) -> float:
    """
    Convert UPCR to UACR using the formula from data_cleaning.py:1244-1292
    
    Args:
        upcr_mgmmol: UPCR value in mg/mmol
        
    Returns:
        Predicted UACR value in mg/mmol
    """
    # Convert to mg/g first
    pcr_mg_g = upcr_mgmmol * settings.URINE_MGMMOL_TO_MGG_FACTOR
    
    # Apply prediction formula
    log_min_pcr50 = math.log(min(pcr_mg_g / 50.0, 1.0))
    log_max_pcr500_01 = math.log(max(min(pcr_mg_g / 500.0, 1.0), 0.1))
    log_max_pcr500 = math.log(max(pcr_mg_g / 500.0, 1.0))
    
    acr_mg_g = math.exp(
        settings.ACR_PREDICTION_INTERCEPT + 
        settings.ACR_PREDICTION_COEF1 * log_min_pcr50 +
        settings.ACR_PREDICTION_COEF2 * log_max_pcr500_01 + 
        settings.ACR_PREDICTION_COEF3 * log_max_pcr500
    )
    
    # Convert back to mg/mmol
    return acr_mg_g / settings.URINE_MGMMOL_TO_MGG_FACTOR


def calculate_cci_score(comorbidities: Dict[str, bool]) -> int:
    """
    Calculate Charlson Comorbidity Index score
    
    Args:
        comorbidities: Dictionary of comorbidity flags
        
    Returns:
        CCI score
    """
    # CCI scoring weights
    cci_weights = {
        "myocardial_infarction": 1,
        "congestive_heart_failure": 1,
        "peripheral_vascular_disease": 1,
        "cerebrovascular_disease": 1,
        "dementia": 1,
        "chronic_pulmonary_disease": 1,
        "rheumatic_disease": 1,
        "peptic_ulcer_disease": 1,
        "mild_liver_disease": 1,
        "diabetes_wo_complication": 1,
        "renal_mild_moderate": 2,
        "diabetes_w_complication": 2,
        "hemiplegia_paraplegia": 2,
        "any_malignancy": 2,
        "liver_severe": 3,
        "renal_severe": 3,
        "metastatic_cancer": 6,
        "hiv": 6,
        "aids": 6
    }
    
    score = 0
    for condition, present in comorbidities.items():
        if present and condition in cci_weights:
            score += cci_weights[condition]
    
    return score


def validate_date_range(test_date: Union[date, str], current_date: Optional[date] = None) -> date:
    """
    Validate that test dates are reasonable
    
    Args:
        test_date: Date of test (date object or ISO string)
        current_date: Current date (defaults to today)
        
    Returns:
        Validated date
        
    Raises:
        ValidationError: If date is unreasonable
    """
    if current_date is None:
        current_date = date.today()
    
    # Handle string date
    if isinstance(test_date, str):
        try:
            test_date = datetime.strptime(test_date, '%Y-%m-%d').date()
        except ValueError:
            raise ValidationError(
                field="date", 
                message="Invalid date format. Use YYYY-MM-DD.", 
                suggestion="Provide date in YYYY-MM-DD format"
            )
    
    # Check for future dates
    if test_date > current_date:
        raise ValidationError(
            field="date",
            value=test_date,
            message="Test date cannot be in the future",
            suggestion="Please check the date is entered correctly"
        )
    
    # Check for very old dates (more than 20 years ago)
    years_ago = (current_date - test_date).days / 365.25
    if years_ago > 20:
        raise ValidationError(
            field="date",
            value=test_date,
            message=f"Test date is {years_ago:.1f} years ago - this may be too old for accurate predictions",
            suggestion="Consider using more recent test results if available"
        )
    
    return test_date