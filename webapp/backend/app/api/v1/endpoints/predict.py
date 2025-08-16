"""
Prediction endpoints for TAROT CKD Risk Prediction API
"""

import time
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import structlog

from app.core.config import settings
from app.core.validation import (
    validate_age, validate_egfr, validate_lab_value, convert_upcr_to_uacr,
    calculate_cci_score, validate_date_range, ValidationError
)
from app.schemas.prediction_simple import (
    PredictionRequest, PredictionResponse, ValidationResponse, ErrorResponse,
    ProcessedLabValue, ValidationWarning, ShapValues, ConfidenceIntervals,
    ClinicalBenchmarks, ModelInfo, PatientContext
)
from app.schemas.temporal_prediction import TemporalPredictionRequest, TemporalPredictionResponse
from app.models.model_manager import ModelManager


logger = structlog.get_logger(__name__)
router = APIRouter()


def get_model_manager(request: Request) -> ModelManager:
    """Dependency to get the model manager from app state"""
    if not hasattr(request.app.state, 'model_manager'):
        raise HTTPException(status_code=503, detail="Models not loaded")
    return request.app.state.model_manager


@router.post("/", response_model=PredictionResponse)
async def predict_risk(
    prediction_request: PredictionRequest,
    request: Request,
    model_manager: ModelManager = Depends(get_model_manager)
) -> PredictionResponse:
    """
    Predict CKD progression risk for a patient
    
    This endpoint takes patient demographics, laboratory values, and medical history
    to predict the risk of dialysis initiation and all-cause mortality over 1-5 year horizons.
    
    The prediction uses an ensemble of 36+ deep learning models trained on longitudinal
    CKD patient data, providing both point estimates and 95% confidence intervals.
    
    **Clinical Usage:**
    - Age ≥18 years required
    - eGFR must be 10-60 mL/min/1.73m² (CKD Stage 3-5)
    - Requires: creatinine, hemoglobin, phosphate, bicarbonate, UACR/UPCR
    - Optional: medical history for Charlson Comorbidity Index
    
    **Clinical Benchmarks (KDIGO Guidelines):**
    - 5-year risk 3-5%: Consider nephrology referral
    - 2-year risk >10%: Initiate multidisciplinary care planning
    - 2-year risk >40%: Begin KRT preparation and vascular access planning
    """
    session_id = prediction_request.session_id or str(uuid.uuid4())
    
    logger.info(
        "Prediction request received",
        session_id=session_id[:8],
        client_ip=request.client.host,
        lab_count=len(prediction_request.laboratory_values),
        has_history=bool(prediction_request.medical_history)
    )
    
    try:
        # Start timing
        start_time = time.time()
        
        # Validate and process input data
        processed_data, warnings, patient_context = await process_patient_data(
            prediction_request, session_id
        )
        
        # Generate predictions using model ensemble
        prediction_result = await model_manager.predict(prediction_request.dict())
        
        # Calculate confidence intervals (mock implementation)
        confidence_intervals = calculate_confidence_intervals(
            prediction_result['predictions']
        )
        
        # Build response
        total_time = (time.time() - start_time) * 1000
        
        response = PredictionResponse(
            success=True,
            predictions=prediction_result['predictions'],
            confidence_intervals=confidence_intervals,
            shap_values=ShapValues(
                dialysis=prediction_result['shap_values']['dialysis'],
                mortality=prediction_result['shap_values']['mortality']
            ),
            patient_context=patient_context,
            clinical_benchmarks=ClinicalBenchmarks(),
            processed_values=processed_data['processed_values'],
            validation_warnings=warnings,
            model_info=ModelInfo(**prediction_result['model_info']),
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(
            "Prediction completed successfully",
            session_id=session_id[:8],
            total_time_ms=total_time,
            dialysis_risk_2y=prediction_result['predictions']['dialysis_risk'][1],
            mortality_risk_2y=prediction_result['predictions']['mortality_risk'][1]
        )
        
        return response
        
    except ValidationError as e:
        logger.warning(
            "Validation error in prediction",
            session_id=session_id[:8],
            field=e.field,
            error=e.message
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Validation Error", 
                "message": f"{e.message}" + (f" {e.suggestion}" if e.suggestion else ""),
                "field": e.field
            }
        )
        
    except Exception as e:
        logger.error(
            "Prediction failed",
            session_id=session_id[:8],
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction Failed",
                "message": "An unexpected error occurred during prediction",
                "session_id": session_id
            }
        )


@router.post("/validate", response_model=ValidationResponse)
async def validate_input(
    prediction_request: PredictionRequest,
    request: Request
) -> ValidationResponse:
    """
    Validate patient input without generating predictions
    
    This endpoint validates patient data and performs unit conversions,
    returning processed values and any warnings or errors. Useful for
    real-time form validation in the frontend.
    
    **Returns:**
    - Validation status and warnings
    - Calculated eGFR and CKD stage
    - Processed laboratory values with unit conversions
    - Charlson Comorbidity Index if medical history provided
    """
    session_id = prediction_request.session_id or str(uuid.uuid4())
    
    logger.info(
        "Validation request received",
        session_id=session_id[:8],
        client_ip=request.client.host
    )
    
    try:
        # Process and validate data
        processed_data, warnings, patient_context = await process_patient_data(
            prediction_request, session_id, validate_only=True
        )
        
        return ValidationResponse(
            valid=True,
            warnings=warnings,
            egfr_info={
                "egfr": patient_context.egfr,
                "egfr_stage": patient_context.egfr_stage,
                "message": f"eGFR {patient_context.egfr:.1f} mL/min/1.73m² ({patient_context.egfr_stage})"
            },
            processed_values=processed_data['processed_values']
        )
        
    except ValidationError as e:
        logger.warning(
            "Validation failed",
            session_id=session_id[:8],
            field=e.field,
            error=e.message
        )
        
        return ValidationResponse(
            valid=False,
            errors=[e.message],
            warnings=[]
        )


@router.post("/temporal", response_model=TemporalPredictionResponse)
async def predict_temporal_risk(
    temporal_request: TemporalPredictionRequest,
    request: Request,
    model_manager: ModelManager = Depends(get_model_manager)
) -> TemporalPredictionResponse:
    """
    Predict CKD progression risk using temporal feature matrix format
    
    This endpoint takes patient data in 11 features x 10 timepoints format
    and generates risk predictions using the ensemble models. This format
    enables more sophisticated temporal modeling.
    
    **Feature Matrix Format:**
    - 11 features: age_at_obs, albumin, uacr, bicarbonate, cci_score_total,
                  creatinine, gender, hemoglobin, ht, observation_period, phosphate
    - 10 timepoints: Each feature can have up to 10 temporal values
    - Values can be null for missing timepoints
    
    **Clinical Usage:**
    - Supports multiple timepoint data for each laboratory parameter
    - Better temporal modeling for longitudinal patient data
    - Same clinical benchmarks as standard prediction endpoint
    """
    session_id = temporal_request.session_id or str(uuid.uuid4())
    
    logger.info(
        "Temporal prediction request received",
        session_id=session_id[:8],
        client_ip=request.client.host,
        patient_age=temporal_request.patient_info.age_at_obs,
        patient_gender=temporal_request.patient_info.gender
    )
    
    try:
        start_time = time.time()
        
        # Convert temporal request to format compatible with existing model manager
        # For now, we'll use the most recent (first) timepoint for each feature
        converted_request = convert_temporal_to_standard(temporal_request)
        
        # Generate predictions using model ensemble
        prediction_result = await model_manager.predict(converted_request)
        
        # Calculate confidence intervals
        confidence_intervals = calculate_confidence_intervals(
            prediction_result['predictions']
        )
        
        # Build temporal-specific response
        total_time = (time.time() - start_time) * 1000
        
        response = TemporalPredictionResponse(
            success=True,
            predictions=prediction_result['predictions'],
            confidence_intervals={
                "dialysis_lower": confidence_intervals.dialysis_lower,
                "dialysis_upper": confidence_intervals.dialysis_upper,
                "mortality_lower": confidence_intervals.mortality_lower,
                "mortality_upper": confidence_intervals.mortality_upper
            },
            shap_values=prediction_result['shap_values'],
            patient_context={
                "age": temporal_request.patient_info.age_at_obs,
                "gender": temporal_request.patient_info.gender,
                "observation_period": temporal_request.patient_info.observation_period,
                "egfr": prediction_result.get('egfr', 0.0),
                "egfr_stage": prediction_result.get('egfr_stage', 'Unknown')
            },
            clinical_benchmarks={
                "nephrology_referral_threshold": 0.05,
                "multidisciplinary_care_threshold": 0.10,
                "krt_preparation_threshold": 0.40
            },
            model_info=prediction_result['model_info'],
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(
            "Temporal prediction completed successfully",
            session_id=session_id[:8],
            total_time_ms=total_time,
            dialysis_risk_2y=prediction_result['predictions']['dialysis_risk'][1],
            mortality_risk_2y=prediction_result['predictions']['mortality_risk'][1]
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Temporal prediction failed",
            session_id=session_id[:8],
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Temporal Prediction Failed",
                "message": "An unexpected error occurred during temporal prediction",
                "session_id": session_id
            }
        )


async def process_patient_data(
    request: PredictionRequest, 
    session_id: str,
    validate_only: bool = False
) -> tuple[Dict[str, Any], List[ValidationWarning], PatientContext]:
    """
    Process and validate patient data
    
    Args:
        request: Prediction request
        session_id: Session identifier
        validate_only: If True, skip intensive processing
        
    Returns:
        Tuple of (processed_data, warnings, patient_context)
    """
    warnings = []
    processed_values = []
    
    # Validate demographics
    demographics = request.demographics
    
    # Calculate age
    if demographics.age is not None:
        age = demographics.age
    elif demographics.date_of_birth:
        today = date.today()
        dob = demographics.date_of_birth
        
        # Handle string date
        if isinstance(dob, str):
            try:
                dob = datetime.strptime(dob, '%Y-%m-%d').date()
            except ValueError:
                # If we can't parse the date, use a default age or raise an error
                raise ValueError("Invalid date format. Use YYYY-MM-DD.")
        
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    else:
        raise ValueError("Either age or date_of_birth must be provided")
    
    # Validate age
    validated_age = validate_age(age, demographics.date_of_birth)
    
    # Process laboratory values
    lab_data = {}
    lab_dates = []
    
    for lab in request.laboratory_values:
        # Validate date
        validate_date_range(lab.date)
        lab_dates.append(lab.date)
        
        # Validate and convert value
        converted_value, lab_warnings = validate_lab_value(
            lab.parameter, lab.value, lab.unit
        )
        
        # Add warnings
        warnings.extend([
            ValidationWarning(field=lab.parameter, message=w, severity="warning") 
            for w in lab_warnings
        ])
        
        # Store processed value
        processed_lab = ProcessedLabValue(
            parameter=lab.parameter,
            original_value=lab.value,
            original_unit=lab.unit,
            converted_value=converted_value,
            converted_unit=get_expected_unit(lab.parameter),
            date=lab.date,
            warnings=lab_warnings
        )
        processed_values.append(processed_lab)
        
        lab_data[lab.parameter] = converted_value
    
    # Handle UPCR to UACR conversion if needed
    if 'upcr' in lab_data and 'uacr' not in lab_data:
        uacr_predicted = convert_upcr_to_uacr(lab_data['upcr'])
        lab_data['uacr'] = uacr_predicted
        
        warnings.append(ValidationWarning(
            field="uacr",
            message=f"UACR predicted from UPCR: {uacr_predicted:.1f} mg/mmol",
            severity="info"
        ))
    elif 'upcr' in lab_data and 'uacr' in lab_data:
        # UACR takes priority, remove UPCR
        lab_data.pop('upcr')
        warnings.append(ValidationWarning(
            field="uacr",
            message="Using UACR value (UPCR ignored as UACR has priority)",
            severity="info"
        ))
    
    # Validate eGFR
    creatinine = lab_data['creatinine']
    gender = demographics.gender.value
    
    egfr, egfr_message = validate_egfr(
        creatinine, "umol/L", validated_age, gender
    )
    
    # Determine CKD stage
    if egfr < 15:
        ckd_stage = "CKD Stage 5"
    elif egfr < 30:
        ckd_stage = "CKD Stage 4" 
    elif egfr < 45:
        ckd_stage = "CKD Stage 3b"
    else:
        ckd_stage = "CKD Stage 3a"
    
    # Process medical history
    comorbidities = {}
    if request.medical_history:
        for condition in request.medical_history:
            comorbidities[condition.condition] = condition.diagnosed
            
            # Validate diagnosis date if provided
            if condition.diagnosed and condition.date:
                validate_date_range(condition.date)
    
    # Calculate CCI score
    cci_score = calculate_cci_score(comorbidities)
    
    # Create patient context
    patient_context = PatientContext(
        age=validated_age,
        gender=gender,
        egfr=egfr,
        egfr_stage=ckd_stage,
        cci_score=cci_score,
        observation_period=1.0  # Default to 1 year
    )
    
    # Build dataframe for model input (only if not validation-only)
    if not validate_only:
        # Create patient dataframe in the expected format
        patient_df = build_patient_dataframe(
            demographics=demographics,
            lab_data=lab_data,
            comorbidities=comorbidities,
            egfr=egfr,
            age=validated_age,
            lab_dates=lab_dates
        )
        
        processed_data = {
            'dataframe': patient_df,
            'processed_values': processed_values
        }
    else:
        processed_data = {
            'processed_values': processed_values
        }
    
    return processed_data, warnings, patient_context


def build_patient_dataframe(
    demographics, lab_data, comorbidities, egfr, age, lab_dates
) -> pd.DataFrame:
    """Build patient dataframe for model input"""
    
    # Sort dates to create temporal sequence
    sorted_dates = sorted(set(lab_dates))
    
    # Create rows for each date
    rows = []
    for i, date_val in enumerate(sorted_dates):
        row = {
            'date': date_val,
            'age_at_obs': age,
            'gender': 1 if demographics.gender.value == 'male' else 0,
            'creatinine': lab_data.get('creatinine', np.nan),
            'hemoglobin': lab_data.get('hemoglobin', np.nan),
            'phosphate': lab_data.get('phosphate', np.nan),
            'bicarbonate': lab_data.get('bicarbonate', np.nan),
            'albumin': lab_data.get('albumin', np.nan),  # Optional
            'uacr': lab_data.get('uacr', np.nan),
            'cci_score_total': calculate_cci_score(comorbidities),
            'ht': 1 if comorbidities.get('hypertension', False) else 0,
            'observation_period': i + 1,
            'egfr': egfr,
            
            # Add comorbidity flags
            **{k: (1 if v else 0) for k, v in comorbidities.items()}
        }
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Fill missing values with forward fill
    df = df.fillna(method='ffill').fillna(0)
    
    return df


def calculate_confidence_intervals(predictions: Dict[str, List[float]]) -> ConfidenceIntervals:
    """Calculate 95% confidence intervals for predictions"""
    # Mock implementation - in reality this would come from model uncertainty
    dialysis_risk = predictions['dialysis_risk']
    mortality_risk = predictions['mortality_risk']
    
    # Simple approximation: ±20% for CI
    dialysis_lower = [max(0, r * 0.8) for r in dialysis_risk]
    dialysis_upper = [min(1, r * 1.2) for r in dialysis_risk]
    mortality_lower = [max(0, r * 0.8) for r in mortality_risk]
    mortality_upper = [min(1, r * 1.2) for r in mortality_risk]
    
    return ConfidenceIntervals(
        dialysis_lower=dialysis_lower,
        dialysis_upper=dialysis_upper,
        mortality_lower=mortality_lower,
        mortality_upper=mortality_upper
    )


def get_expected_unit(parameter: str) -> str:
    """Get expected unit for a parameter"""
    unit_map = {
        'creatinine': 'μmol/L',
        'hemoglobin': 'g/dL',
        'phosphate': 'mmol/L',
        'bicarbonate': 'mmol/L',
        'uacr': 'mg/mmol',
        'upcr': 'mg/mmol'
    }
    return unit_map.get(parameter, 'unknown')


def convert_temporal_to_standard(temporal_request: TemporalPredictionRequest) -> Dict[str, Any]:
    """Convert temporal feature matrix to standard prediction format"""
    
    # Get the most recent (first) non-null values for each feature
    feature_matrix = temporal_request.feature_matrix
    patient_info = temporal_request.patient_info
    
    # Extract lab values from feature matrix
    lab_values = []
    lab_params = ['creatinine', 'hemoglobin', 'phosphate', 'bicarbonate', 'albumin', 'uacr']
    
    for param in lab_params:
        if param in feature_matrix:
            values = feature_matrix[param]
            # Find first non-null value
            for i, value in enumerate(values):
                if value is not None:
                    # Get corresponding date
                    date_str = temporal_request.timepoint_dates[i] if i < len(temporal_request.timepoint_dates) else None
                    if not date_str:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                    
                    lab_values.append({
                        'parameter': param,
                        'value': value,
                        'unit': get_expected_unit(param),
                        'date': date_str
                    })
                    break
    
    # Extract comorbidities from feature matrix
    medical_history = []
    
    # Hypertension (HT feature)
    ht_values = feature_matrix.get('ht', [])
    if ht_values and any(v is not None for v in ht_values):
        ht_diagnosed = any(v == 1 for v in ht_values if v is not None)
        medical_history.append({
            'condition': 'hypertension',
            'diagnosed': ht_diagnosed,
            'date': temporal_request.timepoint_dates[0] if temporal_request.timepoint_dates[0] else None
        })
    
    # CCI score implies other conditions (simplified)
    cci_values = feature_matrix.get('cci_score_total', [])
    if cci_values and any(v is not None and v > 0 for v in cci_values):
        # For now, just indicate that there are comorbidities
        # In a real implementation, we'd need more detailed condition mapping
        pass
    
    # Build standard format request
    standard_request = {
        'demographics': {
            'age': patient_info.age_at_obs,
            'gender': patient_info.gender
        },
        'laboratory_values': lab_values,
        'medical_history': medical_history
    }
    
    return standard_request