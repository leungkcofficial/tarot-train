"""
Static information endpoints for TAROT CKD Risk Prediction API
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import structlog

from app.core.config import settings


logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get model performance metrics
    
    Returns detailed performance comparison between the ensemble model
    and traditional KFRE models, including:
    - C-index scores with 95% confidence intervals
    - Brier scores
    - Index of Prediction Accuracy (IPA)
    - Statistical significance tests
    
    Based on validation study results from ensemble_kfre_comparison.
    """
    # Performance data from /mnt/dump/yard/projects/tarot2/results/ensemble_kfre_comparison/results_table.md
    performance_data = {
        "study_info": {
            "title": "Model Performance Comparison: Ensemble vs KFRE Models",
            "description": "Comprehensive evaluation of the TAROT ensemble model against traditional KFRE calculators",
            "datasets": ["Temporal", "Spatial"],
            "time_horizons": ["2-year", "5-year"],
            "sample_size": "Large multi-center CKD cohort",
            "validation_method": "External validation with 95% confidence intervals"
        },
        "metrics": {
            "temporal_2year": {
                "ensemble": {
                    "c_index": 0.8495,
                    "c_index_ci": [0.8425, 0.8563],
                    "brier_score": 0.0188,
                    "brier_score_ci": [0.0180, 0.0195],
                    "ipa": 0.5883,
                    "ipa_ci": [0.5566, 0.6222]
                },
                "kfre_4v": {
                    "c_index": 0.7795,
                    "c_index_ci": [0.7709, 0.7890],
                    "brier_score": 0.0447,
                    "brier_score_ci": [0.0428, 0.0464],
                    "ipa": 0.0199,
                    "ipa_ci": [-0.0532, 0.1013],
                    "p_values": {"c_index": 0.001, "brier_score": 0.001, "ipa": 0.001}
                },
                "kfre_8v": {
                    "c_index": 0.7902,
                    "c_index_ci": [0.7810, 0.8003],
                    "brier_score": 0.0458,
                    "brier_score_ci": [0.0441, 0.0473],
                    "ipa": -0.0047,
                    "ipa_ci": [-0.0756, 0.0722],
                    "p_values": {"c_index": 0.001, "brier_score": 0.001, "ipa": 0.001}
                }
            },
            "temporal_5year": {
                "ensemble": {
                    "c_index": 0.8537,
                    "c_index_ci": [0.8470, 0.8600],
                    "brier_score": 0.0276,
                    "brier_score_ci": [0.0269, 0.0283],
                    "ipa": 0.3937,
                    "ipa_ci": [0.3492, 0.4329]
                },
                "kfre_4v": {
                    "c_index": 0.7794,
                    "c_index_ci": [0.7710, 0.7876],
                    "brier_score": 0.0421,
                    "brier_score_ci": [0.0396, 0.0442],
                    "ipa": 0.0750,
                    "ipa_ci": [-0.0176, 0.1648],
                    "p_values": {"c_index": 0.001, "brier_score": 0.001, "ipa": 0.001}
                },
                "kfre_8v": {
                    "c_index": 0.7893,
                    "c_index_ci": [0.7815, 0.7968],
                    "brier_score": 0.0452,
                    "brier_score_ci": [0.0437, 0.0469],
                    "ipa": 0.0067,
                    "ipa_ci": [-0.0800, 0.0783],
                    "p_values": {"c_index": 0.001, "brier_score": 0.001, "ipa": 0.001}
                }
            },
            "spatial_2year": {
                "ensemble": {
                    "c_index": 0.7723,
                    "c_index_ci": [0.7704, 0.7739],
                    "brier_score": 0.0544,
                    "brier_score_ci": [0.0538, 0.0553],
                    "ipa": 0.6449,
                    "ipa_ci": [0.6349, 0.6517]
                },
                "kfre_4v": {
                    "c_index": 0.7482,
                    "c_index_ci": [0.7461, 0.7505],
                    "brier_score": 0.1514,
                    "brier_score_ci": [0.1498, 0.1530],
                    "ipa": 0.0113,
                    "ipa_ci": [-0.0093, 0.0296],
                    "p_values": {"c_index": 0.001, "brier_score": 0.001, "ipa": 0.001}
                }
            },
            "spatial_5year": {
                "ensemble": {
                    "c_index": 0.7859,
                    "c_index_ci": [0.7833, 0.7886],
                    "brier_score": 0.0718,
                    "brier_score_ci": [0.0711, 0.0727],
                    "ipa": 0.6158,
                    "ipa_ci": [0.6074, 0.6236]
                },
                "kfre_4v": {
                    "c_index": 0.7482,
                    "c_index_ci": [0.7456, 0.7507],
                    "brier_score": 0.1787,
                    "brier_score_ci": [0.1769, 0.1803],
                    "ipa": 0.0442,
                    "ipa_ci": [0.0256, 0.0631],
                    "p_values": {"c_index": 0.001, "brier_score": 0.001, "ipa": 0.001}
                }
            }
        },
        "key_findings": [
            "Ensemble model consistently outperforms KFRE models across all metrics and time points",
            "Statistical significance: p < 0.001 for all comparisons vs. KFRE",
            "Ensemble shows substantial improvement over null model (IPA 39-65%)",
            "KFRE models show minimal or negative improvement vs. null model",
            "Performance gap more pronounced in spatial dataset validation"
        ],
        "clinical_interpretation": {
            "c_index": "Discrimination ability - how well the model ranks patients by risk",
            "brier_score": "Calibration accuracy - lower scores indicate better calibration",
            "ipa": "Index of Prediction Accuracy - improvement over null model",
            "confidence_intervals": "95% CI provide uncertainty estimates for clinical decision-making"
        }
    }
    
    return performance_data


@router.get("/disclaimer")
async def get_clinical_disclaimer() -> Dict[str, Any]:
    """
    Get clinical disclaimer and usage guidelines
    
    Returns important information about:
    - Clinical usage limitations
    - Professional supervision requirements
    - Liability disclaimers
    - Model limitations and assumptions
    """
    disclaimer_content = {
        "title": "TAROT CKD Risk Prediction - Clinical Disclaimer",
        "last_updated": "2024-01-01",
        "sections": {
            "clinical_decision_support": {
                "title": "Clinical Decision Support Tool",
                "content": [
                    "This tool is designed as a clinical decision support aid and should NOT replace clinical judgment.",
                    "All predictions should be interpreted by qualified healthcare professionals.",
                    "Risk predictions are estimates based on population data and may not apply to individual patients.",
                    "Clinical context, patient preferences, and additional factors not captured by the model should always be considered."
                ]
            },
            "intended_use": {
                "title": "Intended Use",
                "content": [
                    "Designed for healthcare professionals treating adult CKD patients (age ≥18 years).",
                    "Applicable to patients with CKD Stage 3-5 (eGFR 10-60 mL/min/1.73m²).",
                    "Intended for risk stratification and care planning discussions.",
                    "Should complement, not replace, established clinical guidelines (KDIGO, KDOQI)."
                ]
            },
            "limitations": {
                "title": "Model Limitations",
                "content": [
                    "Based on specific patient populations and may not generalize to all ethnicities or regions.",
                    "Requires accurate laboratory values and complete medical history for optimal performance.",
                    "Cannot account for rapid changes in clinical status or acute kidney injury.",
                    "UPCR-to-UACR conversion is an approximation and may introduce uncertainty.",
                    "Model performance may vary in patients with rare conditions or extreme laboratory values."
                ]
            },
            "validation_data": {
                "title": "Validation and Performance",
                "content": [
                    "Model trained and validated on longitudinal CKD patient data from multiple centers.",
                    "Performance metrics available through /api/v1/info/performance endpoint.",
                    "External validation demonstrates superior performance vs. traditional KFRE calculators.",
                    "Confidence intervals provided to support clinical decision-making under uncertainty."
                ]
            },
            "privacy_security": {
                "title": "Privacy and Data Security",
                "content": [
                    "NO patient data is stored, logged, or transmitted to external systems.",
                    "All processing occurs in temporary session memory and is immediately discarded.",
                    "Healthcare institutions should ensure compliance with local privacy regulations.",
                    "Session IDs are anonymized and do not contain patient identifiers."
                ]
            },
            "professional_responsibility": {
                "title": "Professional Responsibility", 
                "content": [
                    "Healthcare providers remain fully responsible for all clinical decisions.",
                    "Risk predictions should be discussed with patients in appropriate clinical context.",
                    "Emergency situations require immediate medical attention regardless of model predictions.",
                    "Regular monitoring and reassessment are essential for CKD patients."
                ]
            },
            "liability": {
                "title": "Liability Disclaimer",
                "content": [
                    "This tool is provided 'as is' without warranties of any kind.",
                    "The developers and institutions are not liable for clinical decisions or patient outcomes.",
                    "Users assume full responsibility for appropriate clinical use and interpretation.",
                    "Healthcare providers should maintain appropriate professional liability coverage."
                ]
            }
        },
        "contact_info": {
            "support": "For technical support, see documentation at /docs/",
            "clinical_questions": "Clinical questions should be directed to qualified nephrologists",
            "reporting_issues": "Report technical issues through appropriate channels"
        },
        "regulatory_note": "This tool has not been evaluated by regulatory agencies (FDA, Health Canada, etc.) for diagnostic use."
    }
    
    return disclaimer_content


@router.get("/disclaimer/html", response_class=HTMLResponse)
async def get_disclaimer_html() -> str:
    """Get disclaimer as formatted HTML page"""
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TAROT CKD Risk Prediction - Clinical Disclaimer</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
            h1 { color: #d32f2f; border-bottom: 2px solid #d32f2f; padding-bottom: 10px; }
            h2 { color: #1976d2; margin-top: 30px; }
            .warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; margin: 20px 0; border-radius: 5px; }
            .important { background: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; margin: 20px 0; border-radius: 5px; }
            ul li { margin-bottom: 8px; }
            .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 0.9em; color: #666; }
        </style>
    </head>
    <body>
        <h1>🏥 TAROT CKD Risk Prediction - Clinical Disclaimer</h1>
        
        <div class="warning">
            <strong>⚠️ IMPORTANT:</strong> This tool is designed as a clinical decision support aid and should NOT replace clinical judgment. All predictions must be interpreted by qualified healthcare professionals.
        </div>

        <h2>🎯 Intended Use</h2>
        <ul>
            <li>Designed for healthcare professionals treating adult CKD patients (age ≥18 years)</li>
            <li>Applicable to patients with CKD Stage 3-5 (eGFR 10-60 mL/min/1.73m²)</li>
            <li>Intended for risk stratification and care planning discussions</li>
            <li>Should complement, not replace, established clinical guidelines (KDIGO, KDOQI)</li>
        </ul>

        <h2>⚕️ Clinical Decision Support</h2>
        <ul>
            <li>Risk predictions are estimates based on population data and may not apply to individual patients</li>
            <li>Clinical context, patient preferences, and additional factors not captured by the model should always be considered</li>
            <li>All predictions should be interpreted within the broader clinical picture</li>
            <li>Emergency situations require immediate medical attention regardless of model predictions</li>
        </ul>

        <h2>🔬 Model Limitations</h2>
        <ul>
            <li>Based on specific patient populations and may not generalize to all ethnicities or regions</li>
            <li>Requires accurate laboratory values and complete medical history for optimal performance</li>
            <li>Cannot account for rapid changes in clinical status or acute kidney injury</li>
            <li>UPCR-to-UACR conversion is an approximation and may introduce uncertainty</li>
            <li>Model performance may vary in patients with rare conditions or extreme laboratory values</li>
        </ul>

        <h2>📊 Validation and Performance</h2>
        <ul>
            <li>Model trained and validated on longitudinal CKD patient data from multiple centers</li>
            <li>External validation demonstrates superior performance vs. traditional KFRE calculators</li>
            <li>Confidence intervals provided to support clinical decision-making under uncertainty</li>
            <li>Performance metrics available through the API performance endpoint</li>
        </ul>

        <h2>🔒 Privacy and Data Security</h2>
        <ul>
            <li><strong>NO patient data is stored, logged, or transmitted to external systems</strong></li>
            <li>All processing occurs in temporary session memory and is immediately discarded</li>
            <li>Healthcare institutions should ensure compliance with local privacy regulations</li>
            <li>Session IDs are anonymized and do not contain patient identifiers</li>
        </ul>

        <h2>👨‍⚕️ Professional Responsibility</h2>
        <ul>
            <li>Healthcare providers remain fully responsible for all clinical decisions</li>
            <li>Risk predictions should be discussed with patients in appropriate clinical context</li>
            <li>Regular monitoring and reassessment are essential for CKD patients</li>
            <li>Users should maintain appropriate professional liability coverage</li>
        </ul>

        <div class="important">
            <h2>⚖️ Liability Disclaimer</h2>
            <p><strong>This tool is provided 'as is' without warranties of any kind. The developers and institutions are not liable for clinical decisions or patient outcomes. Users assume full responsibility for appropriate clinical use and interpretation.</strong></p>
        </div>

        <h2>📞 Support Information</h2>
        <ul>
            <li>Technical support: See documentation at /docs/</li>
            <li>Clinical questions: Consult qualified nephrologists</li>
            <li>Report technical issues through appropriate institutional channels</li>
        </ul>

        <div class="footer">
            <p><strong>Regulatory Note:</strong> This tool has not been evaluated by regulatory agencies (FDA, Health Canada, etc.) for diagnostic use.</p>
            <p><strong>Last Updated:</strong> 2024-01-01 | <strong>Version:</strong> 1.0.0</p>
        </div>
    </body>
    </html>
    """
    
    return html_content


@router.get("/clinical-benchmarks")
async def get_clinical_benchmarks() -> Dict[str, Any]:
    """
    Get KDIGO clinical benchmarks for CKD risk management
    
    Returns evidence-based thresholds for:
    - Nephrology referral decisions
    - Multidisciplinary care initiation
    - Kidney replacement therapy preparation
    """
    return {
        "title": "KDIGO Clinical Practice Guideline Benchmarks",
        "source": "KDIGO 2012 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease",
        "benchmarks": {
            "nephrology_referral": {
                "threshold": "5-year kidney failure risk ≥3-5%",
                "rationale": "Evidence suggests benefit of nephrology care at this risk level",
                "action": "Consider nephrology referral for specialized CKD management",
                "evidence_level": "Expert opinion based on observational studies"
            },
            "multidisciplinary_care": {
                "threshold": "2-year kidney failure risk >10%",
                "rationale": "High risk warrants comprehensive care planning",
                "action": "Initiate multidisciplinary care including education, preparation for KRT",
                "evidence_level": "Expert consensus"
            },
            "krt_preparation": {
                "threshold": "2-year kidney failure risk >40%",
                "rationale": "Very high risk requires immediate preparation for kidney replacement therapy",
                "action": "Begin vascular access planning, transplant evaluation, modality education",
                "evidence_level": "Clinical practice guideline recommendation"
            }
        },
        "risk_categories": {
            "low_risk": {
                "range": "2-year risk <5%",
                "management": "Standard CKD care, annual monitoring",
                "color_code": "green"
            },
            "moderate_risk": {
                "range": "2-year risk 5-10%",
                "management": "Enhanced monitoring, lifestyle counseling",
                "color_code": "yellow"
            },
            "high_risk": {
                "range": "2-year risk 10-40%",
                "management": "Multidisciplinary care, preparation planning",
                "color_code": "orange"
            },
            "very_high_risk": {
                "range": "2-year risk >40%",
                "management": "Urgent KRT preparation, access planning",
                "color_code": "red"
            }
        },
        "interpretation_notes": [
            "Thresholds are guidelines and should be interpreted with clinical context",
            "Individual patient factors may modify risk interpretation",
            "Regular reassessment is recommended as patient status changes",
            "Shared decision-making with patients is essential"
        ]
    }


@router.get("/units")
async def get_unit_conversions() -> Dict[str, Any]:
    """
    Get laboratory unit conversion information
    
    Returns conversion factors and examples for all supported
    laboratory parameters and their common units.
    """
    return {
        "title": "Laboratory Unit Conversions",
        "description": "Conversion factors between common laboratory units",
        "conversions": {
            "creatinine": {
                "expected_unit": "μmol/L",
                "alternative_unit": "mg/dL",
                "conversion_factor": 88.4,
                "conversion_direction": "mg/dL → μmol/L (multiply by 88.4)",
                "example": "1.5 mg/dL = 132.6 μmol/L",
                "notes": "CKD-EPI equation uses mg/dL internally"
            },
            "hemoglobin": {
                "expected_unit": "g/dL",
                "alternative_unit": "g/L",
                "conversion_factor": 0.1,
                "conversion_direction": "g/L → g/dL (divide by 10)",
                "example": "120 g/L = 12.0 g/dL",
                "notes": "Most common in North American labs as g/dL"
            },
            "phosphate": {
                "expected_unit": "mmol/L",
                "alternative_unit": "mg/dL",
                "conversion_factor": 0.3229,
                "conversion_direction": "mg/dL → mmol/L (multiply by 0.3229)",
                "example": "4.0 mg/dL = 1.29 mmol/L",
                "notes": "SI units preferred in most international labs"
            },
            "bicarbonate": {
                "expected_unit": "mmol/L",
                "alternative_unit": "mEq/L",
                "conversion_factor": 1.0,
                "conversion_direction": "mEq/L = mmol/L (same value)",
                "example": "24 mEq/L = 24 mmol/L",
                "notes": "Numerically identical for bicarbonate"
            },
            "uacr": {
                "expected_unit": "mg/mmol",
                "alternative_unit": "mg/g",
                "conversion_factor": 0.113,
                "conversion_direction": "mg/g → mg/mmol (multiply by 0.113)",
                "example": "30 mg/g = 3.4 mg/mmol",
                "notes": "UACR conversion from UPCR available"
            },
            "upcr": {
                "expected_unit": "mg/mmol",
                "alternative_unit": "mg/g",
                "conversion_factor": 0.113,
                "conversion_direction": "mg/g → mg/mmol (multiply by 0.113)",
                "example": "150 mg/g = 17 mg/mmol",
                "notes": "Can be converted to UACR using prediction model"
            }
        },
        "regional_preferences": {
            "north_america": ["mg/dL", "g/dL", "mg/g"],
            "europe_asia": ["μmol/L", "mmol/L", "g/L", "mg/mmol"],
            "uk": "Mixed usage depending on laboratory"
        },
        "validation_ranges": {
            "creatinine": "10-3000 μmol/L (0.11-33.9 mg/dL)",
            "hemoglobin": "3-25 g/dL (30-250 g/L)",
            "phosphate": "0.1-5.0 mmol/L (0.31-15.5 mg/dL)",
            "bicarbonate": "1-50 mmol/L (same as mEq/L)",
            "uacr": "0-10000 mg/mmol (0-1130 mg/g)",
            "upcr": "0-10000 mg/mmol (0-1130 mg/g)"
        }
    }