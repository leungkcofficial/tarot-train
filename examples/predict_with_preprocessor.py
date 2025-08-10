"""
Example: Making Predictions with CKD Models

This example shows how to use the saved preprocessor and models
to make predictions for new patients.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path

from src.ckd_preprocessor import CKDPreprocessor
from src.nn_architectures import create_network
from pycox.models import DeepSurv, DeepHit


def load_model_and_baseline_hazards(model_path: str, model_config: dict, device='cpu'):
    """Load a trained model and its baseline hazards."""
    
    # Determine model type and architecture
    model_type = model_config.get('model_type', 'DeepSurv')
    architecture = model_config.get('architecture', 'ANN')
    
    # Load model
    if model_type.lower() == 'deepsurv':
        # Create network
        if architecture.upper() == 'LSTM':
            # For LSTM, we need to detect architecture from state dict
            state_dict = torch.load(model_path, map_location=device)
            # Infer dimensions from state dict
            input_size = state_dict['net.lstm.weight_ih_l0'].shape[1]
            hidden_size = state_dict['net.lstm.weight_ih_l0'].shape[0] // 4
            num_layers = len([k for k in state_dict.keys() if 'lstm.weight_ih' in k])
            
            net = create_network(
                architecture='LSTM',
                in_features=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                out_features=1,
                dropout=0.1
            )
        else:
            # For ANN
            net = create_network(
                architecture='ANN',
                in_features=model_config.get('in_features', 100),
                num_nodes=model_config.get('num_nodes', [256, 256]),
                out_features=1,
                dropout=model_config.get('dropout', 0.1),
                activation=model_config.get('activation', 'relu')
            )
        
        # Create DeepSurv model
        model = DeepSurv(net)
        model.net.load_state_dict(torch.load(model_path, map_location=device))
        
        # Load baseline hazards
        model_num = model_config.get('model_no', 1)
        baseline_pattern = f'baseline_hazards_model{model_num}_*.pkl'
        baseline_files = list(Path('results/final_deploy/models').glob(baseline_pattern))
        
        if baseline_files:
            with open(baseline_files[0], 'rb') as f:
                baseline_data = pickle.load(f)
                model.baseline_hazards_ = baseline_data['baseline_hazards_']
                model.baseline_cumhazards_ = baseline_data['baseline_cumhazards_']
        
    elif model_type.lower() == 'deephit':
        # Similar loading for DeepHit
        # ... (implementation depends on your DeepHit setup)
        pass
    
    model.net.eval()
    return model


def predict_single_patient(patient_data: dict, preprocessor_path: str, model_paths: dict):
    """
    Make predictions for a single patient using all models.
    
    Args:
        patient_data: Dictionary with patient clinical variables
        preprocessor_path: Path to the saved preprocessor
        model_paths: Dictionary mapping model names to their paths
    
    Returns:
        Dictionary with predictions from all models
    """
    
    # Load preprocessor
    print("Loading preprocessor...")
    preprocessor = CKDPreprocessor.load(preprocessor_path)
    
    # Preprocess patient data
    print("Preprocessing patient data...")
    preprocessed = preprocessor.transform(patient_data)
    
    # Convert to numpy array for model input
    # Note: You may need to select specific columns based on your model's expected features
    X = preprocessed.values
    
    # Make predictions with each model
    predictions = {}
    
    for model_name, model_info in model_paths.items():
        print(f"\nMaking prediction with {model_name}...")
        
        try:
            # Load model
            model = load_model_and_baseline_hazards(
                model_info['path'],
                model_info['config']
            )
            
            # Make prediction
            # For survival models, we typically want survival probabilities at specific time points
            time_points = np.array([365, 730, 1095, 1825, 2555])  # 1, 2, 3, 5, 7 years
            
            if hasattr(model, 'predict_surv'):
                # DeepSurv prediction
                surv_probs = model.predict_surv(X, time_points)
                predictions[model_name] = {
                    'survival_probabilities': surv_probs,
                    'time_points_days': time_points,
                    'time_points_years': time_points / 365.25
                }
            
            print(f"✓ {model_name} prediction complete")
            
        except Exception as e:
            print(f"✗ Error with {model_name}: {e}")
            predictions[model_name] = {'error': str(e)}
    
    return predictions


def main():
    """Example usage of the preprocessing and prediction pipeline."""
    
    # Example patient data (11 core variables + additional info)
    patient = {
        # Demographics
        'key': 'PATIENT001',
        'date': '2024-01-15',
        'dob': '1955-03-20',
        'gender': 1,  # 1=Male, 0=Female
        'first_sub_60_date': '2020-06-01',  # Date when eGFR first < 60
        
        # Laboratory values (in correct units for your data)
        'creatinine': 186,      # µmol/L (NOT mg/dL!)
        'hemoglobin': 11.5,     # g/dL
        'albumin': 36,          # g/L (NOT g/dL!)
        'a1c': 7.2,            # %
        'phosphate': 1.3,       # mmol/L (NOT mg/dL!)
        'calcium': 2.3,         # mmol/L (NOT mg/dL!)
        'ca_adjusted': None,    # mmol/L (Will be imputed)
        'bicarbonate': 22,      # mmol/L (or mEq/L - same value)
        'upcr': 1.8,           # g/g
        'uacr': None,          # mg/g (Will be imputed)
        'egfr': 35,            # mL/min/1.73m²
        
        # Comorbidities
        'ht': 1,               # Hypertension
        'dm': 1,               # Diabetes
        
        # CCI components (Charlson Comorbidity Index)
        'myocardial_infarction': 0,
        'congestive_heart_failure': 1,
        'peripheral_vascular_disease': 0,
        'cerebrovascular_disease': 0,
        'dementia': 0,
        'chronic_pulmonary_disease': 0,
        'rheumatic_disease': 0,
        'peptic_ulcer_disease': 0,
        'mild_liver_disease': 0,
        'diabetes_wo_complication': 0,
        'renal_mild_moderate': 1,
        'diabetes_w_complication': 1,
        'hemiplegia_paraplegia': 0,
        'any_malignancy': 0,
        'liver_severe': 0,
        'renal_severe': 0,
        'hiv': 0,
        'metastatic_cancer': 0,
        'aids': 0,
        'cci_score_total': 3
    }
    
    # Paths to preprocessor and models
    preprocessor_path = 'results/final_deploy/ckd_preprocessor.pkl'
    
    # Example model paths (you would load these from your model registry)
    model_paths = {
        'DeepSurv_ANN_Event1_Model1': {
            'path': 'results/final_deploy/models/Ensemble_model1_DeepSurv_ANN_Event_1_20250804_111238.pt',
            'config': {
                'model_type': 'DeepSurv',
                'architecture': 'ANN',
                'model_no': 1,
                'in_features': 100,  # Adjust based on your actual model
                'num_nodes': [256, 256],
                'dropout': 0.1
            }
        },
        # Add more models as needed
    }
    
    # Make predictions
    print("="*60)
    print("CKD Risk Prediction Example")
    print("="*60)
    print(f"\nPatient Information:")
    print(f"  Age: {(pd.to_datetime('2024-01-15') - pd.to_datetime(patient['dob'])).days / 365.25:.1f} years")
    print(f"  Gender: {'Male' if patient['gender'] == 1 else 'Female'}")
    print(f"  eGFR: {patient['egfr']} mL/min/1.73m²")
    print(f"  Creatinine: {patient['creatinine']} mg/dL")
    print(f"  Comorbidities: Hypertension={'Yes' if patient['ht'] else 'No'}, Diabetes={'Yes' if patient['dm'] else 'No'}")
    
    # Check if files exist
    if not os.path.exists(preprocessor_path):
        print(f"\n✗ Preprocessor not found at {preprocessor_path}")
        print("  Please run: python scripts/extract_preprocessing_pipeline.py")
        return
    
    # Make predictions
    try:
        predictions = predict_single_patient(patient, preprocessor_path, model_paths)
        
        # Display results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        
        for model_name, pred in predictions.items():
            if 'error' in pred:
                print(f"\n{model_name}: Error - {pred['error']}")
            else:
                print(f"\n{model_name}:")
                print("  Survival Probabilities:")
                for t, p in zip(pred['time_points_years'], pred['survival_probabilities'][0]):
                    print(f"    {t:.0f} years: {p:.1%}")
        
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()