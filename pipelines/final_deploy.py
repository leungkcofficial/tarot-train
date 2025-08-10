"""
Multi-Model Deployment Pipeline for CKD Risk Prediction

This module implements ensemble predictions from multiple pre-trained models.
It loads 36 models (24 DeepSurv + 12 DeepHit) and generates ensemble predictions.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import gc
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from zenml.pipelines import pipeline
from zenml.steps import step
from zenml.client import Client

# Import necessary modules
from src.util import load_yaml_file, save_predictions_to_hdf5
from src.nn_architectures import create_network
from src.competing_risks_evaluation import save_competing_risks_predictions
from pycox.models import CoxPH, DeepHit
from steps.model_train import prepare_survival_dataset

# ==================== Helper Functions ====================

def load_all_model_configurations(config_csv_path: str, config_dir: str) -> List[Dict[str, Any]]:
    """
    Load model configurations from CSV and corresponding JSON files.
    
    Args:
        config_csv_path: Path to model_config.csv
        config_dir: Directory containing model JSON files
        
    Returns:
        List of dictionaries with model metadata and configurations
    """
    print(f"\n=== Loading Model Configurations ===")
    print(f"CSV path: {config_csv_path}")
    print(f"Config directory: {config_dir}")
    
    # Read CSV file
    model_df = pd.read_csv(config_csv_path)
    print(f"Found {len(model_df)} models in configuration CSV")
    
    model_configs = []
    
    for idx, row in model_df.iterrows():
        model_no = row['Model No.']
        
        # Load model details JSON
        details_pattern = f"model{model_no}_details_*.json"
        details_files = [f for f in os.listdir(config_dir) if f.startswith(f"model{model_no}_details_")]
        
        if not details_files:
            print(f"Warning: No details file found for model {model_no}")
            continue
            
        details_path = os.path.join(config_dir, details_files[0])
        
        # Load optimization metrics JSON (optional)
        metrics_files = [f for f in os.listdir(config_dir) if f.startswith(f"model{model_no}_optimization_metrics_")]
        metrics_path = os.path.join(config_dir, metrics_files[0]) if metrics_files else None
        
        try:
            with open(details_path, 'r') as f:
                model_details = json.load(f)
                
            optimization_metrics = None
            if metrics_path and os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    optimization_metrics = json.load(f)
            
            # Combine CSV metadata with JSON details
            model_config = {
                'model_no': model_no,
                'algorithm': row['Algorithm'],
                'structure': row['Structure'],
                'balancing_method': row['Balancing Method'],
                'prediction_endpoint': row['Prediction Endpoint'],
                'optimization_target': row['Optimization target'],
                'model_details': model_details,
                'optimization_metrics': optimization_metrics,
                'details_path': details_path,
                'metrics_path': metrics_path
            }
            
            model_configs.append(model_config)
            print(f"Loaded configuration for Model {model_no}: {row['Algorithm']} - {row['Structure']} - {row['Prediction Endpoint']}")
            
        except Exception as e:
            print(f"Error loading configuration for model {model_no}: {e}")
            continue
    
    print(f"\nSuccessfully loaded {len(model_configs)} model configurations")
    return model_configs


def create_model_from_config(model_config: Dict[str, Any], device: str = 'cpu') -> Any:
    """
    Create and load a model based on configuration.
    
    Args:
        model_config: Model configuration dictionary
        device: Device to load model on ('cpu' or 'cuda')
        
    Returns:
        Loaded model instance
    """
    model_details = model_config['model_details']
    model_type = model_details['model_type']
    
    # Extract network parameters
    network_type = 'lstm' if model_config['structure'].upper() == 'LSTM' else 'ann'
    input_dim = model_details['input_dim']
    output_dim = model_details.get('output_dim', 1)
    hidden_dims = model_details.get('hidden_dims', [128, 64, 32])
    dropout = model_details.get('dropout', 0.2)
    
    # LSTM specific parameters
    lstm_num_layers = model_details.get('lstm_num_layers', 2)
    lstm_bidirectional = model_details.get('bidirectional', True)
    sequence_length = model_details.get('sequence_length', 5)
    
    # Create network
    if network_type == 'lstm':
        net = create_network(
            model_type=model_type,
            network_type=network_type,
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
            lstm_num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
            sequence_length=sequence_length,
            lstm_hidden_dim=hidden_dims[0] if hidden_dims else 64
        )
    else:
        net = create_network(
            model_type=model_type,
            network_type=network_type,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )
    
    # Load model weights
    model_path = model_details['model_path']
    if not os.path.exists(model_path):
        # Try to find the model in the models directory
        model_filename = os.path.basename(model_path)
        model_path = os.path.join('models', model_filename)
        
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        net.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    # Create optimizer (required for model creation)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # Create model instance
    if model_type.lower() == "deepsurv":
        model = CoxPH(net, optimizer=optimizer)
    else:  # deephit
        alpha = model_details.get('alpha', 0.2)
        sigma = model_details.get('sigma', 0.1)
        time_grid = np.array(model_details.get('time_grid', [365, 730, 1095, 1460, 1825]))
        model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
    
    # Move model to device
    model.net = model.net.to(device)
    model.net.eval()
    
    return model


def generate_model_predictions(
    model: Any,
    test_data: pd.DataFrame,
    feature_cols: List[str],
    model_type: str,
    endpoint: Optional[int] = None,
    batch_size: int = 1000,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions for a single model.
    
    Args:
        model: Trained model instance
        test_data: Test dataframe
        feature_cols: List of feature columns
        model_type: 'deepsurv' or 'deephit'
        endpoint: Target endpoint for DeepSurv (1 or 2)
        batch_size: Batch size for processing
        device: Device for computation
        
    Returns:
        Tuple of (predictions, durations, events)
        - DeepSurv: predictions shape (1825, n_samples)
        - DeepHit: predictions shape (2, 5, n_samples)
    """
    # Prepare dataset
    if model_type.lower() == 'deepsurv':
        dataset = prepare_survival_dataset(
            test_data,
            feature_cols=feature_cols,
            target_endpoint=endpoint
        )
    else:
        dataset = prepare_survival_dataset(
            test_data,
            feature_cols=feature_cols,
            target_endpoint=None  # DeepHit handles all endpoints
        )
    
    X_test, durations, events = dataset
    n_samples = len(X_test)
    
    # Process in batches
    all_predictions = []
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_X = X_test[i:batch_end]
        
        # Convert to tensor
        batch_tensor = torch.tensor(batch_X, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if isinstance(model, DeepHit):
                # DeepHit returns CIF predictions
                batch_preds = model.predict_cif(batch_tensor)
            else:
                # DeepSurv returns survival probabilities
                batch_preds = model.predict_surv_df(batch_tensor)
                
        all_predictions.append(batch_preds)
        
    # Combine predictions
    if isinstance(model, DeepHit):
        # For DeepHit: concatenate along samples axis
        predictions = np.concatenate(all_predictions, axis=2)
    else:
        # For DeepSurv: concatenate DataFrames
        predictions = pd.concat(all_predictions, axis=1)
        predictions = predictions.values  # Convert to numpy array
        
    return predictions, durations, events


def convert_survival_to_cif(survival_probs: np.ndarray) -> np.ndarray:
    """
    Convert survival probabilities to cumulative incidence function.
    
    Args:
        survival_probs: Array of survival probabilities
        
    Returns:
        Array of CIF values (1 - survival probability)
    """
    # Ensure valid range [0, 1]
    survival_probs = np.clip(survival_probs, 0, 1)
    
    # Convert to CIF
    cif = 1 - survival_probs
    
    return cif


def extract_time_points(
    predictions: np.ndarray,
    time_grid: np.ndarray = None,
    target_times: List[int] = [365, 730, 1095, 1460, 1825]
) -> np.ndarray:
    """
    Extract predictions at specific time points.
    
    Args:
        predictions: Predictions array
        time_grid: Time grid for predictions (for DeepSurv)
        target_times: Target time points to extract
        
    Returns:
        Array of shape (len(target_times), n_samples)
    """
    if time_grid is None:
        # For DeepSurv, assume time grid is 1 to 1825 days
        time_grid = np.arange(1, 1826)
    
    # Find indices of target times
    indices = []
    for t in target_times:
        idx = np.argmin(np.abs(time_grid - t))
        indices.append(idx)
    
    # Extract predictions at these indices
    extracted = predictions[indices, :]
    
    return extracted


def stack_deepsurv_predictions(
    predictions_dict: Dict[int, Dict[str, Any]],
    model_configs: List[Dict[str, Any]]
) -> List[np.ndarray]:
    """
    Stack DeepSurv predictions by grouping criteria.
    
    Groups by: Algorithm, Structure, Balancing Method, Optimization Target
    Stacks Event 1 and Event 2 predictions
    
    Args:
        predictions_dict: Dictionary of predictions by model number
        model_configs: List of model configurations
        
    Returns:
        List of stacked predictions, each of shape (2, 5, n_samples)
    """
    # Create groups
    groups = {}
    
    for config in model_configs:
        if config['algorithm'] != 'DeepSurv':
            continue
            
        # Create group key
        group_key = (
            config['algorithm'],
            config['structure'],
            config['balancing_method'],
            config['optimization_target']
        )
        
        if group_key not in groups:
            groups[group_key] = {'event1': None, 'event2': None}
        
        # Assign predictions to group
        model_no = config['model_no']
        if model_no in predictions_dict:
            if config['prediction_endpoint'] == 'Event 1':
                groups[group_key]['event1'] = predictions_dict[model_no]['predictions']
            else:  # Event 2
                groups[group_key]['event2'] = predictions_dict[model_no]['predictions']
    
    # Stack predictions for each group
    stacked_predictions = []
    
    for group_key, group_preds in groups.items():
        if group_preds['event1'] is not None and group_preds['event2'] is not None:
            # Stack Event 1 and Event 2
            stacked = np.stack([group_preds['event1'], group_preds['event2']], axis=0)
            stacked_predictions.append(stacked)
            print(f"Stacked group {group_key}: shape {stacked.shape}")
        else:
            print(f"Warning: Incomplete group {group_key}")
    
    return stacked_predictions


def ensemble_predictions(
    all_predictions: np.ndarray,
    method: str = 'average',
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply ensemble method to predictions.
    
    Args:
        all_predictions: Array of shape (n_models, 2, 5, n_samples)
        method: Ensemble method ('average', 'weighted', 'voting')
        weights: Weights for weighted averaging
        
    Returns:
        Ensemble predictions of shape (2, 5, n_samples)
    """
    if method == 'average':
        # Simple averaging
        ensemble = np.mean(all_predictions, axis=0)
    elif method == 'weighted' and weights is not None:
        # Weighted averaging
        weights = weights.reshape(-1, 1, 1, 1)  # Reshape for broadcasting
        weighted_preds = all_predictions * weights
        ensemble = np.sum(weighted_preds, axis=0) / np.sum(weights)
    else:
        raise ValueError(f"Unsupported ensemble method: {method}")
    
    # Ensure valid range [0, 1] for CIF
    ensemble = np.clip(ensemble, 0, 1)
    
    return ensemble


# ==================== Main Pipeline ====================

@pipeline(enable_cache=False)
def multi_model_deploy_pipeline():
    """
    Pipeline for deploying multiple models and generating ensemble predictions.
    """
    # Import steps
    from steps.ingest_data import ingest_data
    from steps.clean_data import clean_data
    from steps.merge_data import merge_data
    from steps.split_data import split_data
    from steps.impute_data import impute_data
    from steps.preprocess_data import preprocess_data
    
    # Data preparation steps (same as training pipeline)
    cr_df, hb_df, a1c_df, alb_df, po4_df, ca_df, ca_adjusted_df, hco3_df, upcr_df, uacr_df, demo_df, icd10_df, death_df, operation_df = ingest_data()
    
    patient_df, icd10_df_clean, cr_df_clean, hb_df_clean, a1c_df_clean, alb_df_clean, po4_df_clean, ca_df_clean, ca_adjusted_df_clean, hco3_df_clean, upcr_df_clean, uacr_df_clean, operation_df_clean, death_df_clean, cci_df, cci_score_df, hypertension_df, egfr_df = clean_data(
        cr_df=cr_df,
        hb_df=hb_df,
        a1c_df=a1c_df,
        alb_df=alb_df,
        po4_df=po4_df,
        ca_df=ca_df,
        ca_adjusted_df=ca_adjusted_df,
        hco3_df=hco3_df,
        upcr_df=upcr_df,
        uacr_df=uacr_df,
        icd10_df=icd10_df,
        operation_df=operation_df,
        death_df=death_df,
        demo_df=demo_df
    )
    
    final_df, prediction_df = merge_data(
        patient_df=patient_df,
        icd10_df=icd10_df_clean,
        cr_df=cr_df_clean,
        hb_df=hb_df_clean,
        a1c_df=a1c_df_clean,
        alb_df=alb_df_clean,
        po4_df=po4_df_clean,
        ca_df=ca_df_clean,
        ca_adjusted_df=ca_adjusted_df_clean,
        hco3_df=hco3_df_clean,
        upcr_df=upcr_df_clean,
        uacr_df=uacr_df_clean,
        operation_df=operation_df_clean,
        death_df=death_df_clean,
        cci_df=cci_df,
        cci_score_df=cci_score_df,
        hypertension_df=hypertension_df,
        egfr_df=egfr_df
    )
    
    train_df, temporal_test_df, spatial_test_df, raw_df = split_data(
        raw_df=final_df,
        prediction_df=prediction_df
    )
    
    train_df_imputed, temporal_test_df_imputed, spatial_test_df_imputed = impute_data(
        train_df=train_df,
        temporal_test_df=temporal_test_df,
        spatial_test_df=spatial_test_df
    )
    
    train_df_preprocessed, temporal_test_df_preprocessed, spatial_test_df_preprocessed = preprocess_data(
        train_df=train_df_imputed,
        temporal_test_df=temporal_test_df_imputed,
        spatial_test_df=spatial_test_df_imputed
    )
    
    # Multi-model deployment step
    ensemble_results = deploy_multiple_models(
        train_df_preprocessed=train_df_preprocessed,
        temporal_test_df_preprocessed=temporal_test_df_preprocessed,
        spatial_test_df_preprocessed=spatial_test_df_preprocessed
    )
    
    return ensemble_results


@step(enable_cache=False)
def deploy_multiple_models(
    train_df_preprocessed: pd.DataFrame,
    temporal_test_df_preprocessed: pd.DataFrame,
    spatial_test_df_preprocessed: pd.DataFrame,
    config_csv_path: str = "results/final_deploy/model_config/model_config.csv",
    config_dir: str = "results/final_deploy/model_config",
    master_df_mapping_path: str = "src/default_master_df_mapping.yml",
    ensemble_method: str = 'average',
    batch_size: int = 1000
) -> Dict[str, Any]:
    """
    Deploy multiple models and generate ensemble predictions.
    
    Args:
        train_df_preprocessed: Preprocessed training data (for baseline hazards)
        temporal_test_df_preprocessed: Preprocessed temporal test data
        spatial_test_df_preprocessed: Preprocessed spatial test data
        config_csv_path: Path to model configuration CSV
        config_dir: Directory containing model JSON files
        master_df_mapping_path: Path to master dataframe mapping
        ensemble_method: Method for ensemble ('average', 'weighted')
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with deployment results
    """
    print("\n=== Multi-Model Deployment Pipeline ===\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load master dataframe mapping
    master_df_mapping = load_yaml_file(master_df_mapping_path)
    feature_cols = master_df_mapping.get("features", [])
    print(f"Loaded {len(feature_cols)} features from master dataframe mapping")
    
    # Load all model configurations
    model_configs = load_all_model_configurations(config_csv_path, config_dir)
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    individual_dir = "results/final_deploy/individual_predictions"
    ensemble_dir = "results/final_deploy/ensemble_predictions"
    os.makedirs(individual_dir, exist_ok=True)
    os.makedirs(ensemble_dir, exist_ok=True)
    
    # Process each model
    temporal_predictions = {}
    spatial_predictions = {}
    
    for config in model_configs:
        model_no = config['model_no']
        print(f"\n=== Processing Model {model_no} ===")
        print(f"Type: {config['algorithm']} - {config['structure']}")
        print(f"Endpoint: {config['prediction_endpoint']}")
        print(f"Balancing: {config['balancing_method']}")
        
        try:
            # Create and load model
            model = create_model_from_config(config, device)
            
            # For DeepSurv, compute baseline hazards
            if config['algorithm'] == 'DeepSurv':
                print("Computing baseline hazards for DeepSurv model...")
                endpoint = 1 if config['prediction_endpoint'] == 'Event 1' else 2
                train_data = prepare_survival_dataset(
                    train_df_preprocessed,
                    feature_cols=feature_cols,
                    target_endpoint=endpoint
                )
                x_train = torch.tensor(train_data[0], dtype=torch.float32).to(device)
                durations_train = torch.tensor(train_data[1], dtype=torch.float32)
                events_train = torch.tensor(train_data[2], dtype=torch.float32)
                model.compute_baseline_hazards(input=x_train, target=(durations_train, events_train))
            
            # Generate predictions for temporal test set
            print("Generating temporal test predictions...")
            endpoint = None
            if config['algorithm'] == 'DeepSurv':
                endpoint = 1 if config['prediction_endpoint'] == 'Event 1' else 2
                
            temp_preds, temp_durations, temp_events = generate_model_predictions(
                model=model,
                test_data=temporal_test_df_preprocessed,
                feature_cols=feature_cols,
                model_type=config['model_details']['model_type'],
                endpoint=endpoint,
                batch_size=batch_size,
                device=device
            )
            
            # Process predictions based on model type
            if config['algorithm'] == 'DeepSurv':
                # Convert to CIF and extract time points
                temp_cif = convert_survival_to_cif(temp_preds)
                temp_extracted = extract_time_points(temp_cif)
                temporal_predictions[model_no] = {
                    'predictions': temp_extracted,
                    'durations': temp_durations,
                    'events': temp_events,
                    'config': config
                }
            else:  # DeepHit
                temporal_predictions[model_no] = {
                    'predictions': temp_preds,
                    'durations': temp_durations,
                    'events': temp_events,
                    'config': config
                }
            
            # Generate predictions for spatial test set
            print("Generating spatial test predictions...")
            spat_preds, spat_durations, spat_events = generate_model_predictions(
                model=model,
                test_data=spatial_test_df_preprocessed,
                feature_cols=feature_cols,
                model_type=config['model_details']['model_type'],
                endpoint=endpoint,
                batch_size=batch_size,
                device=device
            )
            
            # Process predictions based on model type
            if config['algorithm'] == 'DeepSurv':
                # Convert to CIF and extract time points
                spat_cif = convert_survival_to_cif(spat_preds)
                spat_extracted = extract_time_points(spat_cif)
                spatial_predictions[model_no] = {
                    'predictions': spat_extracted,
                    'durations': spat_durations,
                    'events': spat_events,
                    'config': config
                }
            else:  # DeepHit
                spatial_predictions[model_no] = {
                    'predictions': spat_preds,
                    'durations': spat_durations,
                    'events': spat_events,
                    'config': config
                }
            
            # Save individual predictions
            print("Saving individual predictions...")
            
            # Temporal predictions
            temp_pred_path = os.path.join(individual_dir, f"model{model_no}_temporal_predictions_{timestamp}.h5")
            temp_meta_path = os.path.join(individual_dir, f"model{model_no}_temporal_metadata_{timestamp}.csv")
            
            if config['algorithm'] == 'DeepHit':
                save_competing_risks_predictions(
                    cif_predictions=temp_preds,
                    time_grid=np.array([365, 730, 1095, 1460, 1825]),
                    durations=temp_durations,
                    events=temp_events,
                    save_path=temp_pred_path,
                    metadata_path=temp_meta_path
                )
            else:
                # For DeepSurv, save the extracted predictions
                save_predictions_to_hdf5(
                    temp_extracted,
                    temp_pred_path,
                    metadata={'durations': temp_durations, 'events': temp_events}
                )
            
            # Spatial predictions
            spat_pred_path = os.path.join(individual_dir, f"model{model_no}_spatial_predictions_{timestamp}.h5")
            spat_meta_path = os.path.join(individual_dir, f"model{model_no}_spatial_metadata_{timestamp}.csv")
            
            if config['algorithm'] == 'DeepHit':
                save_competing_risks_predictions(
                    cif_predictions=spat_preds,
                    time_grid=np.array([365, 730, 1095, 1460, 1825]),
                    durations=spat_durations,
                    events=spat_events,
                    save_path=spat_pred_path,
                    metadata_path=spat_meta_path
                )
            else:
                # For DeepSurv, save the extracted predictions
                save_predictions_to_hdf5(
                    spat_extracted,
                    spat_pred_path,
                    metadata={'durations': spat_durations, 'events': spat_events}
                )
            
            print(f"Model {model_no} processed successfully")
            
            # Clear memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        except Exception as e:
            print(f"Error processing model {model_no}: {e}")
            continue
    
    # Stack predictions
    print("\n=== Stacking Predictions ===")
    
    # Stack DeepSurv predictions
    deepsurv_configs = [c for c in model_configs if c['algorithm'] == 'DeepSurv']
    
    # Temporal stacking
    temp_deepsurv_stacked = stack_deepsurv_predictions(temporal_predictions, deepsurv_configs)
    print(f"Stacked {len(temp_deepsurv_stacked)} DeepSurv groups for temporal test")
    
    # Spatial stacking
    spat_deepsurv_stacked = stack_deepsurv_predictions(spatial_predictions, deepsurv_configs)
    print(f"Stacked {len(spat_deepsurv_stacked)} DeepSurv groups for spatial test")
    
    # Collect DeepHit predictions
    temp_deephit_preds = []
    spat_deephit_preds = []
    
    for model_no, pred_data in temporal_predictions.items():
        if pred_data['config']['algorithm'] == 'DeepHit':
            temp_deephit_preds.append(pred_data['predictions'])
            
    for model_no, pred_data in spatial_predictions.items():
        if pred_data['config']['algorithm'] == 'DeepHit':
            spat_deephit_preds.append(pred_data['predictions'])
    
    print(f"Collected {len(temp_deephit_preds)} DeepHit predictions")
    
    # Combine all predictions
    print("\n=== Creating Ensemble ===")
    
    # Temporal ensemble
    all_temp_preds = temp_deepsurv_stacked + temp_deephit_preds
    if all_temp_preds:
        all_temp_array = np.array(all_temp_preds)
        print(f"Combined temporal predictions shape: {all_temp_array.shape}")
        
        # Apply ensemble
        temp_ensemble = ensemble_predictions(all_temp_array, method=ensemble_method)
        print(f"Temporal ensemble shape: {temp_ensemble.shape}")
        
        # Save ensemble predictions
        ensemble_temp_path = os.path.join(ensemble_dir, f"ensemble_temporal_predictions_{timestamp}.h5")
        ensemble_temp_meta_path = os.path.join(ensemble_dir, f"ensemble_temporal_metadata_{timestamp}.csv")
        
        # Use the first model's durations and events (they should all be the same)
        first_model_no = list(temporal_predictions.keys())[0]
        temp_durations = temporal_predictions[first_model_no]['durations']
        temp_events = temporal_predictions[first_model_no]['events']
        
        save_competing_risks_predictions(
            cif_predictions=temp_ensemble,
            time_grid=np.array([365, 730, 1095, 1460, 1825]),
            durations=temp_durations,
            events=temp_events,
            save_path=ensemble_temp_path,
            metadata_path=ensemble_temp_meta_path
        )
    
    # Spatial ensemble
    all_spat_preds = spat_deepsurv_stacked + spat_deephit_preds
    if all_spat_preds:
        all_spat_array = np.array(all_spat_preds)
        print(f"Combined spatial predictions shape: {all_spat_array.shape}")
        
        # Apply ensemble
        spat_ensemble = ensemble_predictions(all_spat_array, method=ensemble_method)
        print(f"Spatial ensemble shape: {spat_ensemble.shape}")
        
        # Save ensemble predictions
        ensemble_spat_path = os.path.join(ensemble_dir, f"ensemble_spatial_predictions_{timestamp}.h5")
        ensemble_spat_meta_path = os.path.join(ensemble_dir, f"ensemble_spatial_metadata_{timestamp}.csv")
        
        # Use the first model's durations and events
        first_model_no = list(spatial_predictions.keys())[0]
        spat_durations = spatial_predictions[first_model_no]['durations']
        spat_events = spatial_predictions[first_model_no]['events']
        
        save_competing_risks_predictions(
            cif_predictions=spat_ensemble,
            time_grid=np.array([365, 730, 1095, 1460, 1825]),
            durations=spat_durations,
            events=spat_events,
            save_path=ensemble_spat_path,
            metadata_path=ensemble_spat_meta_path
        )
    
    # Save deployment log
    deployment_log = {
        'timestamp': timestamp,
        'n_models_processed': len(temporal_predictions),
        'n_deepsurv_groups': len(temp_deepsurv_stacked),
        'n_deephit_models': len(temp_deephit_preds),
        'ensemble_method': ensemble_method,
        'model_configs': [
            {
                'model_no': c['model_no'],
                'algorithm': c['algorithm'],
                'structure': c['structure'],
                'endpoint': c['prediction_endpoint'],
                'balancing': c['balancing_method']
            }
            for c in model_configs
        ]
    }
    
    log_path = os.path.join(ensemble_dir, f"deployment_log_{timestamp}.json")
    with open(log_path, 'w') as f:
        json.dump(deployment_log, f, indent=2)
    
    print(f"\n=== Deployment Complete ===")
    print(f"Processed {len(temporal_predictions)} models")
    print(f"Individual predictions saved to: {individual_dir}")
    print(f"Ensemble predictions saved to: {ensemble_dir}")
    print(f"Deployment log saved to: {log_path}")
    
    return {
        'n_models': len(temporal_predictions),
        'ensemble_temporal_path': ensemble_temp_path if 'ensemble_temp_path' in locals() else None,
        'ensemble_spatial_path': ensemble_spat_path if 'ensemble_spat_path' in locals() else None,
        'deployment_log_path': log_path,
        'timestamp': timestamp
    }


if __name__ == "__main__":
    # Run the pipeline
    pipeline = multi_model_deploy_pipeline()
    pipeline.run()