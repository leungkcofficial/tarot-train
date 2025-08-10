"""
Multi-Model Deployment Pipeline for CKD Risk Prediction (Version 2 - Fixed)

This module implements ensemble predictions from multiple pre-trained models.
Fixed issues:
- Handle NaN balancing methods properly
- Fix LSTM architecture loading
- Handle GPU/CPU tensor conversion for DeepHit
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
from src.balance_data import balance_dataframe
from pycox.models import CoxPH, DeepHit
from pycox.evaluation import EvalSurv
from torchtuples.callbacks import EarlyStopping
from steps.model_train import prepare_survival_dataset
from steps.cv_utils import time_based_patient_cv, create_cv_datasets

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
        model_no = row['Model No']
        algorithm = row['Algorithm']
        structure = row['Structure']
        endpoint = row['Prediction Endpoint']
        balancing = row['Balancing Method']
        optimization = row['Optimization target']
        
        # Find corresponding JSON files
        details_json_files = [f for f in os.listdir(config_dir) if f.startswith(f"model{model_no}_details_") and f.endswith('.json')]
        optim_json_files = [f for f in os.listdir(config_dir) if f.startswith(f"model{model_no}_optimization_metrics_") and f.endswith('.json')]
        
        if not details_json_files:
            print(f"Warning: No details JSON file found for model {model_no}")
            continue
            
        details_json_file = details_json_files[0]
        details_json_path = os.path.join(config_dir, details_json_file)
        
        # Load JSON details
        with open(details_json_path, 'r') as f:
            model_details = json.load(f)
        
        # Load optimization metrics if available (for LSTM sequence parameter)
        optimization_details = {}
        if optim_json_files:
            optim_json_path = os.path.join(config_dir, optim_json_files[0])
            with open(optim_json_path, 'r') as f:
                optimization_details = json.load(f)
        
        # Create combined configuration
        config = {
            'model_no': model_no,
            'algorithm': algorithm,
            'structure': structure,
            'prediction_endpoint': endpoint,
            'balancing_method': balancing,
            'optimization_target': optimization,
            'model_details': model_details,
            'optimization_details': optimization_details,
            'json_file': details_json_file
        }
        
        model_configs.append(config)
        print(f"Loaded config for Model {model_no}: {algorithm} - {structure}")
    
    print(f"\nSuccessfully loaded {len(model_configs)} model configurations")
    return model_configs


def create_model_specific_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a model-specific configuration dictionary that mimics hyperparameter_config.yml structure.
    
    Args:
        model_config: Model configuration from CSV and JSON
        
    Returns:
        Configuration dictionary with model-specific settings
    """
    # Extract information from model config
    algorithm = model_config['algorithm'].lower()
    structure = model_config['structure'].lower()
    balancing_method = model_config['balancing_method']
    optimization_target = model_config['optimization_target']
    model_details = model_config['model_details']
    
    # Handle NaN balancing method - check if it's NaN or string 'nan'
    if pd.isna(balancing_method) or str(balancing_method).lower() == 'nan':
        balancing_method = 'None'
    
    # Parse balancing configuration
    balance_config = {
        'enable': False,
        'method': 'random_under_sampler',
        'sampling_strategy': 'majority'
    }
    
    if balancing_method != 'None':
        balance_config['enable'] = True
        
        if 'NearMiss' in str(balancing_method):
            balance_config['method'] = 'near_miss'
            # Extract version number from "NearMiss version X"
            if 'version' in str(balancing_method):
                version = int(balancing_method.split()[-1])
                balance_config['near_miss_version'] = version
        elif balancing_method == 'KNN':
            balance_config['method'] = 'knn'
        else:
            balance_config['method'] = 'random_under_sampler'
    
    # Get sequence length from optimization details for LSTM
    sequence_length = 5  # default
    if structure.lower() == 'lstm' and 'optimization_details' in model_config:
        opt_details = model_config['optimization_details']
        if 'best_params' in opt_details and 'sequence' in opt_details['best_params']:
            sequence_length = opt_details['best_params']['sequence']
            model_details['sequence_length'] = sequence_length
    
    # Create configuration structure
    config = {
        'model_type': algorithm,
        'target_endpoint': None if algorithm == 'deephit' else (1 if model_config['prediction_endpoint'] == 'Event 1' else 2),
        'network': {
            'type': structure,
            'default': {
                'hidden_dims': model_details.get('hidden_dims', [128, 64, 32]),
                'num_layers': len(model_details.get('hidden_dims', [128, 64, 32])),
                'dropout': model_details.get('dropout', 0.2),
                'batch_size': model_details.get('batch_size', 64),
                'learning_rate': model_details.get('learning_rate', 0.001),
                'epochs': model_details.get('epochs', 100)
            },
            'lstm': {
                'hidden_dims': model_details.get('lstm_hidden_dims', [128, 64, 32]),
                'num_layers': model_details.get('lstm_num_layers', 2),
                'bidirectional': model_details.get('bidirectional', True),
                'sequence_length': sequence_length
            },
            'deephit': {
                'alpha': model_details.get('alpha', 0.2),
                'sigma': model_details.get('sigma', 0.1),
                'time_grid': model_details.get('time_grid', [365, 730, 1095, 1460, 1825])
            }
        },
        'optimization': {
            'n_trials': 1,  # We're not optimizing, just using the found parameters
            'patience': 10,
            'seed': 42,
            'metric': 'cidx' if 'Concordance' in optimization_target else 'loglik'
        },
        'balance': balance_config
    }
    
    return config


def load_model_with_flexible_architecture(
    model_path: str,
    model_type: str,
    network_type: str,
    model_details: Dict[str, Any],
    device: str = 'cpu'
) -> Any:
    """
    Load a model with flexible architecture matching the saved weights.
    
    Args:
        model_path: Path to model weights
        model_type: 'deepsurv' or 'deephit'
        network_type: 'ann' or 'lstm'
        model_details: Model configuration details
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Load the state dict to inspect architecture
    state_dict = torch.load(model_path, map_location=device)
    
    # For LSTM models, infer architecture from state dict
    if network_type == 'lstm':
        # Extract LSTM hidden dimensions from state dict
        lstm_hidden_dims = []
        lstm_layers = []
        
        # Find all LSTM layers
        for key in state_dict.keys():
            if 'lstm_layers' in key and 'weight_ih_l0' in key:
                layer_idx = int(key.split('.')[1])
                weight_shape = state_dict[key].shape
                hidden_dim = weight_shape[0] // 4  # LSTM has 4 gates
                lstm_layers.append((layer_idx, hidden_dim))
        
        # Sort by layer index
        lstm_layers.sort(key=lambda x: x[0])
        lstm_hidden_dims = [dim for _, dim in lstm_layers]
        
        # Check if bidirectional
        bidirectional = any('reverse' in key for key in state_dict.keys())
        
        # Update model details
        model_details['lstm_hidden_dims'] = lstm_hidden_dims
        model_details['lstm_num_layers'] = len(lstm_hidden_dims)
        model_details['bidirectional'] = bidirectional
        
        print(f"Detected LSTM architecture: layers={len(lstm_hidden_dims)}, dims={lstm_hidden_dims}, bidirectional={bidirectional}")
    
    # Create network with detected architecture
    input_dim = model_details.get('input_dim', 11)
    output_dim = model_details.get('output_dim', 1)
    dropout = model_details.get('dropout', 0.2)
    
    if network_type == 'lstm':
        net = create_network(
            model_type=model_type,
            network_type=network_type,
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
            lstm_num_layers=len(lstm_hidden_dims),
            bidirectional=bidirectional,
            sequence_length=model_details.get('sequence_length', 5),
            lstm_hidden_dim=lstm_hidden_dims[0] if lstm_hidden_dims else 64
        )
    else:
        # For ANN, extract hidden dimensions from state dict
        hidden_dims = model_details.get('hidden_dims', [128, 64, 32])
        
        # Try to infer from state dict if not provided
        if 'layers.0.weight' in state_dict:
            hidden_dims = []
            layer_idx = 0
            while f'layers.{layer_idx}.weight' in state_dict:
                weight_shape = state_dict[f'layers.{layer_idx}.weight'].shape
                hidden_dims.append(weight_shape[0])
                layer_idx += 2  # Skip activation layers
            
            # Remove the last dimension as it's the output layer
            if hidden_dims and hidden_dims[-1] == output_dim:
                hidden_dims = hidden_dims[:-1]
        
        net = create_network(
            model_type=model_type,
            network_type=network_type,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )
    
    # Create optimizer (dummy, won't be used for inference)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # Create model
    if model_type.lower() == "deepsurv":
        model = CoxPH(net, optimizer=optimizer)
    else:  # deephit
        alpha = model_details.get('alpha', 0.2)
        sigma = model_details.get('sigma', 0.1)
        time_grid = np.array(model_details.get('time_grid', [365, 730, 1095, 1460, 1825]))
        model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
    
    # Load the state dict
    model.net.load_state_dict(state_dict)
    model.net = model.net.to(device)
    model.net.eval()
    
    return model


def create_and_train_model(
    model_config: Dict[str, Any],
    train_data: pd.DataFrame,
    feature_cols: List[str],
    device: str = 'cpu',
    retrain: bool = True
) -> Any:
    """
    Create and optionally train a model based on configuration.
    
    Args:
        model_config: Model configuration dictionary
        train_data: Training data (preprocessed)
        feature_cols: List of feature columns
        device: Device to load model on ('cpu' or 'cuda')
        retrain: Whether to retrain the model or just load weights
        
    Returns:
        Trained or loaded model
    """
    model_details = model_config['model_details']
    model_type = model_config['algorithm'].lower()
    network_type = model_config['structure'].lower()
    
    # Extract model architecture details
    input_dim = model_details.get('input_dim', len(feature_cols))
    output_dim = model_details.get('output_dim', 1)
    hidden_dims = model_details.get('hidden_dims', [128, 64, 32])
    dropout = model_details.get('dropout', 0.2)
    
    # LSTM specific parameters
    lstm_num_layers = model_details.get('lstm_num_layers', 2)
    lstm_bidirectional = model_details.get('bidirectional', True)
    sequence_length = model_details.get('sequence_length', 5)
    
    if not retrain:
        # Just load existing weights with flexible architecture
        model_path = model_details['model_path']
        if not os.path.exists(model_path):
            # Try alternative path in results/model_details
            alt_path = os.path.join('results/model_details', os.path.basename(model_path))
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(f"Model weights not found at {model_path} or {alt_path}")
        
        print(f"Loading pre-trained weights from {model_path}")
        return load_model_with_flexible_architecture(
            model_path=model_path,
            model_type=model_type,
            network_type=network_type,
            model_details=model_details,
            device=device
        )
    
    # If retraining, create network normally
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
    
    # Create optimizer
    optimizer_name = model_details.get('optimizer', 'Adam')
    lr = model_details.get('learning_rate', 0.001)
    
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    
    # Create model
    if model_type.lower() == "deepsurv":
        model = CoxPH(net, optimizer=optimizer)
    else:  # deephit
        alpha = model_details.get('alpha', 0.2)
        sigma = model_details.get('sigma', 0.1)
        time_grid = np.array(model_details.get('time_grid', [365, 730, 1095, 1460, 1825]))
        model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
    
    # Move model to device
    model.net = model.net.to(device)
    
    # Training code would go here if retrain=True
    # For now, we're just loading pre-trained models
    
    return model


def generate_predictions(
    model: Any,
    test_data: pd.DataFrame,
    feature_cols: List[str],
    model_type: str,
    network_type: str,
    model_config: Dict[str, Any],
    device: str = 'cpu'
) -> np.ndarray:
    """
    Generate predictions from a model.
    
    Args:
        model: Trained model
        test_data: Test data
        feature_cols: Feature columns
        model_type: 'deepsurv' or 'deephit'
        network_type: 'ann' or 'lstm'
        model_config: Model configuration with sequence length info
        device: Device for computation
        
    Returns:
        Predictions array
    """
    # Prepare data
    X_test = test_data[feature_cols].values
    
    # Convert to appropriate format
    if network_type.lower() == 'lstm':
        # Get sequence length from model config
        sequence_length = 5  # Default
        if 'model_details' in model_config and 'sequence_length' in model_config['model_details']:
            sequence_length = model_config['model_details']['sequence_length']
        elif 'optimization_details' in model_config:
            opt_details = model_config['optimization_details']
            if 'best_params' in opt_details and 'sequence' in opt_details['best_params']:
                sequence_length = opt_details['best_params']['sequence']
        
        n_samples = X_test.shape[0]
        n_features = X_test.shape[1]
        
        # Create sequences by repeating the same features
        X_test_seq = np.repeat(X_test[:, np.newaxis, :], sequence_length, axis=1)
        X_test = X_test_seq.astype('float32')
    else:
        X_test = X_test.astype('float32')
    
    # Generate predictions
    model.net.eval()
    with torch.no_grad():
        if model_type.lower() == 'deepsurv':
            # DeepSurv returns hazard predictions
            predictions = model.predict_surv_df(X_test).values  # Shape: (n_time_points, n_samples)
        else:
            # DeepHit returns CIF directly
            predictions = model.predict_cif(X_test)  # Shape: (n_samples, n_causes, n_time_points)
            # Move to CPU if on GPU
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            # Transpose to (n_causes, n_time_points, n_samples)
            predictions = predictions.transpose(1, 2, 0)
    
    return predictions


def convert_deepsurv_to_cif(
    surv_predictions: np.ndarray,
    event_type: int
) -> np.ndarray:
    """
    Convert DeepSurv survival predictions to CIF format.
    
    Args:
        surv_predictions: Survival function predictions (n_time_points, n_samples)
        event_type: 1 or 2 for the event type
        
    Returns:
        CIF predictions (2, n_time_points, n_samples)
    """
    n_time_points, n_samples = surv_predictions.shape
    
    # Calculate hazard from survival function
    hazard = np.zeros_like(surv_predictions)
    hazard[0] = 1 - surv_predictions[0]
    hazard[1:] = surv_predictions[:-1] - surv_predictions[1:]
    
    # Create CIF array
    cif = np.zeros((2, n_time_points, n_samples))
    
    # Assign hazard to the appropriate event
    if event_type == 1:
        cif[0] = np.cumsum(hazard, axis=0)
        # Event 2 remains zero
    else:  # event_type == 2
        cif[1] = np.cumsum(hazard, axis=0)
        # Event 1 remains zero
    
    return cif


def extract_time_points(
    predictions: np.ndarray,
    original_time_grid: np.ndarray,
    target_time_points: np.ndarray
) -> np.ndarray:
    """
    Extract predictions at specific time points.
    
    Args:
        predictions: CIF predictions (2, n_time_points, n_samples)
        original_time_grid: Original time grid (1825 points for DeepSurv)
        target_time_points: Target time points [365, 730, 1095, 1460, 1825]
        
    Returns:
        Extracted predictions (2, 5, n_samples)
    """
    n_causes, _, n_samples = predictions.shape
    n_target_points = len(target_time_points)
    
    extracted = np.zeros((n_causes, n_target_points, n_samples))
    
    for i, t in enumerate(target_time_points):
        # Find closest time point in original grid
        idx = np.argmin(np.abs(original_time_grid - t))
        extracted[:, i, :] = predictions[:, idx, :]
    
    return extracted


def stack_predictions_by_group(
    predictions_dict: Dict[int, Dict[str, Any]],
    model_configs: List[Dict[str, Any]]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Stack predictions by matching algorithm, structure, balancing, and optimization.
    
    Args:
        predictions_dict: Dictionary of predictions by model number
        model_configs: List of model configurations
        
    Returns:
        Tuple of (stacked_deepsurv_predictions, deephit_predictions)
    """
    # Group DeepSurv models
    deepsurv_groups = {}
    deephit_predictions = []
    
    for model_no, pred_data in predictions_dict.items():
        # Find corresponding config
        config = next((c for c in model_configs if c['model_no'] == model_no), None)
        if not config:
            continue
        
        algorithm = config['algorithm']
        structure = config['structure']
        balancing = config['balancing_method']
        optimization = config['optimization_target']
        
        # Handle NaN balancing - check both pandas NaN and string 'nan'
        if pd.isna(balancing) or str(balancing).lower() == 'nan':
            balancing = 'None'
        
        if algorithm.lower() == 'deepsurv':
            # Create group key
            group_key = (algorithm, structure, balancing, optimization)
            
            if group_key not in deepsurv_groups:
                deepsurv_groups[group_key] = []
            
            deepsurv_groups[group_key].append(pred_data['predictions'])
        else:
            # DeepHit predictions
            deephit_predictions.append(pred_data['predictions'])
    
    # Stack DeepSurv groups
    stacked_deepsurv = []
    for group_key, group_preds in deepsurv_groups.items():
        if len(group_preds) == 2:
            # We have both Event 1 and Event 2
            stacked = np.stack(group_preds, axis=0)  # Shape: (2, 2, 5, n_samples)
            # Average across the two models
            averaged = np.mean(stacked, axis=0)  # Shape: (2, 5, n_samples)
            stacked_deepsurv.append(averaged)
            print(f"Stacked group {group_key}: shape {averaged.shape}")
        else:
            print(f"Warning: Incomplete group {group_key} - has {len(group_preds)} models instead of 2")
    
    return stacked_deepsurv, deephit_predictions


def ensemble_predictions(
    all_predictions: np.ndarray,
    method: str = 'average'
) -> np.ndarray:
    """
    Ensemble multiple predictions.
    
    Args:
        all_predictions: Array of shape (n_models, 2, 5, n_samples)
        method: Ensemble method ('average', 'weighted', etc.)
        
    Returns:
        Ensemble predictions of shape (2, 5, n_samples)
    """
    if method == 'average':
        return np.mean(all_predictions, axis=0)
    else:
        # Add other ensemble methods here
        raise NotImplementedError(f"Ensemble method {method} not implemented")


# ==================== ZenML Pipeline ====================

@pipeline(enable_cache=True)
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
        demo_df=demo_df,
        icd10_df=icd10_df,
        death_df=death_df,
        operation_df=operation_df
    )
    
    raw_df, prediction_df = merge_data(
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
    
    temporal_train_df, temporal_test_df, spatial_train_df, spatial_test_df = split_data(
        raw_df=raw_df,
        prediction_df=prediction_df
    )
    
    temporal_train_imputed, temporal_test_imputed = impute_data(temporal_train_df, temporal_test_df)
    spatial_train_imputed, spatial_test_imputed = impute_data(spatial_train_df, spatial_test_df)
    
    temporal_train_preprocessed, temporal_test_preprocessed, temporal_scaler = preprocess_data(temporal_train_imputed, temporal_test_imputed)
    spatial_train_preprocessed, spatial_test_preprocessed, spatial_scaler = preprocess_data(spatial_train_imputed, spatial_test_imputed)
    
    # Deploy multiple models
    deploy_multiple_models_v2(
        temporal_train_preprocessed,
        temporal_test_preprocessed,
        spatial_train_preprocessed,
        spatial_test_preprocessed
    )


@step(enable_cache=False)
def deploy_multiple_models_v2(
    temporal_train_df: pd.DataFrame,
    temporal_test_df: pd.DataFrame,
    spatial_train_df: pd.DataFrame,
    spatial_test_df: pd.DataFrame,
    config_csv_path: str = "results/final_deploy/model_config/model_config.csv",
    config_dir: str = "results/final_deploy/model_config",
    output_dir: str = "results/final_deploy",
    retrain_models: bool = False,
    ensemble_method: str = 'average'
) -> Dict[str, Any]:
    """
    Deploy multiple models and generate ensemble predictions.
    """
    print(f"\n{'='*60}")
    print("MULTI-MODEL DEPLOYMENT PIPELINE V2 (FIXED)")
    print(f"{'='*60}")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load all model configurations
    model_configs = load_all_model_configurations(config_csv_path, config_dir)
    
    # Get feature columns (excluding target columns)
    exclude_cols = ['duration', 'endpoint', 'key', 'date', 'endpoint_date', 
                   'first_sub_60_date', 'dob', 'icd10']
    feature_cols = [col for col in temporal_train_df.columns if col not in exclude_cols]
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    
    # Create output directories
    individual_dir = os.path.join(output_dir, "individual_predictions")
    ensemble_dir = os.path.join(output_dir, "ensemble_predictions")
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
        print(f"Optimization: {config['optimization_target']}")
        
        try:
            # Create model-specific configuration
            model_specific_config = create_model_specific_config(config)
            
            # Create and load model
            model = create_and_train_model(
                model_config=config,
                train_data=temporal_train_df,
                feature_cols=feature_cols,
                device=device,
                retrain=retrain_models
            )
            
            # Generate predictions for temporal test
            print("Generating temporal test predictions...")
            temp_predictions = generate_predictions(
                model=model,
                test_data=temporal_test_df,
                feature_cols=feature_cols,
                model_type=config['algorithm'],
                network_type=config['structure'],
                model_config=config,
                device=device
            )
            
            # Convert DeepSurv to CIF if needed
            if config['algorithm'].lower() == 'deepsurv':
                # DeepSurv predictions need conversion to CIF
                event_type = 1 if config['prediction_endpoint'] == 'Event 1' else 2
                temp_cif = convert_deepsurv_to_cif(temp_predictions, event_type)
                
                # Extract specific time points
                original_time_grid = np.arange(1, 1826)  # 1 to 1825 days
                target_time_points = np.array([365, 730, 1095, 1460, 1825])
                temp_cif_extracted = extract_time_points(temp_cif, original_time_grid, target_time_points)
            else:
                # DeepHit already in correct format
                temp_cif_extracted = temp_predictions
            
            # Save temporal predictions
            temporal_predictions[model_no] = {
                'predictions': temp_cif_extracted,
                'durations': temporal_test_df['duration'].values,
                'events': temporal_test_df['endpoint'].values
            }
            
            # Save individual predictions
            save_path = os.path.join(individual_dir, f"model{model_no}_predictions.npz")
            np.savez(save_path,
                     predictions=temp_predictions,
                     cif_predictions=temp_cif_extracted,
                     durations=temporal_test_df['duration'].values,
                     events=temporal_test_df['endpoint'].values,
                     metadata=json.dumps(config))
            
            # Generate predictions for spatial test
            print("Generating spatial test predictions...")
            spat_predictions = generate_predictions(
                model=model,
                test_data=spatial_test_df,
                feature_cols=feature_cols,
                model_type=config['algorithm'],
                network_type=config['structure'],
                model_config=config,
                device=device
            )
            
            # Convert and extract for spatial
            if config['algorithm'].lower() == 'deepsurv':
                event_type = 1 if config['prediction_endpoint'] == 'Event 1' else 2
                spat_cif = convert_deepsurv_to_cif(spat_predictions, event_type)
                spat_cif_extracted = extract_time_points(spat_cif, original_time_grid, target_time_points)
            else:
                spat_cif_extracted = spat_predictions
            
            # Save spatial predictions
            spatial_predictions[model_no] = {
                'predictions': spat_cif_extracted,
                'durations': spatial_test_df['duration'].values,
                'events': spatial_test_df['endpoint'].values
            }
            
            print(f"Successfully processed model {model_no}")
            
        except Exception as e:
            print(f"Error processing model {model_no}: {e}")
            continue
    
    # Stack predictions by group
    print(f"\n=== Stacking Predictions ===")
    temp_deepsurv_stacked, temp_deephit_preds = stack_predictions_by_group(temporal_predictions, model_configs)
    spat_deepsurv_stacked, spat_deephit_preds = stack_predictions_by_group(spatial_predictions, model_configs)
    
    print(f"Stacked {len(temp_deepsurv_stacked)} DeepSurv groups for temporal test")
    print(f"Stacked {len(spat_deepsurv_stacked)} DeepSurv groups for spatial test")
    print(f"Collected {len(temp_deephit_preds)} DeepHit predictions")
    
    # Create ensemble
    print(f"\n=== Creating Ensemble ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
        
        # Use the first model's durations and events
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
    
    # Save test data for ensemble evaluation
    test_data_path = os.path.join(individual_dir, "test_data.npz")
    np.savez(test_data_path,
             X_test=temporal_test_df[feature_cols].values,
             y_test=np.rec.fromarrays([temp_durations, temp_events], names=['time', 'event']))
    
    # Save deployment log
    deployment_log = {
        'timestamp': timestamp,
        'n_models_processed': len(temporal_predictions),
        'n_deepsurv_groups': len(temp_deepsurv_stacked),
        'n_deephit_models': len(temp_deephit_preds),
        'ensemble_method': ensemble_method,
        'retrain_models': retrain_models,
        'model_configs': [
            {
                'model_no': c['model_no'],
                'algorithm': c['algorithm'],
                'structure': c['structure'],
                'endpoint': c['prediction_endpoint'],
                'balancing': c['balancing_method'],
                'optimization': c['optimization_target']
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
        'temporal_predictions': temporal_predictions,
        'spatial_predictions': spatial_predictions,
        'ensemble_dir': ensemble_dir,
        'individual_dir': individual_dir,
        'timestamp': timestamp,
        'retrain_models': retrain_models
    }


if __name__ == "__main__":
    # Run the pipeline
    pipeline = multi_model_deploy_pipeline()
    # In ZenML, calling the pipeline function directly executes it
    # The pipeline object is already the result of execution