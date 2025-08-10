"""
Generate predictions for all models with baseline hazards.

This step loads all models (both DeepSurv and DeepHit), loads baseline hazards
for DeepSurv models, and generates predictions for both temporal and spatial test sets.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import h5py
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any
from zenml import step
import re

from src.sequence_utils_fixed import create_sequences_from_dataframe_fixed
from src.nn_architectures import MLP, CNNMLP, LSTMSurvival, LSTMDeepHit
from pycox.models import CoxPH, DeepHit


def get_feature_columns():
    """Get the feature columns used for modeling"""
    feature_cols = [
        'gender', 'creatinine', 'hemoglobin', 'phosphate',
        'age_at_obs', 'bicarbonate', 'albumin',
        'uacr', 'cci_score_total', 'ht', 'observation_period'
    ]
    return feature_cols


def load_baseline_hazards(model_name: str, models_dir: str) -> Dict[str, Any]:
    """Load baseline hazards for a DeepSurv model"""
    # Extract model number from model name
    match = re.search(r'model(\d+)', model_name)
    if match:
        model_num = match.group(1)
        # Look for baseline hazards file
        baseline_pattern = f'baseline_hazards_model{model_num}_*.pkl'
        baseline_files = list(Path(models_dir).glob(baseline_pattern))
        
        if baseline_files:
            baseline_path = baseline_files[0]  # Take the first match
            print(f"Loading baseline hazards from: {baseline_path}")
            with open(baseline_path, 'rb') as f:
                baseline_data = pickle.load(f)
            return baseline_data
        else:
            print(f"Warning: Baseline hazards not found for model{model_num}")
    else:
        print(f"Warning: Could not extract model number from {model_name}")
    return None


def detect_mlp_architecture(state_dict):
    """Detect MLP architecture from state dict"""
    hidden_dims = []
    layer_idx = 0
    
    while f'model.{layer_idx}.weight' in state_dict:
        weight_shape = state_dict[f'model.{layer_idx}.weight'].shape
        hidden_dims.append(weight_shape[0])
        # Skip batch norm, activation, and dropout layers
        layer_idx += 4
    
    # Remove the last dimension (output layer)
    if hidden_dims:
        hidden_dims = hidden_dims[:-1]
    
    return hidden_dims


def detect_cnnmlp_architecture(state_dict):
    """Detect CNN-MLP architecture from state dict"""
    # For DeepHit, we need to check cause_networks
    hidden_dims = []
    layer_idx = 0
    
    # Check first cause network
    while f'cause_networks.0.{layer_idx}.weight' in state_dict:
        weight_shape = state_dict[f'cause_networks.0.{layer_idx}.weight'].shape
        if len(weight_shape) == 2:  # Linear layer
            hidden_dims.append(weight_shape[0])
        # Skip batch norm, activation, and dropout layers
        layer_idx += 4
    
    # Remove the last dimension (output layer)
    if hidden_dims:
        hidden_dims = hidden_dims[:-1]
    
    return hidden_dims


def detect_lstm_architecture(state_dict):
    """Detect LSTM architecture from state dict"""
    lstm_hidden_dims = []
    lstm_layers = []
    
    # Find all LSTM layers
    for key in state_dict.keys():
        if 'lstm_layers' in key and 'weight_ih_l0' in key and 'reverse' not in key:
            layer_idx = int(key.split('.')[1])
            weight_shape = state_dict[key].shape
            hidden_dim = weight_shape[0] // 4  # LSTM has 4 gates
            lstm_layers.append((layer_idx, hidden_dim))
    
    # Sort by layer index
    lstm_layers.sort(key=lambda x: x[0])
    lstm_hidden_dims = [dim for _, dim in lstm_layers]
    
    # Check if bidirectional
    bidirectional = any('reverse' in key for key in state_dict.keys())
    
    return lstm_hidden_dims, bidirectional


def load_model_for_prediction(model_info: Dict[str, Any], models_dir: str, device: str = 'cpu'):
    """Load a model with the correct architecture for prediction"""
    model_name = model_info['model_name']
    model_type = model_info['model_type']
    
    # Find the model file with timestamp
    model_pattern = f"{model_name}_*.pt"
    model_files = list(Path(models_dir).glob(model_pattern))
    
    if not model_files:
        print(f"Model file not found for pattern: {model_pattern}")
        return None, None, None
    
    # Take the first match (should be only one)
    model_path = model_files[0]
    print(f"Found model file: {model_path}")
    
    # Load model configuration JSON
    match = re.search(r'model(\d+)', model_name)
    if match:
        model_num = match.group(1)
        json_pattern = f"model{model_num}_details_*.json"
        json_files = list(Path("results/final_deploy/model_config").glob(json_pattern))
        
        if json_files:
            with open(json_files[0], 'r') as f:
                model_config = json.load(f)
            print(f"Loaded model configuration from {json_files[0]}")
        else:
            print(f"Warning: Model configuration not found for model {model_num}")
            model_config = {}
    else:
        model_config = {}
    
    # Determine network type
    if 'LSTM' in model_name:
        network_type = 'lstm'
    elif 'ANN' in model_name:
        network_type = 'ann'
    else:
        network_type = 'ann'
    
    # Get model parameters
    input_dim = model_config.get('input_dim', model_info.get('input_dim', 11))
    output_dim = model_config.get('output_dim', model_info.get('output_dim', 1))
    dropout = model_config.get('dropout', model_info.get('hyperparameters', {}).get('dropout', 0.2))
    sequence_length = model_config.get('sequence_length', 10)  # Default to 10 if not found
    
    # Load state dict to inspect architecture
    state_dict = torch.load(model_path, map_location=device)
    
    try:
        if network_type == 'lstm':
            # Extract LSTM architecture from state dict
            lstm_hidden_dims, bidirectional = detect_lstm_architecture(state_dict)
            
            print(f"Detected LSTM architecture: layers={len(lstm_hidden_dims)}, dims={lstm_hidden_dims}, bidirectional={bidirectional}")
            
            if model_type.lower() == "deepsurv":
                net = LSTMSurvival(
                    input_dim=input_dim,
                    sequence_length=sequence_length,
                    lstm_hidden_dims=lstm_hidden_dims,
                    output_dim=output_dim,
                    dropout=dropout,
                    bidirectional=bidirectional
                )
            else:  # DeepHit
                net = LSTMDeepHit(
                    input_dim=input_dim,
                    sequence_length=sequence_length,
                    lstm_hidden_dims=lstm_hidden_dims,
                    output_dim=5,  # 5 time intervals
                    num_causes=2,
                    dropout=dropout,
                    bidirectional=bidirectional
                )
        else:
            # ANN model
            if model_type.lower() == "deepsurv":
                hidden_dims = detect_mlp_architecture(state_dict)
                print(f"Detected MLP architecture: hidden_dims={hidden_dims}")
                net = MLP(
                    in_features=input_dim,
                    hidden_dims=hidden_dims,
                    out_features=output_dim,
                    dropout=dropout,
                    batch_norm=True
                )
            else:  # DeepHit
                hidden_dims = detect_cnnmlp_architecture(state_dict)
                print(f"Detected CNN-MLP architecture: hidden_dims={hidden_dims}")
                net = CNNMLP(
                    in_features=input_dim,
                    hidden_dims=hidden_dims,
                    out_features=5,  # 5 time intervals
                    num_causes=2,
                    dropout=dropout,
                    batch_norm=True
                )
        
        # Create optimizer (dummy, won't be used for inference)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        
        # Create model
        if model_type.lower() == "deepsurv":
            model = CoxPH(net, optimizer=optimizer)
        else:  # deephit
            alpha = model_info.get('hyperparameters', {}).get('alpha', 0.2)
            sigma = model_info.get('hyperparameters', {}).get('sigma', 0.1)
            time_grid = np.array(model_info.get('time_grid', [365, 730, 1095, 1460, 1825]))
            model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
        
        # Load the state dict
        model.net.load_state_dict(state_dict)
        model.net.to(device)
        model.net.eval()
        
        # For DeepSurv models, load baseline hazards
        if model_type.lower() == "deepsurv":
            baseline_data = load_baseline_hazards(model_name, models_dir)
            if baseline_data:
                model.baseline_hazards_ = baseline_data['baseline_hazards_']
                model.baseline_cumulative_hazards_ = baseline_data['baseline_cumulative_hazards_']
                print("Baseline hazards loaded successfully")
            else:
                print("ERROR: Could not load baseline hazards for DeepSurv model!")
                return None, network_type
        
        return model, network_type, model_config
        
    except Exception as e:
        print(f"Error creating model architecture: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, network_type, {}


def prepare_data_for_model(df: pd.DataFrame, feature_cols: list, network_type: str,
                          sequence_length: int = 5, target_endpoint: int = None,
                          include_all_patients: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for model prediction"""
    if network_type == 'lstm':
        # For LSTM, create a sequence for EACH row by looking back
        print(f"Creating sequences for each row with sequence_length={sequence_length}")
        
        # Sort by patient and date
        df_sorted = df.sort_values(['key', 'date']).reset_index(drop=True)
        
        # Initialize arrays for all rows
        n_rows = len(df_sorted)
        n_features = len(feature_cols)
        sequences = np.zeros((n_rows, sequence_length, n_features), dtype='float32')
        
        # Get durations and events for all rows
        durations = df_sorted['duration'].values.astype('float32')
        events = df_sorted['endpoint'].values.astype('float32')
        patient_ids = df_sorted['key'].values
        
        # For DeepSurv models, filter events to the specific endpoint
        if target_endpoint is not None:
            events = (events == target_endpoint).astype('float32')
        
        # Create sequence for each row
        for idx in range(n_rows):
            patient_id = patient_ids[idx]
            
            # Get all rows for this patient up to and including current row
            patient_mask = (df_sorted['key'] == patient_id) & (df_sorted.index <= idx)
            patient_rows = df_sorted[patient_mask]
            
            # Get the feature values
            patient_features = patient_rows[feature_cols].values
            
            # Create sequence
            if len(patient_features) >= sequence_length:
                # Take the last sequence_length observations
                sequences[idx] = patient_features[-sequence_length:]
            else:
                # Pad with zeros at the beginning
                pad_length = sequence_length - len(patient_features)
                sequences[idx, pad_length:] = patient_features
                # sequences[idx, :pad_length] remains zeros
        
        print(f"Created sequences shape: {sequences.shape}")
        print(f"Total rows: {n_rows}")
        
        return sequences, durations, events, patient_ids
        
    else:
        # For ANN models, use ALL observations (not just latest per patient)
        # This is important for getting the full dataset shape
        X = df[feature_cols].values.astype('float32')
        
        # Always use 'duration' and 'endpoint' columns
        durations = df['duration'].values.astype('float32')
        events = df['endpoint'].values.astype('float32')
        
        # For DeepSurv models, filter events to the specific endpoint
        if target_endpoint is not None:
            # Create binary event indicator for the specific endpoint
            events = (events == target_endpoint).astype('float32')
        
        patient_ids = df['key'].values
        
        return X, durations, events, patient_ids


def generate_predictions(model, model_type: str, X: np.ndarray, device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate predictions for a model"""
    batch_size = 256  # Process in smaller batches to avoid memory issues
    n_samples = X.shape[0]
    
    with torch.no_grad():
        if model_type.lower() == 'deepsurv':
            # DeepSurv predictions
            X_tensor = torch.FloatTensor(X).to(device)
            surv_df = model.predict_surv_df(X_tensor)
            # Convert to CIF (1 - survival)
            cif = 1 - surv_df.values
            return cif, None, None
        else:
            # DeepHit predictions - process in batches
            all_cif = []
            
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X[i:batch_end]
                X_tensor = torch.FloatTensor(X_batch).to(device)
                
                # Use predict_cif which returns (n_events, n_times, n_samples)
                batch_cif = model.predict_cif(X_tensor)
                
                # Convert to numpy if it's a tensor
                if isinstance(batch_cif, torch.Tensor):
                    batch_cif = batch_cif.cpu().numpy()
                
                # Ensure correct shape (n_events, n_times, batch_size)
                if batch_cif.shape[0] != 2:
                    print(f"Warning: Expected 2 events, got {batch_cif.shape[0]}")
                if batch_cif.shape[1] != 5:
                    print(f"Warning: Expected 5 time points, got {batch_cif.shape[1]}")
                
                all_cif.append(batch_cif)
            
            # Concatenate all batches
            cif = np.concatenate(all_cif, axis=2)  # (n_events, n_times, n_samples)
            
            # Extract individual event CIFs
            cause1_cif = cif[0]  # (n_times, n_samples)
            cause2_cif = cif[1]  # (n_times, n_samples)
            
            return cif, cause1_cif, cause2_cif


@step(enable_cache=False)
def generate_all_predictions_with_baseline(
    temporal_test_df_preprocessed: pd.DataFrame,
    spatial_test_df_preprocessed: pd.DataFrame,
    models_dir: str = "results/final_deploy/models",
    model_config_dir: str = "results/final_deploy/model_config",
    output_dir: str = "results/final_deploy/individual_predictions"
) -> Dict[str, Any]:
    """
    Generate predictions for all models with baseline hazards.
    
    Args:
        temporal_test_df_preprocessed: Preprocessed temporal test dataframe
        spatial_test_df_preprocessed: Preprocessed spatial test dataframe
        models_dir: Directory containing model weights and baseline hazards
        model_config_dir: Directory containing model configurations
        output_dir: Directory to save individual predictions
        
    Returns:
        Dictionary with summary of prediction generation results
    """
    print("\n=== Generating Predictions for All Models ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model configurations
    model_config_path = os.path.join(model_config_dir, "model_config.json")
    if not os.path.exists(model_config_path):
        # Try loading from ZenML artifact
        model_config_path = '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_0/67c9050f-777c-40ad-b087-a4bd53d6c47a/99a24532/data.json'
    
    with open(model_config_path, 'r') as f:
        all_models = json.load(f)
    
    print(f"Found {len(all_models)} models to process")
    
    # Get feature columns
    feature_cols = get_feature_columns()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Process each model
    successful = 0
    failed = 0
    results = []
    
    # Process each model individually
    for i, model_info in enumerate(all_models):
        model_name = model_info['model_name']
        model_type = model_info['model_type']
        
        # Extract model number
        match = re.search(r'model(\d+)', model_name)
        if match:
            model_num = int(match.group(1))
        else:
            model_num = i + 1
        
        print(f"\n{'='*60}")
        print(f"Processing model {i+1}/{len(all_models)}: {model_name}")
        print(f"Model type: {model_type}")
        print(f"Model number: {model_num}")
        
        try:
            # Load the model
            model, network_type, model_config = load_model_for_prediction(model_info, models_dir, device)
            if model is None:
                print(f"Failed to load model: {model_name}")
                failed += 1
                continue
            
            # Get target endpoint for DeepSurv models
            target_endpoint = model_info.get('model_endpoint') if model_type.lower() == 'deepsurv' else None
            
            # Get sequence length from model config
            sequence_length = model_config.get('sequence_length', 10)
            print(f"Using sequence length: {sequence_length}")
            
            # Prepare temporal test data
            print("\nPreparing temporal test data...")
            temporal_X, temporal_durations, temporal_events, temporal_ids = prepare_data_for_model(
                temporal_test_df_preprocessed, feature_cols, network_type,
                sequence_length=sequence_length,
                target_endpoint=target_endpoint, include_all_patients=True
            )
            print(f"Temporal data shape: {temporal_X.shape}")
            print(f"Number of temporal patients: {len(temporal_ids)}")
            
            # Prepare spatial test data
            print("\nPreparing spatial test data...")
            spatial_X, spatial_durations, spatial_events, spatial_ids = prepare_data_for_model(
                spatial_test_df_preprocessed, feature_cols, network_type,
                sequence_length=sequence_length,
                target_endpoint=target_endpoint, include_all_patients=True
            )
            print(f"Spatial data shape: {spatial_X.shape}")
            print(f"Number of spatial patients: {len(spatial_ids)}")
            
            # Generate predictions
            print("\nGenerating predictions...")
            temporal_preds, temporal_c1, temporal_c2 = generate_predictions(
                model, model_type, temporal_X, device
            )
            spatial_preds, spatial_c1, spatial_c2 = generate_predictions(
                model, model_type, spatial_X, device
            )
            
            print(f"Temporal predictions shape: {temporal_preds.shape}")
            print(f"Spatial predictions shape: {spatial_preds.shape}")
            
            # For DeepSurv models, we save predictions as-is (1825, n_samples)
            # The stacking will be done later in the ensemble pipeline
            if model_type.lower() == 'deepsurv':
                print(f"DeepSurv model - saving predictions for Event {target_endpoint}")
                print(f"Temporal predictions shape: {temporal_preds.shape}")
                print(f"Spatial predictions shape: {spatial_preds.shape}")
            
            # Save individual predictions
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save temporal predictions
            temporal_file = os.path.join(output_dir, f'temporal_predictions_model{model_num}_{timestamp}.h5')
            with h5py.File(temporal_file, 'w') as f:
                f.create_dataset('predictions', data=temporal_preds)
                f.attrs['model_name'] = model_name
                f.attrs['model_type'] = model_type
                f.attrs['model_number'] = model_num
                f.attrs['n_samples'] = temporal_preds.shape[-1]
                f.attrs['shape'] = str(temporal_preds.shape)
                if target_endpoint is not None:
                    f.attrs['target_endpoint'] = target_endpoint
            
            # Save spatial predictions
            spatial_file = os.path.join(output_dir, f'spatial_predictions_model{model_num}_{timestamp}.h5')
            with h5py.File(spatial_file, 'w') as f:
                f.create_dataset('predictions', data=spatial_preds)
                f.attrs['model_name'] = model_name
                f.attrs['model_type'] = model_type
                f.attrs['model_number'] = model_num
                f.attrs['n_samples'] = spatial_preds.shape[-1]
                f.attrs['shape'] = str(spatial_preds.shape)
                if target_endpoint is not None:
                    f.attrs['target_endpoint'] = target_endpoint
            
            # Store results
            results.append({
                'model_name': model_name,
                'model_type': model_type,
                'model_number': model_num,
                'network_type': network_type,
                'temporal_predictions_path': temporal_file,
                'spatial_predictions_path': spatial_file,
                'temporal_n_samples': len(temporal_ids),
                'spatial_n_samples': len(spatial_ids),
                'temporal_shape': str(temporal_preds.shape),
                'spatial_shape': str(spatial_preds.shape)
            })
            
            
            successful += 1
            print(f"Successfully processed {model_name}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(all_models),
        'successful': successful,
        'failed': failed,
        'results': results
    }
    
    summary_file = os.path.join(output_dir, f'prediction_generation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Prediction generation complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Summary saved to: {summary_file}")
    
    return summary