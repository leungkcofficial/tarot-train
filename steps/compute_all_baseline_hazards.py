"""
Compute All Baseline Hazards Step

This step computes baseline hazards for all DeepSurv models in the final deployment directory.
"""

import os
import json
import glob
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from zenml.steps import step
from typing import Dict, Any, List, Tuple
from pycox.models import CoxPH
import traceback


def detect_lstm_architecture_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Detect LSTM architecture from saved state dict.
    
    Args:
        state_dict: The saved model state dictionary
        
    Returns:
        Dictionary with detected architecture parameters
    """
    # Find LSTM layers
    lstm_layers = []
    layer_idx = 0
    
    while f'lstm_layers.{layer_idx}.weight_ih_l0' in state_dict:
        weight_ih = state_dict[f'lstm_layers.{layer_idx}.weight_ih_l0']
        weight_hh = state_dict[f'lstm_layers.{layer_idx}.weight_hh_l0']
        
        # weight_ih shape is (4*hidden_size, input_size)
        # weight_hh shape is (4*hidden_size, hidden_size)
        hidden_size = weight_hh.shape[0] // 4
        input_size = weight_ih.shape[1]
        
        # Check if bidirectional
        is_bidirectional = f'lstm_layers.{layer_idx}.weight_ih_l0_reverse' in state_dict
        
        lstm_layers.append({
            'hidden_size': hidden_size,
            'input_size': input_size,
            'bidirectional': is_bidirectional
        })
        
        layer_idx += 1
    
    # Extract hidden dimensions
    hidden_dims = [layer['hidden_size'] for layer in lstm_layers]
    
    # Check bidirectionality (should be same for all layers)
    bidirectional = lstm_layers[0]['bidirectional'] if lstm_layers else False
    
    # Get output dimension from output layer
    if 'output_layer.weight' in state_dict:
        output_weight = state_dict['output_layer.weight']
        output_dim = output_weight.shape[0]
        final_hidden_dim = output_weight.shape[1]
    else:
        output_dim = 1
        final_hidden_dim = hidden_dims[-1] * (2 if bidirectional else 1)
    
    print(f"Detected LSTM architecture from state dict:")
    print(f"  Hidden dimensions: {hidden_dims}")
    print(f"  Bidirectional: {bidirectional}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Final hidden dimension: {final_hidden_dim}")
    
    return {
        'hidden_dims': hidden_dims,
        'bidirectional': bidirectional,
        'output_dim': output_dim,
        'num_layers': len(hidden_dims)
    }


@step(enable_cache=False)  # Disable cache since we're computing new baseline hazards
def compute_all_baseline_hazards(
    train_df_preprocessed: pd.DataFrame,
    models_dir: str = "results/final_deploy/models",
    model_config_dir: str = "results/final_deploy/model_config",
    output_dir: str = "results/final_deploy/models"
) -> Dict[str, Any]:
    """
    Compute baseline hazards for all DeepSurv models.
    
    Args:
        train_df_preprocessed: Preprocessed training data
        models_dir: Directory containing model weight files
        model_config_dir: Directory containing model configuration files
        output_dir: Directory to save baseline hazard files
        
    Returns:
        Dictionary with computation results and summary
    """
    from src.nn_architectures import create_network
    from src.sequence_utils_row_based import create_sequences_for_all_rows
    
    print(f"\n=== Computing Baseline Hazards for All DeepSurv Models ===")
    print(f"Models directory: {models_dir}")
    print(f"Config directory: {model_config_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get feature columns
    feature_cols = [
        'gender', 'creatinine', 'hemoglobin', 'phosphate',
        'age_at_obs', 'bicarbonate', 'albumin',
        'uacr', 'cci_score_total', 'ht', 'observation_period'
    ]
    
    # Initialize results tracking
    results = {
        'total_count': 24,  # Total DeepSurv models
        'successful_count': 0,
        'failed_count': 0,
        'successful_models': [],
        'failed_models': [],
        'baseline_hazard_files': []
    }
    
    # Pre-compute sequences with max length (10) for all LSTM models
    print("\n=== Pre-computing sequences for LSTM models ===")
    print("Creating sequences with length 10 for all rows (will slice as needed)...")
    
    # Create sequences once with max length 10
    max_sequence_length = 10
    sequence_cache_dir = os.path.join(output_dir, "sequence_cache")
    os.makedirs(sequence_cache_dir, exist_ok=True)
    
    # Pre-compute feature sequences ONCE (features don't change with event type)
    cache_file = os.path.join(sequence_cache_dir, f"sequences_features_len{max_sequence_length}.npz")
    
    if os.path.exists(cache_file):
        print(f"\nLoading pre-computed feature sequences from cache...")
        cached_data = np.load(cache_file)
        print(f"Loaded {len(cached_data['row_indices'])} sequences from {cache_file}")
    else:
        print(f"\nComputing feature sequences...")
        # Use target_endpoint=None to get all rows without filtering
        X_seq, _, _, row_indices = create_sequences_for_all_rows(
            df=train_df_preprocessed,
            sequence_length=max_sequence_length,
            feature_cols=feature_cols,
            target_endpoint=None,  # Don't filter by event - we'll handle that separately
            duration_col='duration',
            event_col='endpoint'
        )
        
        # Save to disk
        print(f"Saving sequences to {cache_file}...")
        np.savez_compressed(
            cache_file,
            sequences=X_seq,
            row_indices=row_indices
        )
        print(f"Saved {len(row_indices)} feature sequences")
    
    print("\nSequence pre-computation/loading complete!")
    
    # Process each DeepSurv model (models 1-24)
    for model_no in range(1, 25):
        print(f"\n{'='*60}")
        print(f"Processing Model {model_no}/24")
        
        try:
            # Find model weight file
            model_pattern = f"Ensemble_model{model_no}_DeepSurv_*.pt"
            model_files = glob.glob(os.path.join(models_dir, model_pattern))
            
            if not model_files:
                raise FileNotFoundError(f"No model file found for pattern: {model_pattern}")
            
            model_path = model_files[0]
            model_filename = os.path.basename(model_path)
            print(f"Model file: {model_filename}")
            
            # Extract model info from filename
            # Format: Ensemble_model{N}_DeepSurv_{Structure}_Event_{E}_{timestamp}.pt
            parts = model_filename.split('_')
            structure = parts[3]  # ANN or LSTM
            event_num = int(parts[5])  # 1 or 2
            timestamp = parts[6].replace('.pt', '')
            
            print(f"Structure: {structure}, Event: {event_num}")
            
            # Find and load model configuration
            config_pattern = f"model{model_no}_details_*.json"
            config_files = glob.glob(os.path.join(model_config_dir, config_pattern))
            
            if not config_files:
                raise FileNotFoundError(f"No config file found for pattern: {config_pattern}")
            
            config_path = config_files[0]
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            print(f"Config file: {os.path.basename(config_path)}")
            
            # Create network architecture based on structure
            if structure.upper() == 'ANN':
                # Create ANN network
                net = create_network(
                    model_type='deepsurv',
                    network_type='ann',
                    input_dim=model_config.get('input_dim', 11),
                    hidden_dims=model_config.get('hidden_dims', [256, 128, 64]),
                    output_dim=model_config.get('output_dim', 1),
                    dropout=model_config.get('dropout', 0.2)
                )
                
                # Prepare standard features for ANN
                X_train = train_df_preprocessed[feature_cols].values.astype('float32')
                
            elif structure.upper() == 'LSTM':
                # Get sequence length from config (this should be correct)
                sequence_length = model_config.get('sequence_length', 10)
                
                # Load state dict first to detect actual architecture
                print("Loading model state dict to detect architecture...")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                state_dict = torch.load(model_path, map_location=device)
                
                # Detect actual LSTM architecture from state dict
                detected_arch = detect_lstm_architecture_from_state_dict(state_dict)
                
                # Use detected architecture instead of config
                lstm_hidden_dims = detected_arch['hidden_dims']
                bidirectional = detected_arch['bidirectional']
                
                print(f"Using detected LSTM architecture: sequence_length={sequence_length}, "
                      f"hidden_dims={lstm_hidden_dims}, bidirectional={bidirectional}")
                
                # Create LSTM network with detected architecture
                net = create_network(
                    model_type='deepsurv',
                    network_type='lstm',
                    input_dim=model_config.get('input_dim', 11),
                    lstm_hidden_dim=lstm_hidden_dims,
                    lstm_num_layers=len(lstm_hidden_dims),
                    bidirectional=bidirectional,
                    sequence_length=sequence_length,
                    output_dim=detected_arch['output_dim'],
                    dropout=model_config.get('dropout', 0.2)
                )
                
                # Load pre-computed feature sequences from disk and slice to required length
                print(f"Loading pre-computed feature sequences and slicing to length {sequence_length}...")
                
                # Load cached feature sequences
                cache_file = os.path.join(sequence_cache_dir, f"sequences_features_len{max_sequence_length}.npz")
                cached_data = np.load(cache_file)
                full_sequences = cached_data['sequences']
                row_indices = cached_data['row_indices']
                
                # Slice sequences to required length
                # full_sequences shape: (n_rows, 10, n_features)
                # We want the most recent sequence_length timesteps
                if sequence_length < max_sequence_length:
                    # Slice from the end (most recent observations)
                    X_train = full_sequences[:, -sequence_length:, :]
                else:
                    # Use full sequences
                    X_train = full_sequences
                
                print(f"Using {len(row_indices)} pre-computed sequences, sliced to length {sequence_length}")
                print(f"Sliced sequences shape: {X_train.shape}")
                
            else:
                raise ValueError(f"Unknown structure: {structure}")
            
            # IMPORTANT: For baseline hazard computation, always use the full training data
            # durations and events, not the sequence-aggregated ones
            # This ensures we get baseline hazards covering the full time range
            y_event = (train_df_preprocessed['endpoint'] == event_num).values.astype('float32')
            y_duration = train_df_preprocessed['duration'].values.astype('float32')
            
            print(f"\nUsing full training data for baseline hazard computation:")
            print(f"Total rows: {len(y_duration)}")
            
            print(f"Training data shape: {X_train.shape}")
            print(f"Events: {np.sum(y_event)} out of {len(y_event)} ({np.mean(y_event)*100:.1f}%)")
            print(f"Duration range: {np.min(y_duration):.0f} - {np.max(y_duration):.0f} days")
            
            # Create optimizer (required for CoxPH but won't be used)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            
            # Create CoxPH model
            model = CoxPH(net, optimizer=optimizer)
            
            # Load model weights
            print("Loading model weights...")
            if structure.upper() == 'ANN':
                # For ANN models, load state dict here
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                state_dict = torch.load(model_path, map_location=device)
            # For LSTM models, state_dict is already loaded above
            
            model.net.load_state_dict(state_dict)
            model.net.to(device)
            model.net.eval()
            
            # Convert data to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            
            # Compute baseline hazards
            print("Computing baseline hazards...")
            _ = model.compute_baseline_hazards(
                input=X_train_tensor,
                target=(y_duration, y_event),
                batch_size=256,
                set_hazards=True
            )
            
            print("Baseline hazards computed successfully")
            print(f"Baseline hazards shape: {model.baseline_hazards_.shape}")
            print(f"Baseline cumulative hazards shape: {model.baseline_cumulative_hazards_.shape}")
            
            # Save baseline hazards
            baseline_hazards_data = {
                'baseline_hazards_': model.baseline_hazards_,
                'baseline_cumulative_hazards_': model.baseline_cumulative_hazards_,
                'model_config': model_config,
                'model_info': {
                    'model_no': model_no,
                    'structure': structure,
                    'event': event_num,
                    'timestamp': timestamp,
                    'model_path': model_path
                }
            }
            
            # Save to file
            output_filename = f"baseline_hazards_model{model_no}_{timestamp}.pkl"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'wb') as f:
                pickle.dump(baseline_hazards_data, f)
            
            print(f"Saved baseline hazards to: {output_path}")
            
            # Update results
            results['successful_count'] += 1
            results['successful_models'].append(model_no)
            results['baseline_hazard_files'].append(output_path)
            
            # Clear GPU memory if using CUDA
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nERROR processing model {model_no}: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            
            results['failed_count'] += 1
            results['failed_models'].append({
                'model_no': model_no,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            # Continue with next model
            continue
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("=== Baseline Hazard Computation Summary ===")
    print(f"Total models: {results['total_count']}")
    print(f"Successful: {results['successful_count']}")
    print(f"Failed: {results['failed_count']}")
    
    if results['successful_count'] > 0:
        print(f"\nSuccessfully processed models: {results['successful_models']}")
    
    if results['failed_count'] > 0:
        print(f"\nFailed models:")
        for failure in results['failed_models']:
            print(f"  - Model {failure['model_no']}: {failure['error']}")
    
    # Save summary report
    summary_path = os.path.join(output_dir, f"baseline_hazards_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nSummary report saved to: {summary_path}")
    
    return results