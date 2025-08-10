"""
Test script to verify baseline hazard computation for a single model.

This script tests the baseline hazard computation process with one ANN model
and one LSTM model to ensure everything works correctly before running the full pipeline.
"""

import os
import sys
import json
import glob
import torch
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nn_architectures import create_network
from src.sequence_utils import create_sequences_from_dataframe
from pycox.models import CoxPH


def test_single_model(model_no: int, train_df_path: str = None):
    """Test baseline hazard computation for a single model."""
    
    print(f"\n{'='*60}")
    print(f"Testing Model {model_no}")
    print(f"{'='*60}")
    
    # Directories
    models_dir = "results/final_deploy/models"
    config_dir = "results/final_deploy/model_config"
    
    # Get feature columns
    feature_cols = [
        'gender', 'creatinine', 'hemoglobin', 'phosphate',
        'age_at_obs', 'bicarbonate', 'albumin',
        'uacr', 'cci_score_total', 'ht', 'observation_period'
    ]
    
    try:
        # Load training data (you'll need to provide the path to preprocessed training data)
        if train_df_path:
            print(f"Loading training data from: {train_df_path}")
            train_df = pd.read_parquet(train_df_path)
        else:
            # Try to find the most recent preprocessed training data
            print("Looking for preprocessed training data...")
            # This is a placeholder - you'll need to provide the actual path
            raise ValueError("Please provide the path to preprocessed training data")
        
        print(f"Training data shape: {train_df.shape}")
        
        # Find model file
        model_pattern = f"Ensemble_model{model_no}_DeepSurv_*.pt"
        model_files = glob.glob(os.path.join(models_dir, model_pattern))
        
        if not model_files:
            raise FileNotFoundError(f"No model file found for pattern: {model_pattern}")
        
        model_path = model_files[0]
        model_filename = os.path.basename(model_path)
        print(f"Model file: {model_filename}")
        
        # Extract model info
        parts = model_filename.split('_')
        structure = parts[3]  # ANN or LSTM
        event_num = int(parts[5])  # 1 or 2
        timestamp = parts[6].replace('.pt', '')
        
        print(f"Structure: {structure}, Event: {event_num}")
        
        # Load model configuration
        config_pattern = f"model{model_no}_details_*.json"
        config_files = glob.glob(os.path.join(config_dir, config_pattern))
        
        if not config_files:
            raise FileNotFoundError(f"No config file found for pattern: {config_pattern}")
        
        config_path = config_files[0]
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        print(f"Config file: {os.path.basename(config_path)}")
        print(f"Model config: {json.dumps(model_config, indent=2)}")
        
        # Create network
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
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
            
            # Prepare features
            X_train = train_df[feature_cols].values.astype('float32')
            train_df_for_targets = train_df
            
        elif structure.upper() == 'LSTM':
            # Get LSTM parameters
            sequence_length = model_config.get('sequence_length', 10)
            lstm_hidden_dims = model_config.get('lstm_hidden_dims', [64, 32])
            lstm_num_layers = model_config.get('lstm_num_layers', 2)
            bidirectional = model_config.get('lstm_bidirectional', True)
            
            print(f"LSTM config: sequence_length={sequence_length}, "
                  f"hidden_dims={lstm_hidden_dims}, bidirectional={bidirectional}")
            
            # Create LSTM network
            net = create_network(
                model_type='deepsurv',
                network_type='lstm',
                input_dim=model_config.get('input_dim', 11),
                lstm_hidden_dim=lstm_hidden_dims,
                lstm_num_layers=lstm_num_layers,
                bidirectional=bidirectional,
                sequence_length=sequence_length,
                output_dim=model_config.get('output_dim', 1),
                dropout=model_config.get('dropout', 0.2)
            )
            
            # Create sequences
            print(f"Creating sequences with length {sequence_length}...")
            X_seq, y_seq, seq_keys = create_sequences_from_dataframe(
                train_df,
                feature_cols,
                sequence_length=sequence_length,
                include_all_patients=False
            )
            
            # Get event and duration for sequences
            seq_df = train_df[train_df['key'].isin(seq_keys)].drop_duplicates('key')
            X_train = X_seq
            train_df_for_targets = seq_df
            
            print(f"Created {len(seq_keys)} sequences")
        
        # Get target variables
        y_event = train_df_for_targets[f'event{event_num}'].values.astype('float32')
        y_duration = train_df_for_targets[f'duration{event_num}'].values.astype('float32')
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Events: {np.sum(y_event)} out of {len(y_event)} ({np.mean(y_event)*100:.1f}%)")
        print(f"Duration range: {np.min(y_duration):.0f} - {np.max(y_duration):.0f} days")
        
        # Create CoxPH model
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        model = CoxPH(net, optimizer=optimizer)
        
        # Load weights
        print("Loading model weights...")
        state_dict = torch.load(model_path, map_location=device)
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
        
        print("Baseline hazards computed successfully!")
        print(f"Baseline hazards shape: {model.baseline_hazards_.shape}")
        print(f"Baseline cumulative hazards shape: {model.baseline_cumulative_hazards_.shape}")
        
        # Save test results
        test_output_dir = "test_baseline_hazards"
        os.makedirs(test_output_dir, exist_ok=True)
        
        baseline_hazards_data = {
            'baseline_hazards_': model.baseline_hazards_,
            'baseline_cumulative_hazards_': model.baseline_cumulative_hazards_,
            'model_config': model_config,
            'model_info': {
                'model_no': model_no,
                'structure': structure,
                'event': event_num,
                'timestamp': timestamp
            }
        }
        
        output_path = os.path.join(test_output_dir, f"test_baseline_hazards_model{model_no}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(baseline_hazards_data, f)
        
        print(f"\nTest baseline hazards saved to: {output_path}")
        
        # Verify the saved file can be loaded
        print("\nVerifying saved file...")
        with open(output_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        print("File loaded successfully!")
        print(f"Loaded baseline hazards shape: {loaded_data['baseline_hazards_'].shape}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test baseline hazard computation with one ANN and one LSTM model."""
    
    print("BASELINE HAZARD COMPUTATION TEST")
    print("="*60)
    
    # You need to provide the path to preprocessed training data
    # This should be the output from the preprocess_data step
    train_df_path = None  # Update this with actual path
    
    # Example paths where preprocessed data might be stored:
    # train_df_path = "/home/goma/.config/zenml/local_stores/.../preprocess_data/output_0/.../df.parquet.gzip"
    
    if not train_df_path:
        print("\nPlease provide the path to preprocessed training data.")
        print("You can find this in the ZenML artifact store after running the ensemble pipeline.")
        print("\nAlternatively, you can run a minimal data processing pipeline first.")
        return
    
    # Test Model 1 (ANN, Event 1)
    print("\nTesting ANN model...")
    ann_success = test_single_model(1, train_df_path)
    
    # Test Model 13 (LSTM, Event 1)
    print("\nTesting LSTM model...")
    lstm_success = test_single_model(13, train_df_path)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"ANN model test: {'PASSED' if ann_success else 'FAILED'}")
    print(f"LSTM model test: {'PASSED' if lstm_success else 'FAILED'}")
    
    if ann_success and lstm_success:
        print("\nAll tests passed! You can now run the full pipeline.")
    else:
        print("\nSome tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()