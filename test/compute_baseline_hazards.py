import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from pycox.models import CoxPH
from src.nn_architectures import create_network
import pickle

def load_training_data():
    """Load training dataset from ZenML artifacts"""
    print("Loading training dataset from ZenML artifacts...")
    
    # Load training data
    train_path = '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/preprocess_data/output_0/8c34de25-a3e5-4873-a415-ff854097c480/d5c7e5f5/df.parquet.gzip'
    train_df = pd.read_parquet(train_path)
    print(f"Training data shape: {train_df.shape}")
    print(f"Training unique patients: {train_df['key'].nunique()}")
    
    return train_df

def get_feature_columns():
    """Get the feature columns used for modeling from YAML mapping"""
    feature_cols = [
        'gender', 'creatinine', 'hemoglobin', 'phosphate',
        'age_at_obs', 'bicarbonate', 'albumin',
        'uacr', 'cci_score_total', 'ht', 'observation_period'
    ]
    return feature_cols

def prepare_data_for_event(df, feature_cols, event_num):
    """Prepare data for a specific event"""
    # Filter for the specific event
    event_col = f'event{event_num}'
    duration_col = f'duration{event_num}'
    
    # Get features
    X = df[feature_cols].values.astype('float32')
    
    # Get target variables
    y_event = df[event_col].values.astype('float32')
    y_duration = df[duration_col].values.astype('float32')
    
    return X, y_event, y_duration

def load_deepsurv_model(model_path, model_info, device='cpu'):
    """Load a DeepSurv model with the correct architecture"""
    # Get model details
    input_dim = model_info.get('input_dim', 11)
    output_dim = model_info.get('output_dim', 1)
    dropout = model_info.get('hyperparameters', {}).get('dropout', 0.2)
    
    # Determine network type
    model_name = model_info.get('model_name', '')
    if 'LSTM' in model_name:
        network_type = 'lstm'
    elif 'ANN' in model_name:
        network_type = 'ann'
    else:
        # Default to ANN
        network_type = 'ann'
    
    print(f"Network type: {network_type}")
    
    # Create network
    if network_type == 'ann':
        # For ANN models, use default hidden dimensions
        net = create_network(
            model_type='deepsurv',
            network_type='ann',
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
            hidden_dims=[256, 128, 64]  # Default architecture
        )
    else:
        # For LSTM models, we need to detect architecture from state dict
        state_dict = torch.load(model_path, map_location=device)
        
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
        
        print(f"Detected LSTM architecture: layers={len(lstm_hidden_dims)}, dims={lstm_hidden_dims}, bidirectional={bidirectional}")
        
        net = create_network(
            model_type='deepsurv',
            network_type='lstm',
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
            lstm_num_layers=len(lstm_hidden_dims),
            bidirectional=bidirectional,
            sequence_length=5,  # Default sequence length
            lstm_hidden_dim=lstm_hidden_dims[0] if lstm_hidden_dims else 64
        )
    
    # Create optimizer (dummy, won't be used for inference)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # Create CoxPH model
    model = CoxPH(net, optimizer=optimizer)
    
    # Load the state dict
    if network_type == 'lstm':
        model.net.load_state_dict(state_dict)
    else:
        model.net.load_state_dict(torch.load(model_path, map_location=device))
    
    model.net.to(device)
    model.net.eval()
    
    return model, network_type

def compute_and_save_baseline_hazards():
    """Compute baseline hazards for all DeepSurv models"""
    
    # Load training data
    train_df = load_training_data()
    feature_cols = get_feature_columns()
    
    # Get all DeepSurv model configurations
    model_details_dir = Path('results/model_details')
    deepsurv_configs = []
    
    for json_file in model_details_dir.glob('deployed_model_details_*.json'):
        with open(json_file, 'r') as f:
            config = json.load(f)
            if config.get('model_type', '').lower() == 'deepsurv':
                deepsurv_configs.append(config)
    
    print(f"\nFound {len(deepsurv_configs)} DeepSurv models")
    
    # Create directory for baseline hazards
    baseline_dir = Path('results/baseline_hazards')
    baseline_dir.mkdir(exist_ok=True)
    
    # Process each model
    for i, model_info in enumerate(deepsurv_configs):
        model_name = model_info.get('model_name', 'unknown')
        timestamp = model_info.get('timestamp', 'unknown')
        model_path = model_info.get('deployed_model_weights_path')
        event_num = model_info.get('model_endpoint', 1)
        
        print(f"\n{'='*80}")
        print(f"Processing model {i+1}/{len(deepsurv_configs)}: {model_name}")
        print(f"Timestamp: {timestamp}")
        print(f"Event: {event_num}")
        print(f"Model path: {model_path}")
        
        try:
            # Load the model
            model, network_type = load_deepsurv_model(model_path, model_info)
            
            # Prepare training data for this event
            X_train, y_event, y_duration = prepare_data_for_event(
                train_df, feature_cols, event_num
            )
            
            # For LSTM models, we need to prepare sequences
            if network_type == 'lstm':
                print("Preparing sequences for LSTM model...")
                from src.sequence_utils_fixed import create_sequences_from_dataframe_fixed
                
                # Create sequences
                X_seq, y_seq, seq_keys = create_sequences_from_dataframe_fixed(
                    train_df,
                    feature_cols,
                    sequence_length=5,
                    include_all_patients=False  # For training, we don't need all patients
                )
                
                # Get event and duration for sequences
                seq_df = train_df[train_df['key'].isin(seq_keys)].drop_duplicates('key')
                y_event = seq_df[f'event{event_num}'].values.astype('float32')
                y_duration = seq_df[f'duration{event_num}'].values.astype('float32')
                
                # Convert to torch tensors
                X_train_tensor = torch.FloatTensor(X_seq)
                
                print(f"Sequence shape: {X_seq.shape}")
                print(f"Number of sequences: {len(seq_keys)}")
            else:
                # For ANN models, use regular features
                X_train_tensor = torch.FloatTensor(X_train)
            
            # Compute baseline hazards
            print("Computing baseline hazards...")
            _ = model.compute_baseline_hazards(
                input=X_train_tensor,
                target=(y_duration, y_event),
                batch_size=256,
                set_hazards=True
            )
            
            # Save baseline hazards
            baseline_hazards = {
                'baseline_hazards_': model.baseline_hazards_,
                'baseline_cumulative_hazards_': model.baseline_cumulative_hazards_
            }
            
            # Save to file
            baseline_path = baseline_dir / f'baseline_hazards_{timestamp}.pkl'
            with open(baseline_path, 'wb') as f:
                pickle.dump(baseline_hazards, f)
            
            print(f"Saved baseline hazards to: {baseline_path}")
            
            # Also save as numpy arrays for inspection
            np_path = baseline_dir / f'baseline_hazards_{timestamp}.npz'
            np.savez(
                np_path,
                baseline_hazards=model.baseline_hazards_,
                baseline_cumulative_hazards=model.baseline_cumulative_hazards_
            )
            
            print(f"Also saved as numpy arrays to: {np_path}")
            
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Completed computing baseline hazards for {len(deepsurv_configs)} models")
    print(f"Baseline hazards saved to: {baseline_dir}")

if __name__ == "__main__":
    compute_and_save_baseline_hazards()