import json
import torch
import numpy as np
import pandas as pd
import h5py
import pickle
from pathlib import Path
from datetime import datetime
from src.sequence_utils_fixed import create_sequences_from_dataframe_fixed
from src.nn_architectures import MLP, CNNMLP, LSTMSurvival, LSTMDeepHit
from pycox.models import CoxPH, DeepHitSingle
import re

def load_test_data():
    """Load temporal and spatial test datasets from ZenML artifacts"""
    print("Loading test datasets from ZenML artifacts...")
    
    # Load temporal test data
    temporal_test_path = '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/preprocess_data/output_1/8c34de25-a3e5-4873-a415-ff854097c480/ce7f1fa4/df.parquet.gzip'
    temporal_test = pd.read_parquet(temporal_test_path)
    print(f"Temporal test shape: {temporal_test.shape}")
    print(f"Temporal test unique patients: {temporal_test['key'].nunique()}")
    
    # Load spatial test data
    spatial_test_path = '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/preprocess_data/output_2/8c34de25-a3e5-4873-a415-ff854097c480/f254d971/df.parquet.gzip'
    spatial_test = pd.read_parquet(spatial_test_path)
    print(f"Spatial test shape: {spatial_test.shape}")
    print(f"Spatial test unique patients: {spatial_test['key'].nunique()}")
    
    return temporal_test, spatial_test

def get_feature_columns():
    """Get the feature columns used for modeling from YAML mapping"""
    feature_cols = [
        'gender', 'creatinine', 'hemoglobin', 'phosphate',
        'age_at_obs', 'bicarbonate', 'albumin',
        'uacr', 'cci_score_total', 'ht', 'observation_period'
    ]
    return feature_cols

def load_baseline_hazards(model_name):
    """Load baseline hazards for a DeepSurv model"""
    # Extract model number from model name
    match = re.search(r'model(\d+)', model_name)
    if match:
        model_num = match.group(1)
        # Look for baseline hazards file
        baseline_pattern = f'baseline_hazards_model{model_num}_*.pkl'
        baseline_files = list(Path('results/final_deploy/models').glob(baseline_pattern))
        
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

def load_model_for_prediction(model_info, device='cpu'):
    """Load a model with the correct architecture for prediction"""
    model_name = model_info['model_name']
    model_type = model_info['model_type']
    
    # Find the model file with timestamp
    model_pattern = f"{model_name}_*.pt"
    model_files = list(Path('results/final_deploy/models').glob(model_pattern))
    
    if not model_files:
        print(f"Model file not found for pattern: {model_pattern}")
        return None, None
    
    # Take the first match (should be only one)
    model_path = model_files[0]
    print(f"Found model file: {model_path}")
    
    # Determine network type
    if 'LSTM' in model_name:
        network_type = 'lstm'
    elif 'ANN' in model_name:
        network_type = 'ann'
    else:
        network_type = 'ann'
    
    # Get model parameters
    input_dim = model_info.get('input_dim', 11)
    output_dim = model_info.get('output_dim', 1)
    dropout = model_info.get('hyperparameters', {}).get('dropout', 0.2)
    
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
                    sequence_length=5,
                    lstm_hidden_dims=lstm_hidden_dims,
                    output_dim=output_dim,
                    dropout=dropout,
                    bidirectional=bidirectional
                )
            else:  # DeepHit
                net = LSTMDeepHit(
                    input_dim=input_dim,
                    sequence_length=5,
                    lstm_hidden_dims=lstm_hidden_dims,
                    num_causes=2,
                    num_durations=5,
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
                # For DeepHit, we need to create the shared network and cause-specific networks
                # Create shared network
                shared_net = MLP(
                    in_features=input_dim,
                    hidden_dims=hidden_dims[:-1] if len(hidden_dims) > 1 else [],
                    out_features=hidden_dims[-1] if hidden_dims else 128,
                    dropout=dropout,
                    batch_norm=True
                )
                net = CNNMLP(
                    shared_net=shared_net,
                    num_events=2,
                    num_durations=5,
                    hidden_dims=hidden_dims,
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
            model = DeepHitSingle(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
        
        # Load the state dict
        model.net.load_state_dict(state_dict)
        model.net.to(device)
        model.net.eval()
        
        # For DeepSurv models, load baseline hazards
        if model_type.lower() == "deepsurv":
            baseline_data = load_baseline_hazards(model_name)
            if baseline_data:
                model.baseline_hazards_ = baseline_data['baseline_hazards_']
                model.baseline_cumulative_hazards_ = baseline_data['baseline_cumulative_hazards_']
                print("Baseline hazards loaded successfully")
            else:
                print("ERROR: Could not load baseline hazards for DeepSurv model!")
                return None, network_type
        
        return model, network_type
        
    except Exception as e:
        print(f"Error creating model architecture: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, network_type

def prepare_data_for_model(df, feature_cols, network_type, sequence_length=5, 
                          target_endpoint=None, include_all_patients=True):
    """Prepare data for model prediction"""
    if network_type == 'lstm':
        # Create sequences for LSTM
        sequences, durations, events, patient_ids = create_sequences_from_dataframe_fixed(
            df,
            sequence_length=sequence_length,
            feature_cols=feature_cols,
            cluster_col='key',
            date_col='date',
            duration_col='duration',
            event_col='endpoint',
            target_endpoint=target_endpoint,
            include_all_patients=include_all_patients
        )
        return sequences, durations, events, patient_ids
    else:
        # For ANN models, use the latest observation for each patient
        latest_obs = df.sort_values(['key', 'date']).groupby('key').last().reset_index()
        X = latest_obs[feature_cols].values.astype('float32')
        
        # Always use 'duration' and 'endpoint' columns
        durations = latest_obs['duration'].values.astype('float32')
        events = latest_obs['endpoint'].values.astype('float32')
        
        # For DeepSurv models, filter events to the specific endpoint
        if target_endpoint is not None:
            # Create binary event indicator for the specific endpoint
            events = (events == target_endpoint).astype('float32')
        
        patient_ids = latest_obs['key'].values
        
        return X, durations, events, patient_ids

def generate_predictions(model, model_type, X, durations, events, device='cpu'):
    """Generate predictions for a model"""
    # Convert to tensor
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        if model_type.lower() == 'deepsurv':
            # DeepSurv predictions
            surv_df = model.predict_surv_df(X_tensor)
            # Convert to CIF (1 - survival)
            cif = 1 - surv_df.values
            return cif, None, None
        else:
            # DeepHit predictions
            surv = model.predict_surv(X_tensor)
            
            # Handle different output formats
            if isinstance(surv, tuple) and len(surv) == 2:
                # Returns (cause1_surv, cause2_surv)
                cause1_surv, cause2_surv = surv
                cause1_cif = 1 - cause1_surv
                cause2_cif = 1 - cause2_surv
                
                # Stack to create full CIF array
                cif = np.stack([cause1_cif, cause2_cif], axis=0)  # Shape: (2, n_times, n_samples)
            else:
                # Single array output
                cif = 1 - surv
                n_samples = cif.shape[0]
                n_times = 5  # Default time points
                n_events = 2  # Two competing events
                
                # Reshape and transpose
                cif = cif.reshape(n_samples, n_events, n_times)  # (n_samples, n_events, n_times)
                cif = cif.transpose(1, 2, 0)  # (n_events, n_times, n_samples)
                
                cause1_cif = cif[0]  # Shape: (n_times, n_samples)
                cause2_cif = cif[1]  # Shape: (n_times, n_samples)
            
            return cif, cause1_cif, cause2_cif

def save_predictions(model_info, temporal_preds, spatial_preds, 
                    temporal_cause1=None, temporal_cause2=None,
                    spatial_cause1=None, spatial_cause2=None):
    """Save predictions to H5 files"""
    
    # Create output directory
    output_dir = Path('results/test_predictions_regenerated')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = model_info['model_name']
    
    # Save temporal predictions
    temporal_file = output_dir / f'temporal_test_predictions_{model_name}_{timestamp}.h5'
    with h5py.File(temporal_file, 'w') as f:
        f.create_dataset('predictions', data=temporal_preds)
        if temporal_cause1 is not None:
            f.create_dataset('cause_1_predictions', data=temporal_cause1)
            f.create_dataset('cause_2_predictions', data=temporal_cause2)
        
        # Save metadata
        f.attrs['model_name'] = model_name
        f.attrs['model_type'] = model_info['model_type']
        f.attrs['n_samples'] = temporal_preds.shape[-1]
        f.attrs['regenerated'] = True
        f.attrs['include_all_patients'] = True
    
    print(f"Saved temporal predictions to: {temporal_file}")
    
    # Save spatial predictions
    spatial_file = output_dir / f'spatial_test_predictions_{model_name}_{timestamp}.h5'
    with h5py.File(spatial_file, 'w') as f:
        f.create_dataset('predictions', data=spatial_preds)
        if spatial_cause1 is not None:
            f.create_dataset('cause_1_predictions', data=spatial_cause1)
            f.create_dataset('cause_2_predictions', data=spatial_cause2)
        
        # Save metadata
        f.attrs['model_name'] = model_name
        f.attrs['model_type'] = model_info['model_type']
        f.attrs['n_samples'] = spatial_preds.shape[-1]
        f.attrs['regenerated'] = True
        f.attrs['include_all_patients'] = True
    
    print(f"Saved spatial predictions to: {spatial_file}")
    
    return str(temporal_file), str(spatial_file)

def main():
    """Main function to regenerate all predictions with baseline hazards"""
    
    # Load the model information
    json_file = '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_0/67c9050f-777c-40ad-b087-a4bd53d6c47a/99a24532/data.json'
    
    print("Loading model information...")
    with open(json_file, 'r') as f:
        all_models = json.load(f)
    
    print(f"\nFound {len(all_models)} models to process")
    
    # Load test data
    temporal_test, spatial_test = load_test_data()
    
    # Get feature columns
    feature_cols = get_feature_columns()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Process each model
    successful = 0
    failed = 0
    results = []
    
    for i, model_info in enumerate(all_models):
        model_name = model_info['model_name']
        model_type = model_info['model_type']
        
        print(f"\n{'='*80}")
        print(f"Processing model {i+1}/{len(all_models)}: {model_name}")
        print(f"Model type: {model_type}")
        
        try:
            # Load the model
            model, network_type = load_model_for_prediction(model_info, device)
            if model is None:
                print(f"Failed to load model: {model_name}")
                failed += 1
                continue
            
            # Get target endpoint for DeepSurv models
            target_endpoint = model_info.get('model_endpoint') if model_type.lower() == 'deepsurv' else None
            
            # Prepare temporal test data
            print("\nPreparing temporal test data...")
            temporal_X, temporal_durations, temporal_events, temporal_ids = prepare_data_for_model(
                temporal_test, feature_cols, network_type, 
                target_endpoint=target_endpoint, include_all_patients=True
            )
            print(f"Temporal data shape: {temporal_X.shape}")
            print(f"Number of temporal patients: {len(temporal_ids)}")
            
            # Prepare spatial test data
            print("\nPreparing spatial test data...")
            spatial_X, spatial_durations, spatial_events, spatial_ids = prepare_data_for_model(
                spatial_test, feature_cols, network_type,
                target_endpoint=target_endpoint, include_all_patients=True
            )
            print(f"Spatial data shape: {spatial_X.shape}")
            print(f"Number of spatial patients: {len(spatial_ids)}")
            
            # Generate predictions
            print("\nGenerating predictions...")
            temporal_preds, temporal_c1, temporal_c2 = generate_predictions(
                model, model_type, temporal_X, temporal_durations, temporal_events, device
            )
            spatial_preds, spatial_c1, spatial_c2 = generate_predictions(
                model, model_type, spatial_X, spatial_durations, spatial_events, device
            )
            
            print(f"Temporal predictions shape: {temporal_preds.shape}")
            print(f"Spatial predictions shape: {spatial_preds.shape}")
            
            # Save predictions
            temporal_path, spatial_path = save_predictions(
                model_info, temporal_preds, spatial_preds,
                temporal_c1, temporal_c2, spatial_c1, spatial_c2
            )
            
            # Store results
            results.append({
                'model_name': model_name,
                'model_type': model_type,
                'network_type': network_type,
                'temporal_predictions_path': temporal_path,
                'spatial_predictions_path': spatial_path,
                'temporal_n_samples': len(temporal_ids),
                'spatial_n_samples': len(spatial_ids)
            })
            
            successful += 1
            print(f"Successfully processed {model_name}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Save summary
    output_dir = Path('results/test_predictions_regenerated')
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / 'regeneration_summary.json'
    
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_models': len(all_models),
            'successful': successful,
            'failed': failed,
            'results': results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Regeneration complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()