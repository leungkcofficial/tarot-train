import json
import torch
import numpy as np
import pandas as pd
import h5py
import pickle
from pathlib import Path
from datetime import datetime
from src.sequence_utils_fixed import create_sequences_from_dataframe_fixed
from src.nn_architectures import create_network
from pycox.models import CoxPH, DeepHitSingle

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

def load_model_with_flexible_architecture(
    model_path: str,
    model_type: str,
    network_type: str,
    model_details: dict,
    device: str = 'cpu'
):
    """Load a model with flexible architecture matching the saved weights."""
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
    
    # Create optimizer (dummy, won't be used for inference)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # Create model
    if model_type.lower() == "deepsurv":
        model = CoxPH(net, optimizer=optimizer)
    else:  # deephit
        alpha = model_details.get('alpha', 0.2)
        sigma = model_details.get('sigma', 0.1)
        time_grid = np.array(model_details.get('time_grid', [365, 730, 1095, 1460, 1825]))
        model = DeepHitSingle(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
    
    # Load the state dict
    model.net.load_state_dict(state_dict)
    model.net.to(device)
    model.net.eval()
    
    # For DeepSurv models, load baseline hazards if available
    if model_type.lower() == "deepsurv":
        # Extract model number from model name (e.g., "Ensemble_model14_DeepSurv_LSTM_Event_1" -> "model14")
        model_name = model_details.get('model_name', '')
        if model_name:
            # Extract model number
            import re
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
                        model.baseline_hazards_ = baseline_data['baseline_hazards_']
                        model.baseline_cumulative_hazards_ = baseline_data['baseline_cumulative_hazards_']
                    print("Baseline hazards loaded successfully")
                else:
                    print(f"Warning: Baseline hazards not found for model{model_num}")
            else:
                print(f"Warning: Could not extract model number from {model_name}")
    
    return model


def regenerate_lstm_predictions(model_info, temporal_test, spatial_test, feature_cols):
    """Regenerate predictions for a single LSTM model with proper padding"""
    
    model_name = model_info['model_name']
    model_type = model_info['model_type']
    model_path = model_info['original_model_path']
    
    print(f"\n{'='*80}")
    print(f"Processing: {model_name}")
    print(f"Model type: {model_type}")
    print(f"Model path: {model_path}")
    
    # Get model details
    model_details = model_info.get('model_details', {})
    if not model_details and 'input_dim' in model_info:
        # Build model_details from model_info
        model_details = {
            'input_dim': model_info.get('input_dim', 11),
            'output_dim': model_info.get('output_dim', 1),
            'dropout': model_info.get('hyperparameters', {}).get('dropout', 0.2),
            'sequence_length': model_info.get('sequence_length', 5),
            'lstm_hidden_dims': model_info.get('hidden_dims', [64, 32]),
            'lstm_num_layers': len(model_info.get('hidden_dims', [64, 32])),
            'bidirectional': True,  # Default for LSTM models
            'alpha': 0.2,
            'sigma': 0.1,
            'time_grid': model_info.get('time_grid', [365, 730, 1095, 1460, 1825]),
            'timestamp': model_info.get('timestamp', ''),  # Add timestamp for baseline hazards
            'model_name': model_info.get('model_name', '')  # Add model name for baseline hazards lookup
        }
    
    # Get sequence length
    sequence_length = model_details.get('sequence_length', 5)
    print(f"Sequence length: {sequence_length}")
    
    # Load the model
    print("Loading model...")
    model = load_model_with_flexible_architecture(
        model_path=model_path,
        model_type=model_type,
        network_type='lstm',
        model_details=model_details,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Process temporal test set
    print("\nProcessing temporal test set...")
    temporal_sequences, temporal_durations, temporal_events, temporal_patient_ids = create_sequences_from_dataframe_fixed(
        temporal_test,
        sequence_length=sequence_length,
        feature_cols=feature_cols,
        cluster_col='key',
        date_col='date',
        duration_col='duration',
        event_col='endpoint',
        target_endpoint=model_info.get('model_endpoint') if model_type == 'deepsurv' else None,
        include_all_patients=True  # This ensures all patients are included
    )
    
    print(f"Temporal sequences shape: {temporal_sequences.shape}")
    print(f"Expected shape: ({temporal_test['key'].nunique()}, {sequence_length}, {len(feature_cols)})")
    
    # Process spatial test set
    print("\nProcessing spatial test set...")
    spatial_sequences, spatial_durations, spatial_events, spatial_patient_ids = create_sequences_from_dataframe_fixed(
        spatial_test,
        sequence_length=sequence_length,
        feature_cols=feature_cols,
        cluster_col='key',
        date_col='date',
        duration_col='duration',
        event_col='endpoint',
        target_endpoint=model_info.get('model_endpoint') if model_type == 'deepsurv' else None,
        include_all_patients=True  # This ensures all patients are included
    )
    
    print(f"Spatial sequences shape: {spatial_sequences.shape}")
    print(f"Expected shape: ({spatial_test['key'].nunique()}, {sequence_length}, {len(feature_cols)})")
    
    # Generate predictions
    print("\nGenerating predictions...")
    
    if model_type == 'deepsurv':
        # DeepSurv predictions
        temporal_preds = generate_deepsurv_predictions(
            model, temporal_sequences, temporal_durations, temporal_events
        )
        spatial_preds = generate_deepsurv_predictions(
            model, spatial_sequences, spatial_durations, spatial_events
        )
        
        print(f"Temporal predictions shape: {temporal_preds.shape}")
        print(f"Spatial predictions shape: {spatial_preds.shape}")
        
    else:  # deephit
        # DeepHit predictions
        temporal_preds, temporal_cause1, temporal_cause2 = generate_deephit_predictions(
            model, temporal_sequences, temporal_durations, temporal_events
        )
        spatial_preds, spatial_cause1, spatial_cause2 = generate_deephit_predictions(
            model, spatial_sequences, spatial_durations, spatial_events
        )
        
        print(f"Temporal predictions shape: {temporal_preds.shape}")
        print(f"Spatial predictions shape: {spatial_preds.shape}")
    
    # Save predictions
    save_predictions(model_info, temporal_preds, spatial_preds, 
                    temporal_cause1 if model_type == 'deephit' else None,
                    temporal_cause2 if model_type == 'deephit' else None,
                    spatial_cause1 if model_type == 'deephit' else None,
                    spatial_cause2 if model_type == 'deephit' else None)
    
    return True

def generate_deepsurv_predictions(model, sequences, durations, events):
    """Generate predictions for DeepSurv model"""
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(sequences)
    if torch.cuda.is_available():
        X_tensor = X_tensor.cuda()
    
    # Generate predictions
    with torch.no_grad():
        # DeepSurv models output survival function
        surv_df = model.predict_surv_df(X_tensor)
        
        # Convert to CIF (1 - survival)
        cif = 1 - surv_df.values
        
    return cif

def generate_deephit_predictions(model, sequences, durations, events):
    """Generate predictions for DeepHit model"""
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(sequences)
    if torch.cuda.is_available():
        X_tensor = X_tensor.cuda()
    
    # Generate predictions
    with torch.no_grad():
        # DeepHit outputs CIF directly
        # predict_cif returns shape (n_samples, n_times) for each event
        cif = model.predict_cif(X_tensor)
        
        # For DeepHitSingle with competing risks, we need to handle both events
        # The model should have 2 outputs (one for each event)
        if isinstance(cif, tuple):
            # If it returns a tuple of arrays
            cause1_preds = cif[0]  # Shape: (n_samples, n_times)
            cause2_preds = cif[1]  # Shape: (n_samples, n_times)
            
            # Transpose to (n_times, n_samples)
            cause1_preds = cause1_preds.T
            cause2_preds = cause2_preds.T
            
            # Stack to create full CIF array
            cif = np.stack([cause1_preds, cause2_preds], axis=0)  # Shape: (2, n_times, n_samples)
        else:
            # If it returns a single array with shape (n_samples, n_events * n_times)
            n_samples = cif.shape[0]
            n_times = 5  # Default time points
            n_events = 2  # Two competing events
            
            # Reshape and transpose
            cif = cif.reshape(n_samples, n_events, n_times)  # (n_samples, n_events, n_times)
            cif = cif.transpose(1, 2, 0)  # (n_events, n_times, n_samples)
            
            # Extract individual cause predictions
            cause1_preds = cif[0]  # Shape: (n_times, n_samples)
            cause2_preds = cif[1]  # Shape: (n_times, n_samples)
        
    return cif, cause1_preds, cause2_preds

def save_predictions(model_info, temporal_preds, spatial_preds, 
                    temporal_cause1=None, temporal_cause2=None,
                    spatial_cause1=None, spatial_cause2=None):
    """Save predictions to H5 files"""
    
    # Create output directory
    output_dir = Path('results/test_predictions_fixed')
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
    
    # Update model info with new paths
    model_info['temporal_test_predictions_path_fixed'] = str(temporal_file)
    model_info['spatial_test_predictions_path_fixed'] = str(spatial_file)

def main():
    """Main function to regenerate all LSTM predictions"""
    
    # Load the model information
    json_file = '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_0/67c9050f-777c-40ad-b087-a4bd53d6c47a/99a24532/data.json'
    
    print("Loading model information...")
    with open(json_file, 'r') as f:
        all_models = json.load(f)
    
    # Filter LSTM models only
    lstm_models = []
    for model in all_models:
        if 'LSTM' in model['model_name']:
            lstm_models.append(model)
    
    print(f"\nFound {len(lstm_models)} LSTM models to process:")
    for model in lstm_models:
        print(f"  - {model['model_name']} ({model['model_type']})")
    
    # Load test data
    temporal_test, spatial_test = load_test_data()
    
    # Get feature columns
    feature_cols = get_feature_columns()
    
    # Process each LSTM model
    successful = 0
    failed = 0
    
    for i, model_info in enumerate(lstm_models):
        try:
            print(f"\n\nProcessing model {i+1}/{len(lstm_models)}")
            success = regenerate_lstm_predictions(
                model_info, temporal_test, spatial_test, feature_cols
            )
            if success:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"ERROR processing {model_info['model_name']}: {str(e)}")
            failed += 1
    
    # Save updated model information
    output_file = 'lstm_models_with_fixed_predictions.json'
    with open(output_file, 'w') as f:
        json.dump(lstm_models, f, indent=2)
    
    print(f"\n\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  - Total LSTM models: {len(lstm_models)}")
    print(f"  - Successfully processed: {successful}")
    print(f"  - Failed: {failed}")
    print(f"  - Updated model info saved to: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()