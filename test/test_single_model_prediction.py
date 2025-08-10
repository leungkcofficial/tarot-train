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
    
    # Check columns
    print(f"Columns: {temporal_test.columns.tolist()}")
    
    return temporal_test

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

def main():
    """Test with a single model"""
    
    # Load the model information
    with open('test_models.json', 'r') as f:
        models = json.load(f)
    
    model_info = models[0]
    print(f"Testing with model: {model_info['model_name']}")
    print(f"Model type: {model_info['model_type']}")
    print(f"Model endpoint: {model_info.get('model_endpoint', 'N/A')}")
    
    # Load test data
    temporal_test = load_test_data()
    
    # Get feature columns
    feature_cols = get_feature_columns()
    
    # Prepare data for ANN model
    latest_obs = temporal_test.sort_values(['key', 'date']).groupby('key').last().reset_index()
    print(f"\nLatest observations shape: {latest_obs.shape}")
    
    # Check if required columns exist
    print(f"\nChecking columns:")
    print(f"Has 'duration': {'duration' in latest_obs.columns}")
    print(f"Has 'endpoint': {'endpoint' in latest_obs.columns}")
    
    # Get features
    X = latest_obs[feature_cols].values.astype('float32')
    durations = latest_obs['duration'].values.astype('float32')
    events = latest_obs['endpoint'].values.astype('float32')
    
    # For DeepSurv models, filter events to the specific endpoint
    target_endpoint = model_info.get('model_endpoint')
    if target_endpoint is not None:
        print(f"\nFiltering for endpoint {target_endpoint}")
        # Create binary event indicator for the specific endpoint
        events = (events == target_endpoint).astype('float32')
        print(f"Event rate for endpoint {target_endpoint}: {events.mean():.2%}")
    
    print(f"\nData shapes:")
    print(f"X: {X.shape}")
    print(f"durations: {durations.shape}")
    print(f"events: {events.shape}")
    
    # Load model
    model_name = model_info['model_name']
    model_pattern = f"{model_name}_*.pt"
    model_files = list(Path('results/final_deploy/models').glob(model_pattern))
    
    if not model_files:
        print(f"Model file not found for pattern: {model_pattern}")
        return
    
    model_path = model_files[0]
    print(f"\nFound model file: {model_path}")
    
    # Load state dict
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    state_dict = torch.load(model_path, map_location=device)
    
    # Detect architecture
    hidden_dims = detect_mlp_architecture(state_dict)
    print(f"Detected MLP architecture: hidden_dims={hidden_dims}")
    
    # Create network
    input_dim = model_info.get('input_dim', 11)
    output_dim = model_info.get('output_dim', 1)
    dropout = model_info.get('hyperparameters', {}).get('dropout', 0.2)
    
    net = MLP(
        in_features=input_dim,
        hidden_dims=hidden_dims,
        out_features=output_dim,
        dropout=dropout,
        batch_norm=True
    )
    
    # Create optimizer (dummy, won't be used for inference)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # Create CoxPH model
    model = CoxPH(net, optimizer=optimizer)
    
    # Load the state dict
    model.net.load_state_dict(state_dict)
    model.net.to(device)
    model.net.eval()
    
    # Load baseline hazards
    baseline_data = load_baseline_hazards(model_name)
    if baseline_data:
        model.baseline_hazards_ = baseline_data['baseline_hazards_']
        model.baseline_cumulative_hazards_ = baseline_data['baseline_cumulative_hazards_']
        print("Baseline hazards loaded successfully")
    else:
        print("ERROR: Could not load baseline hazards!")
        return
    
    # Generate predictions
    print("\nGenerating predictions...")
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        # DeepSurv predictions
        surv_df = model.predict_surv_df(X_tensor)
        # Convert to CIF (1 - survival)
        cif = 1 - surv_df.values
    
    print(f"Predictions shape: {cif.shape}")
    print(f"Prediction range: [{cif.min():.4f}, {cif.max():.4f}]")
    print(f"Mean prediction: {cif.mean():.4f}")
    
    # Save predictions
    output_dir = Path('results/test_predictions_regenerated')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'test_predictions_{model_name}_{timestamp}.h5'
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('predictions', data=cif)
        f.attrs['model_name'] = model_name
        f.attrs['model_type'] = model_info['model_type']
        f.attrs['n_samples'] = cif.shape[-1]
        f.attrs['regenerated'] = True
        f.attrs['include_all_patients'] = True
    
    print(f"\nSaved predictions to: {output_file}")
    print("Test completed successfully!")

if __name__ == "__main__":
    main()