import json
import torch
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from datetime import datetime
from src.sequence_utils_fixed import create_sequences_from_dataframe_fixed
from src.nn_architectures import create_network
from pycox.models import CoxPH, DeepHitSingle

def test_single_lstm_model():
    """Test regenerating predictions for a single LSTM model"""
    
    # Load the model information
    json_file = '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_0/67c9050f-777c-40ad-b087-a4bd53d6c47a/99a24532/data.json'
    
    print("Loading model information...")
    with open(json_file, 'r') as f:
        all_models = json.load(f)
    
    # Find the first LSTM DeepSurv model
    lstm_model = None
    for model in all_models:
        if 'LSTM' in model['model_name'] and model['model_type'] == 'deepsurv':
            lstm_model = model
            break
    
    if not lstm_model:
        print("No LSTM DeepSurv model found!")
        return
    
    print(f"\nTesting with model: {lstm_model['model_name']}")
    print(f"Model path: {lstm_model['original_model_path']}")
    
    # Load test data from ZenML artifacts
    print("\nLoading temporal test data from ZenML artifact...")
    # Option 1: Load from ZenML client
    # from zenml.client import Client
    # artifact = Client().get_artifact_version("0597f229-a376-436f-9568-86606ebbef46")
    # temporal_test = artifact.load()
    
    # Option 2: Load directly from parquet file
    temporal_test_path = '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/preprocess_data/output_1/8c34de25-a3e5-4873-a415-ff854097c480/ce7f1fa4/df.parquet.gzip'
    temporal_test = pd.read_parquet(temporal_test_path)
    print(f"Temporal test shape: {temporal_test.shape}")
    print(f"Unique patients: {temporal_test['key'].nunique()}")
    
    # Get feature columns from the YAML mapping
    feature_cols = [
        'gender', 'creatinine', 'hemoglobin', 'phosphate',
        'age_at_obs', 'bicarbonate', 'albumin',
        'uacr', 'cci_score_total', 'ht', 'observation_period'
    ]
    
    # Get sequence length
    sequence_length = lstm_model.get('sequence_length', 5)
    if 'optimization_details' in lstm_model:
        opt_details = lstm_model['optimization_details']
        if 'best_params' in opt_details and 'sequence' in opt_details['best_params']:
            sequence_length = opt_details['best_params']['sequence']
    
    print(f"\nSequence length: {sequence_length}")
    
    # Create sequences with padding for all patients
    print("\nCreating sequences with padding for all patients...")
    sequences, durations, events, patient_ids = create_sequences_from_dataframe_fixed(
        temporal_test,
        sequence_length=sequence_length,
        feature_cols=feature_cols,
        cluster_col='key',
        date_col='date',
        duration_col='duration',
        event_col='endpoint',
        target_endpoint=lstm_model.get('model_endpoint', 1),
        include_all_patients=True  # This is the key parameter
    )
    
    print(f"\nSequences shape: {sequences.shape}")
    print(f"Expected shape: ({temporal_test['key'].nunique()}, {sequence_length}, {len(feature_cols)})")
    
    # Load the model
    print("\nLoading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load state dict
    state_dict = torch.load(lstm_model['original_model_path'], map_location=device)
    
    # Infer LSTM architecture
    lstm_hidden_dims = []
    for key in state_dict.keys():
        if 'lstm_layers' in key and 'weight_ih_l0' in key:
            layer_idx = int(key.split('.')[1])
            weight_shape = state_dict[key].shape
            hidden_dim = weight_shape[0] // 4  # LSTM has 4 gates
            lstm_hidden_dims.append(hidden_dim)
    
    bidirectional = any('reverse' in key for key in state_dict.keys())
    
    print(f"Detected LSTM architecture: dims={lstm_hidden_dims}, bidirectional={bidirectional}")
    
    # Create network
    net = create_network(
        model_type='deepsurv',
        network_type='lstm',
        input_dim=len(feature_cols),
        output_dim=1,
        dropout=lstm_model.get('hyperparameters', {}).get('dropout', 0.2),
        lstm_num_layers=len(lstm_hidden_dims),
        bidirectional=bidirectional,
        sequence_length=sequence_length,
        lstm_hidden_dim=lstm_hidden_dims[0] if lstm_hidden_dims else 64
    )
    
    # Create model
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    model = CoxPH(net, optimizer=optimizer)
    
    # Load weights
    model.net.load_state_dict(state_dict)
    model.net.to(device)
    model.net.eval()
    
    print("\nModel loaded successfully!")
    
    # Generate predictions
    print("\nGenerating predictions...")
    X_tensor = torch.FloatTensor(sequences)
    if device == 'cuda':
        X_tensor = X_tensor.cuda()
    
    with torch.no_grad():
        # DeepSurv models output survival function
        surv_df = model.predict_surv_df(X_tensor)
        
        # Convert to CIF (1 - survival)
        cif = 1 - surv_df.values
    
    print(f"\nPredictions shape: {cif.shape}")
    print(f"Time points: {surv_df.index.tolist()[:10]}...")  # Show first 10 time points
    
    # Save test predictions
    output_dir = Path('results/test_predictions_fixed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'test_{lstm_model["model_name"]}_{timestamp}.h5'
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('predictions', data=cif)
        f.create_dataset('durations', data=durations)
        f.create_dataset('events', data=events)
        
        # Save metadata
        f.attrs['model_name'] = lstm_model['model_name']
        f.attrs['n_samples'] = cif.shape[1]
        f.attrs['n_time_points'] = cif.shape[0]
        f.attrs['sequence_length'] = sequence_length
        f.attrs['include_all_patients'] = True
    
    print(f"\nTest predictions saved to: {output_file}")
    print(f"Successfully generated predictions for all {cif.shape[1]} patients!")

if __name__ == "__main__":
    test_single_lstm_model()