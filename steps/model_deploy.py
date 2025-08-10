"""
Model Deployment Step for CKD Risk Prediction

This module contains the ZenML step for deploying the optimized CKD risk prediction model.
It only generates predictions on test datasets without performing evaluation.
"""

import pandas as pd
import numpy as np
import h5py
import mlflow
import torch
import os
import json
import gc
from zenml.steps import step
from zenml.client import Client
from typing import Dict, Any, Optional, Union
from datetime import datetime

experiment_tracker = Client().active_stack.experiment_tracker
@step(enable_cache=True, experiment_tracker=experiment_tracker.name)
def deploy_model(
    model_metadata: Union[Dict[str, Any], str],
    optimization_metrics: Optional[Union[Dict[str, Any], str]] = None,
    model_name: str = "CKD_Model",
    model_stage: str = "Development",
    register_model: bool = True,
    master_df_mapping_path: str = "src/default_master_df_mapping.yml",
    model_endpoint: Optional[int] = None,
    train_df_preprocessed: Optional[pd.DataFrame] = None,
    temporal_test_df_preprocessed: Optional[pd.DataFrame] = None,
    spatial_test_df_preprocessed: Optional[pd.DataFrame] = None,
    batch_size: int = 1000,
    cv_folds: int = 10
) -> Dict[str, Any]:
    
    """ Deploy the optimized CKD risk prediction model and generate predictions.

    This function creates a new neural network and survival model using the loaded data,
    trains it with train_df_preprocessed using time-based cross-validation with patient grouping,
    selects the best-performing model based on the metric criteria from hyperparameter_config.yml,
    and uses this model to make predictions on the test datasets.
    
    The function will generate predictions for the test datasets and save them to HDF5 files,
    which can be used directly in the evaluation step without recomputing predictions.

    Args:
        model_metadata: Dictionary containing model metadata from train_model step or path to a JSON file containing model metadata
        optimization_metrics: Optional dictionary containing optimization metrics or path to a JSON file containing optimization metrics
        model_name: Name of the model for registration (default: "CKD_Model")
        model_stage: Stage of the model (Development, Staging, Production) (default: "Development")
        register_model: Whether to register the model with MLflow (default: True)
        master_df_mapping_path: Path to the master dataframe mapping file (default: "src/default_master_df_mapping.yml")
        model_endpoint: Endpoint number for deployment (default: None, uses the one from model_metadata)
        train_df_preprocessed: Preprocessed training dataframe for training the best model (default: None)
        temporal_test_df_preprocessed: Preprocessed temporal test dataframe for generating predictions (default: None)
        spatial_test_df_preprocessed: Preprocessed spatial test dataframe for generating predictions (default: None)
        batch_size: Batch size for processing large datasets (default: 1000)
        cv_folds: Number of folds for time-based cross-validation (default: 10)
        
    Returns:
        Dict[str, Any]: Deployment details including model type, endpoint, prediction paths, and model paths.
    """
    import json
    import torch
    import os
    import gc
    import numpy as np
    from datetime import datetime
    from typing import Union, List, Tuple
    from src.util import load_yaml_file, save_predictions_to_hdf5
    from src.nn_architectures import create_network
    from pycox.models import CoxPH, DeepHit
    from pycox.evaluation import EvalSurv
    from torchtuples.callbacks import EarlyStopping
    
    # SciPy compatibility fix for PyCox
    import scipy.integrate
    try:
        from scipy.integrate import simps
    except ImportError:
        # In newer SciPy versions, simps has been renamed to simpson
        from scipy.integrate import simpson
        scipy.integrate.simps = simpson
    from steps.model_train import prepare_survival_dataset
    from steps.cv_utils import time_based_patient_cv, create_cv_datasets
    from src.balance_data import balance_dataframe

    print("\n=== Deploying CKD Risk Prediction Model ===\n")
    
    # Check if model_metadata is a path to a JSON file
    if isinstance(model_metadata, str):
        print(f"Loading model metadata from JSON file: {model_metadata}")
        try:
            with open(model_metadata, 'r') as f:
                model_metadata = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load model metadata from {model_metadata}: {e}")
    
    # Check if optimization_metrics is a path to a JSON file
    if isinstance(optimization_metrics, str):
        print(f"Loading optimization metrics from JSON file: {optimization_metrics}")
        try:
            with open(optimization_metrics, 'r') as f:
                optimization_metrics = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load optimization metrics from {optimization_metrics}: {e}")
            optimization_metrics = None
    
    # Load hyperparameter configuration once at the beginning
    hp_config = load_yaml_file("src/hyperparameter_config.yml")
    optimization_metric = hp_config.get('optimization', {}).get('metric', 'cidx')
    print(f"Loaded hyperparameter configuration with optimization metric: {optimization_metric}")
    
    # 1. Extract model details from metadata
    model_type = model_metadata.get('model_type', 'deepsurv')
    network_type = model_metadata.get('network_type', 'ann')  # Default to ANN if not specified
    input_dim = model_metadata.get('input_dim')
    hidden_dims = model_metadata.get('hidden_dims')
    output_dim = model_metadata.get('output_dim')
    dropout = model_metadata.get('dropout')
    time_grid = model_metadata.get('time_grid')
    
    # For DeepHit models, load time_grid from hyperparameter config if not in model_metadata
    if model_type.lower() == 'deephit' and time_grid is None:
        time_grid = hp_config.get('network', {}).get('deephit', {}).get('time_grid')
        if time_grid is not None:
            print(f"Loaded time_grid from hyperparameter config: {time_grid}")
        else:
            print("Warning: No time_grid found in hyperparameter config for DeepHit model")
    
    # Ensure time_grid is a NumPy array for use with integrated_nbll and other functions
    if time_grid is not None and not isinstance(time_grid, np.ndarray):
        time_grid = np.array(time_grid)
        print(f"Converted time_grid to NumPy array with shape: {time_grid.shape}")
    alpha = model_metadata.get('alpha')
    sigma = model_metadata.get('sigma')
    model_path = model_metadata.get('model_path')
    
    # Extract LSTM-specific parameters if network type is LSTM
    if network_type.lower() == 'lstm':
        lstm_hidden_dims = model_metadata.get('lstm_hidden_dims', [])
        lstm_num_layers = model_metadata.get('lstm_num_layers', 1)
        lstm_bidirectional = model_metadata.get('lstm_bidirectional', False)
        sequence_length = model_metadata.get('sequence_length', 5)
        print(f"LSTM network detected: {lstm_num_layers} layers, hidden_dims={lstm_hidden_dims}, bidirectional={lstm_bidirectional}, sequence_length={sequence_length}")
    
    # Use model_endpoint from metadata if not provided
    if model_endpoint is None and model_type.lower() == 'deepsurv':
        # Load hyperparameter config to get the correct target_endpoint
        try:
            hp_config_path = 'src/hyperparameter_config.yml'
            if os.path.exists(hp_config_path):
                hp_config = load_yaml_file(hp_config_path)
                config_target_endpoint = hp_config.get('target_endpoint')
                print(f"[DEBUG] Loaded target_endpoint from config: {config_target_endpoint}")
                model_endpoint = config_target_endpoint if config_target_endpoint is not None else 1
            else:
                print(f"[DEBUG] Config file not found at {hp_config_path}, using default endpoint 1")
                model_endpoint = 1  # Default to endpoint 1 for DeepSurv if not specified
        except Exception as e:
            print(f"[DEBUG] Error loading config: {e}, using default endpoint 1")
            model_endpoint = 1
    
    print(f"[DEBUG] Final model_endpoint value: {model_endpoint}")
    
    # Create hyperparameters dictionary with default values
    hyperparameters = {
        'learning_rate': 0.001,  # Default value
        'dropout': dropout,
        'optimizer': 'Adam',  # Default value
        'batch_size': 64,  # Default value
    }
    
    # Update hyperparameters with optimized values if available
    if optimization_metrics and 'best_params' in optimization_metrics:
        best_params = optimization_metrics['best_params']
        print(f"Using optimized hyperparameters from optimization metrics")
        
        # Update hyperparameters with optimized values
        if 'learning_rate' in best_params:
            hyperparameters['learning_rate'] = best_params['learning_rate']
        
        if 'optimizer' in best_params:
            hyperparameters['optimizer'] = best_params['optimizer']
        
        if 'batch_size' in best_params:
            hyperparameters['batch_size'] = best_params['batch_size']
        
        # Dropout is already included from model_metadata, but we can update it from best_params if needed
        if 'dropout' in best_params:
            hyperparameters['dropout'] = best_params['dropout']
    
    # Add model-specific hyperparameters
    if model_type.lower() == 'deephit' and alpha is not None and sigma is not None:
        hyperparameters['alpha'] = alpha
        hyperparameters['sigma'] = sigma
    
    print(f"Model type: {model_type}")
    print(f"Model endpoint: {model_endpoint}")
    print(f"Model path: {model_path}")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimensions: {hidden_dims}")
    print(f"Output dimension: {output_dim}")
    print(f"Hyperparameters: {hyperparameters}")
    
    # Load master dataframe mapping
    master_df_mapping = load_yaml_file(master_df_mapping_path)
    feature_cols = master_df_mapping.get("features", [])
    
    print(f"Loaded master dataframe mapping with {len(feature_cols)} features")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if we have training data
    if train_df_preprocessed is None or train_df_preprocessed.empty:
        print("No training data provided. Using pre-trained model from model_metadata.")
        # 5. Create network
        if network_type.lower() == 'lstm':
            net = create_network(
                model_type=model_type,
                network_type=network_type,
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=dropout,
                lstm_num_layers=lstm_num_layers,
                bidirectional=lstm_bidirectional,
                sequence_length=sequence_length,
                lstm_hidden_dim=lstm_hidden_dims[0] if lstm_hidden_dims else 64
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
        
        # 6. Load model weights
        print(f"Loading model weights from {model_path}")
        
        # Create optimizer
        optimizer_name = hyperparameters.get('optimizer', 'Adam')
        lr = hyperparameters.get('learning_rate', 0.001)
        
        print(f"Using optimizer: {optimizer_name} with learning rate: {lr}")
        
        # Load the model weights
        print(f"Loading model weights from {model_path}")
        net.load_state_dict(torch.load(model_path))
        
        # Create optimizer (not needed for deployment, but required for model creation)
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        else:
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
        
        # Create model
        if model_type.lower() == "deepsurv":
            model = CoxPH(net, optimizer=optimizer)
        else:
            model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
        
        # Move model to device
        model.net = model.net.to(device)
    else:
        print("Training data provided. Training a new model with time-based cross-validation.")
        
        # Get balancing configuration from hyperparameter config
        balance_cfg = hp_config.get("balance", {})
        orig_rows = len(train_df_preprocessed)
        
        # Apply balancing if enabled
        if balance_cfg.get("enable", False):
            print("\n=== Applying class balancing to training data ===")
            
            print(f"Model type: {model_type}")
            print(f"Target endpoint: {model_endpoint}")
            
            # Print dataframe info to help with debugging
            print("\nDataFrame info before balancing:")
            print(f"DataFrame shape: {train_df_preprocessed.shape}")
            print(f"DataFrame columns: {train_df_preprocessed.columns.tolist()}")
            print(f"DataFrame dtypes sample:")
            for col, dtype in train_df_preprocessed.dtypes.items():
                if pd.api.types.is_datetime64_any_dtype(dtype):
                    print(f"  {col}: {dtype} (DATETIME)")
                else:
                    print(f"  {col}: {dtype}")
            
            # Apply balancing
            train_df_preprocessed = balance_dataframe(
                train_df_preprocessed,
                duration_col="duration",
                event_col="endpoint",
                method=balance_cfg.get("method", "random_under_sampler"),
                sampling_strategy=balance_cfg.get("sampling_strategy", "majority"),
                model_type=model_type,
                target_endpoint=model_endpoint,
                near_miss_version=balance_cfg.get("near_miss_version", 1),
                feature_cols=feature_cols,  # Pass the features from master dataframe mapping
            )
            
            # Log balancing metrics
            balanced_rows = len(train_df_preprocessed)
            removed_rows = orig_rows - balanced_rows
            removed_pct = (removed_rows / orig_rows) * 100
            
            print(f"Original rows: {orig_rows}")
            print(f"Balanced rows: {balanced_rows}")
            print(f"Removed rows: {removed_rows} ({removed_pct:.2f}%)")
        else:
            print("\n=== Class balancing is disabled ===")
            balanced_rows = orig_rows
            removed_pct = 0.0
            
        # Extract patient IDs (cluster column) and entry dates for time-based cross-validation
        # IMPORTANT: Extract these AFTER balancing to ensure indices match
        cluster_col = master_df_mapping.get("cluster", "key")
        print(f"Using '{cluster_col}' column as patient IDs for time-based cross-validation")
        
        # Check if the cluster column exists in the dataframe
        if cluster_col not in train_df_preprocessed.columns:
            print(f"Warning: '{cluster_col}' column not found in dataframe. Available columns: {train_df_preprocessed.columns[:10]}...")
            patient_ids = None
        else:
            patient_ids = train_df_preprocessed[cluster_col].values
            print(f"Extracted {len(patient_ids)} patient IDs")
        
        # Extract entry dates if available
        if 'date' in train_df_preprocessed.columns:
            entry_dates = train_df_preprocessed['date'].values
            print(f"Using 'date' column for entry dates in time-based cross-validation")
        else:
            print(f"Warning: 'date' column not found in dataframe. Time-based cross-validation may fall back to random splits.")
            entry_dates = None
        
        # Prepare training dataset with filtered features and target endpoint
        print("\n=== Preparing training dataset ===")
        print(f"Training dataframe shape before preparation: {train_df_preprocessed.shape}")
        print(f"Using target_endpoint: {model_endpoint}")
        print(f"Network type: {network_type}")
        
        if network_type.lower() == 'lstm':
            print("Preparing sequence data for LSTM...")
            print(f"Using sequence length: {sequence_length}")
            
            # Import LSTM preparation function
            from src.util import prepare_lstm_survival_dataset
            
            train_data = prepare_lstm_survival_dataset(
                train_df_preprocessed,
                feature_cols=feature_cols,
                sequence_length=sequence_length,
                target_endpoint=model_endpoint
            )
            
            # For LSTM, input_dim is the number of features per timestep
            input_dim = train_data[0].shape[2]  # Shape: (samples, sequence_length, features)
            print(f"Input dimension (features per timestep): {input_dim}")
            print(f"Sequence length: {sequence_length}")
        else:
            train_data = prepare_survival_dataset(
                train_df_preprocessed,
                feature_cols=feature_cols,
                target_endpoint=model_endpoint
            )
            
            # Set input dimension based on the actual number of features in the filtered data
            input_dim = train_data[0].shape[1]
            print(f"Input dimension (after filtering): {input_dim}")
        
        # Get optimization metric from hyperparameter config
        optimization_metric = hp_config.get('optimization', {}).get('metric', 'cidx')
        print(f"Using optimization metric: {optimization_metric}")
        
        # Debug the discretization condition
        print(f"[DEBUG] Checking discretization condition:")
        print(f"[DEBUG] model_type.lower() = '{model_type.lower()}'")
        print(f"[DEBUG] time_grid is not None = {time_grid is not None}")
        if time_grid is not None:
            print(f"[DEBUG] time_grid shape = {time_grid.shape}")
            print(f"[DEBUG] time_grid values = {time_grid}")
        
        # Pre-discretize data for DeepHit models BEFORE creating CV splits
        if model_type.lower() == 'deephit' and time_grid is not None:
            print(f"\n=== Pre-discretizing data for DeepHit model ===")
            
            # Create label transformer for DeepHit
            from pycox.preprocessing.label_transforms import LabTransDiscreteTime
            labtrans = LabTransDiscreteTime(time_grid)
            
            # Fit the transformer on the full training data
            labtrans.fit(train_data[1], train_data[2])
            print(f"Created LabTransDiscreteTime with {len(time_grid)} time points")
            
            # Ensure proper data types before transformation
            train_durations = np.asarray(train_data[1], dtype=np.float64)
            train_events_raw = np.asarray(train_data[2], dtype=np.float64)
            
            # Clip durations to be within time grid bounds
            time_grid_array = np.array(time_grid)
            time_grid_min, time_grid_max = time_grid_array.min(), time_grid_array.max()
            train_durations = np.clip(train_durations, time_grid_min, time_grid_max)
            
            print(f"Pre-transform - Duration bounds: Min: {time_grid_min}, Max: {time_grid_max}")
            print(f"Pre-transform - Data types: durations {train_durations.dtype}, events {train_events_raw.dtype}")
            
            # For DeepHit competing risks, only discretize durations, keep original events
            # LabTransDiscreteTime.transform() combines competing events, which we don't want
            train_idx_durations = labtrans.transform(train_durations, train_events_raw)[0]  # Only take discretized durations
            train_events = train_events_raw  # Keep original competing risk events (0, 1, 2)
            
            # CRITICAL FIX: Ensure discretized durations are within valid bounds [0, len(time_grid)-1]
            max_valid_idx = len(time_grid) - 1
            train_idx_durations = np.clip(train_idx_durations, 0, max_valid_idx)
            
            print(f"Post-transform bounds check:")
            print(f"  Time grid length: {len(time_grid)} (valid indices: 0-{max_valid_idx})")
            print(f"  Discretized duration range: {train_idx_durations.min()}-{train_idx_durations.max()}")
            print(f"  Unique discretized durations: {np.unique(train_idx_durations)}")
            
            # Convert to tensors
            train_idx_durations = torch.tensor(train_idx_durations).long()
            train_events = torch.tensor(train_events).long()
            
            print(f"Pre-transformed labels - idx_durations shape {train_idx_durations.shape}, events shape {train_events.shape}")
            
            # Show detailed discretization verification
            print("\n=== DISCRETIZATION VERIFICATION ===")
            print("Original continuous durations:")
            unique_orig_durations, orig_duration_counts = np.unique(train_durations, return_counts=True)
            print(f"- Number of unique original durations: {len(unique_orig_durations)}")
            print(f"- Original duration range: {unique_orig_durations.min():.1f} to {unique_orig_durations.max():.1f}")
            if len(unique_orig_durations) <= 20:
                for dur, count in zip(unique_orig_durations, orig_duration_counts):
                    print(f"  Duration {dur:.1f}: {count} samples")
            else:
                print(f"- First 10 durations: {unique_orig_durations[:10]}")
                print(f"- Last 10 durations: {unique_orig_durations[-10:]}")
            
            print("\nOriginal events:")
            unique_orig_events, orig_event_counts = np.unique(train_events_raw, return_counts=True)
            print(f"- Number of unique original events: {len(unique_orig_events)}")
            for event, count in zip(unique_orig_events, orig_event_counts):
                print(f"  Event {int(event)}: {count} samples ({count/len(train_events_raw)*100:.1f}%)")
            
            print("\nDiscretized durations (time bins):")
            unique_disc_durations, disc_duration_counts = np.unique(train_idx_durations.numpy(), return_counts=True)
            print(f"- Number of unique discretized duration bins: {len(unique_disc_durations)}")
            print(f"- Discretized duration bin range: {unique_disc_durations.min()} to {unique_disc_durations.max()}")
            if len(unique_disc_durations) <= 20:
                for bin_idx, count in zip(unique_disc_durations, disc_duration_counts):
                    if bin_idx < len(time_grid):
                        time_val = time_grid[bin_idx]
                        print(f"  Bin {bin_idx} (time {time_val:.1f}): {count} samples")
                    else:
                        print(f"  Bin {bin_idx} (beyond time_grid): {count} samples")
            else:
                print(f"- First 10 bins: {unique_disc_durations[:10]}")
                print(f"- Last 10 bins: {unique_disc_durations[-10:]}")
            
            print("\nDiscretized events:")
            unique_disc_events, disc_event_counts = np.unique(train_events.numpy(), return_counts=True)
            print(f"- Number of unique discretized events: {len(unique_disc_events)}")
            for event, count in zip(unique_disc_events, disc_event_counts):
                print(f"  Event {int(event)}: {count} samples ({count/len(train_events)*100:.1f}%)")
            
            print("=== END DISCRETIZATION VERIFICATION ===\n")
            
            # Update train_data with discretized labels for DeepHit
            train_data = (
                train_data[0],  # Features (unchanged)
                train_idx_durations.numpy(),  # Discrete durations
                train_events.numpy()  # Events (unchanged but converted)
            )
            
            # Store original continuous data for evaluation purposes
            original_continuous_data = (train_durations, train_events_raw)
            print(f"Stored original continuous data for evaluation: durations shape {train_durations.shape}, events shape {train_events_raw.shape}")
        else:
            # For non-DeepHit models, no discretization needed
            labtrans = None
            original_continuous_data = None
        
        # Create cross-validation splits
        print(f"\n=== Creating {cv_folds}-fold time-based cross-validation splits ===")
        
        # For LSTM, we need to use sequence data for CV splits, not original tabular data
        if network_type.lower() == 'lstm':
            print("LSTM detected: Using sequence data for cross-validation")
            from steps.cv_utils import create_random_cv_splits
            cv_splits = create_random_cv_splits(train_data, n_splits=cv_folds, seed=hp_config.get('optimization', {}).get('seed', 42))
        else:
            # Use time-based cross-validation if patient_ids and entry_dates are available
            if patient_ids is not None and entry_dates is not None:
                print("Using time-based patient cross-validation")
                cv_splits = time_based_patient_cv(
                    dataset=train_data,
                    patient_ids=patient_ids,
                    entry_dates=entry_dates,
                    cv_folds=cv_folds,
                    seed=hp_config.get('optimization', {}).get('seed', 42)
                )
            else:
                print("Warning: Patient IDs or entry dates not available. Falling back to random cross-validation.")
                from steps.cv_utils import create_random_cv_splits
                cv_splits = create_random_cv_splits(train_data, n_splits=cv_folds, seed=hp_config.get('optimization', {}).get('seed', 42))
            
        print(f"Created {len(cv_splits)} CV splits")
        
        # Initialize list to store models and their performance
        models = []
        model_performances = []
        
        # Train models for each fold
        for fold, (train_indices, val_indices) in enumerate(cv_splits):
            print(f"\n--- Fold {fold+1}/{len(cv_splits)} ---")
            print(f"[DEBUG] CV Split sizes - Train: {len(train_indices)}, Validation: {len(val_indices)}")
            print(f"[DEBUG] Train/Val ratio: {len(train_indices)/len(val_indices):.2f}")
            
            # Create training and validation datasets for this fold
            fold_train_data, fold_val_data = create_cv_datasets(train_data, train_indices, val_indices)
            
            # Create network
            if network_type.lower() == 'lstm':
                net = create_network(
                    model_type=model_type,
                    network_type=network_type,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    dropout=dropout,
                    lstm_num_layers=lstm_num_layers,
                    bidirectional=lstm_bidirectional,
                    sequence_length=sequence_length,
                    lstm_hidden_dim=lstm_hidden_dims[0] if lstm_hidden_dims else 64
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
            optimizer_name = hyperparameters.get('optimizer', 'Adam')
            lr = hyperparameters.get('learning_rate', 0.001)
            
            if optimizer_name == "Adam":
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            else:
                optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
            
            # Create model
            if model_type.lower() == "deepsurv":
                fold_model = CoxPH(net, optimizer=optimizer)
            else:
                fold_model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
            
            # Move model to device
            fold_model.net = fold_model.net.to(device)
            
            # Convert numpy arrays to torch tensors and move to device
            x_train = torch.tensor(fold_train_data[0]).float().to(device)
            durations_train = torch.tensor(fold_train_data[1]).float()
            events_train = torch.tensor(fold_train_data[2]).float()
            
            x_val = torch.tensor(fold_val_data[0]).float().to(device)
            durations_val = torch.tensor(fold_val_data[1]).float()
            events_val = torch.tensor(fold_val_data[2]).float()
            
            # Create PyTorch datasets and dataloaders
            if model_type.lower() == "deepsurv":
                train_tuple = (x_train, (durations_train, events_train))
                val_tuple = (x_val, (durations_val, events_val))
            else:
                # For DeepHit, use the pre-discretized data that was created before CV splits
                train_idx_durations = torch.tensor(fold_train_data[1]).long()
                train_events = torch.tensor(fold_train_data[2]).long()
                val_idx_durations = torch.tensor(fold_val_data[1]).long()
                val_events = torch.tensor(fold_val_data[2]).long()
                
                train_tuple = (x_train, (train_idx_durations, train_events))
                val_tuple = (x_val, (val_idx_durations, val_events))
                
                print(f"Using pre-discretized data for DeepHit training")
                
                # Show unique values being fed into training for verification
                print(f"\n=== TRAINING DATA VERIFICATION FOR FOLD {fold+1} ===")
                
                # Training data unique values
                unique_train_durations, train_dur_counts = np.unique(train_idx_durations.numpy(), return_counts=True)
                unique_train_events, train_event_counts = np.unique(train_events.numpy(), return_counts=True)
                
                print(f"Training durations (discrete bins):")
                print(f"- Unique duration bins: {unique_train_durations}")
                print(f"- Duration bin counts: {train_dur_counts}")
                for bin_idx, count in zip(unique_train_durations, train_dur_counts):
                    if bin_idx < len(time_grid):
                        time_val = time_grid[bin_idx]
                        print(f"  Bin {bin_idx} (time {time_val:.0f}): {count} samples")
                    else:
                        print(f"  Bin {bin_idx} (beyond time_grid): {count} samples")
                
                print(f"\nTraining events:")
                print(f"- Unique events: {unique_train_events}")
                print(f"- Event counts: {train_event_counts}")
                for event, count in zip(unique_train_events, train_event_counts):
                    event_name = "Censored" if event == 0 else f"Event {event}"
                    print(f"  {event_name} ({event}): {count} samples ({count/len(train_events)*100:.1f}%)")
                
                # Validation data unique values
                unique_val_durations, val_dur_counts = np.unique(val_idx_durations.numpy(), return_counts=True)
                unique_val_events, val_event_counts = np.unique(val_events.numpy(), return_counts=True)
                
                print(f"\nValidation durations (discrete bins):")
                print(f"- Unique duration bins: {unique_val_durations}")
                print(f"- Duration bin counts: {val_dur_counts}")
                for bin_idx, count in zip(unique_val_durations, val_dur_counts):
                    if bin_idx < len(time_grid):
                        time_val = time_grid[bin_idx]
                        print(f"  Bin {bin_idx} (time {time_val:.0f}): {count} samples")
                    else:
                        print(f"  Bin {bin_idx} (beyond time_grid): {count} samples")
                
                print(f"\nValidation events:")
                print(f"- Unique events: {unique_val_events}")
                print(f"- Event counts: {val_event_counts}")
                for event, count in zip(unique_val_events, val_event_counts):
                    event_name = "Censored" if event == 0 else f"Event {event}"
                    print(f"  {event_name} ({event}): {count} samples ({count/len(val_events)*100:.1f}%)")
                
                print(f"=== END TRAINING DATA VERIFICATION ===\n")
            
            # Create early stopping callback
            callbacks = [EarlyStopping(patience=hp_config.get('optimization', {}).get('patience', 10))]
            
            # Train the model
            batch_size = hyperparameters.get('batch_size', 64)
            print(f"Training with batch size: {batch_size}")
            
            fold_model.fit(
                *train_tuple,  # Unpack the tuple to provide input and target separately
                batch_size=batch_size,
                epochs=100,  # Use a reasonable number of epochs with early stopping
                callbacks=callbacks,
                val_data=val_tuple,
                verbose=True
            )
            
            # Compute baseline hazards for CoxPH model if needed
            if model_type.lower() == "deepsurv":
                print("\n=== Computing baseline hazards for CoxPH model ===")
                # Check if any values are None
                if fold_train_data[1] is None or fold_train_data[2] is None:
                    print(f"Warning: durations or events are None. durations: {fold_train_data[1] is None}, events: {fold_train_data[2] is None}")
                    # Skip computing baseline hazards and use model directly
                    print("Skipping baseline hazards computation due to None values")
                else:
                    # Use training data to compute baseline hazards
                    print(f"Training data summary for baseline hazards computation:")
                    print(f"- Number of training samples: {len(fold_train_data[1])}")
                    print(f"- Number of training events: {int(np.sum(fold_train_data[2]))}")
                    print(f"- Training event rate: {np.mean(fold_train_data[2]):.2%}")
                    print(f"- Training duration range: {np.min(fold_train_data[1]):.1f} to {np.max(fold_train_data[1]):.1f} days")
                    print(f"- Training data quality checks:")
                    print(f"  - NaN in durations: {np.isnan(fold_train_data[1]).any()}")
                    print(f"  - NaN in events: {np.isnan(fold_train_data[2]).any()}")
                    print(f"  - Zero durations: {(fold_train_data[1] == 0).sum()}")
                    print(f"  - Negative durations: {(fold_train_data[1] < 0).sum()}")
                    
                    # Compute baseline hazards
                    fold_model.compute_baseline_hazards()
                    print("Baseline hazards computed successfully")
            
            # Evaluate the model on validation data
            fold_model.net.eval()
            
            # Use different evaluation approaches for DeepHit vs DeepSurv
            if isinstance(fold_model, DeepHit):
                # For DeepHit competing risks, use specialized evaluation
                from src.competing_risks_evaluation import evaluate_competing_risks_model
                
                evaluation_results = evaluate_competing_risks_model(
                    model=fold_model,
                    x_data=x_val,
                    durations=durations_val.numpy(),
                    events=events_val.numpy(),
                    time_grid=time_grid,
                    optimization_metric=optimization_metric
                )
                
                # Use combined metric for model selection
                metric_value = evaluation_results['combined']['metric_value']
                print(f"Fold {fold+1} combined competing risks metric: {metric_value:.4f}")
                print(f"  - Cause 1 (RRT/eGFR<15): {evaluation_results['cause_1']['metric_value']:.4f}")
                print(f"  - Cause 2 (Mortality): {evaluation_results['cause_2']['metric_value']:.4f}")
                
            else:
                # For DeepSurv, use standard survival evaluation
                with torch.no_grad():
                    val_preds = fold_model.predict_surv_df(x_val)
                
                ev = EvalSurv(
                    val_preds,
                    durations_val.numpy(),
                    events_val.numpy(),
                    censor_surv='km'
                )
                
                # Calculate metric based on optimization_metric
                if optimization_metric == 'cidx':
                    metric_value = ev.concordance_td()
                    print(f"Fold {fold+1} concordance index: {metric_value:.4f}")
                elif optimization_metric == 'brs':
                    metric_value = -ev.integrated_brier_score(time_grid)
                    print(f"Fold {fold+1} integrated Brier score: {-metric_value:.4f}")
                elif optimization_metric == 'loglik':
                    metric_value = -ev.integrated_nbll(time_grid)
                    print(f"Fold {fold+1} log-likelihood: {metric_value:.4f}")
                else:
                    metric_value = ev.concordance_td()
                    print(f"Fold {fold+1} concordance index: {metric_value:.4f}")
            
            # Store model and performance
            models.append(fold_model)
            model_performances.append(metric_value)
            
            # Clear GPU memory
            del x_train, x_val
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        # Select the best model
        best_model_idx = np.argmax(model_performances)
        model = models[best_model_idx]
        
        print(f"\n=== Selected best model from fold {best_model_idx+1} with {optimization_metric} = {model_performances[best_model_idx]:.4f} ===")
        
        # Save the best model weights
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(model.net.state_dict(), model_path)
        print(f"Saved best model weights to {model_path}")
    
    # Function to save metadata to CSV
    def save_metadata_to_csv(durations, events, file_path):
        """
        Save durations and events to CSV file.
        
        Args:
            durations: Array of event times
            events: Array of event indicators
            file_path: Path to save the CSV file
            
        Returns:
            Path to the saved file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Create DataFrame
        metadata_df = pd.DataFrame({
            'duration': durations,
            'event': events
        })
        
        # Save to CSV
        metadata_df.to_csv(file_path, index=False)
        
        print(f"Saved metadata to {file_path}")
        return file_path
    
    # Function to process a dataset in batches
    def process_dataset_in_batches(df, feature_cols, target_endpoint, model, device, batch_size=1000):
        """
        Process a dataset in batches to generate predictions.
        
        Args:
            df: DataFrame to process
            feature_cols: List of feature columns
            target_endpoint: Target endpoint for DeepSurv
            model: Model to use for predictions
            device: Device to use for processing
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (predictions_df, durations, events)
        """
        # Determine the expected number of time points based on model type
        from pycox.models import CoxPH, DeepHit
        
        if isinstance(model, DeepHit):
            # For DeepHit models with competing risks, we expect:
            # num_causes * num_time_points rows (e.g., 2 causes * 5 time points = 10 rows)
            n_time_points = len(model.duration_index)
            num_causes = 2  # Event 1 (RRT/eGFR<15) and Event 2 (Mortality)
            n_expected_rows = num_causes * n_time_points
            print(f"DeepHit model detected with {n_time_points} time points and {num_causes} competing causes")
            print(f"Expected prediction rows: {n_expected_rows} (competing risks format)")
        else:
            # For DeepSurv models, the number of time points is the max value (1825)
            n_time_points = 1825
            n_expected_rows = n_time_points
            print(f"DeepSurv model detected with max time point {n_time_points}")
        
        # Prepare the dataset based on network type
        if network_type.lower() == 'lstm':
            print("Preparing sequence data for LSTM prediction...")
            from src.util import prepare_lstm_survival_dataset
            data = prepare_lstm_survival_dataset(
                df,
                feature_cols=feature_cols,
                sequence_length=sequence_length,
                target_endpoint=target_endpoint
            )
        else:
            # Prepare the dataset
            data = prepare_survival_dataset(
                df,
                feature_cols=feature_cols,
                target_endpoint=target_endpoint
        )
        
        # Extract features, durations, and events
        X, durations, events = data
        
        # Process in batches
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"Processing {n_samples} samples in {n_batches} batches of size {batch_size}")
        
        # Initialize list to store predictions
        all_predictions = []
        
        # Process each batch
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            print(f"Processing batch {i+1}/{n_batches} (samples {start_idx} to {end_idx})")
            
            # Get batch data
            batch_X = X[start_idx:end_idx]
            
            # Convert to torch tensor
            batch_tensor = torch.tensor(batch_X).float().to(device)
            
            # Generate predictions based on model type
            with torch.no_grad():
                if isinstance(model, DeepHit):
                    # For DeepHit, use predict_cif to get competing risks predictions
                    # Shape: (num_causes, num_time_points, batch_size)
                    batch_preds_cif = model.predict_cif(batch_tensor)
                    
                    # Keep the natural (2, 5, n_samples) format for better organization
                    # This format is preferred for:
                    # - Model ensembling
                    # - Clearer metrics calculation
                    # - Better separation of competing events
                    batch_preds = batch_preds_cif.cpu().numpy()
                    
                    print(f"DeepHit CIF predictions shape: {batch_preds.shape} (causes, time_points, samples)")
                else:
                    # For DeepSurv, use the standard survival function
                    batch_preds = model.predict_surv_df(batch_tensor)
            
            # Store predictions
            all_predictions.append(batch_preds)
            
            # Clear GPU memory
            del batch_tensor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        # Combine predictions
        if all_predictions:
            # Handle different prediction formats
            if isinstance(model, DeepHit):
                # For DeepHit, predictions are numpy arrays with shape (2, 5, batch_size)
                print(f"First batch predictions shape: {all_predictions[0].shape}")
                
                # Concatenate along the last axis (samples)
                # Result shape: (2, 5, total_samples)
                predictions_array = np.concatenate(all_predictions, axis=2)
                print(f"Combined DeepHit predictions shape: {predictions_array.shape}")
                print(f"Format: (causes={predictions_array.shape[0]}, time_points={predictions_array.shape[1]}, samples={predictions_array.shape[2]})")
                
                # Store as numpy array for better handling
                predictions_df = predictions_array
                
            elif isinstance(all_predictions[0], pd.DataFrame):
                # For DeepSurv, predictions are DataFrames
                print(f"First batch predictions shape: {all_predictions[0].shape}")
                
                # Check if the number of rows matches expected format
                if abs(all_predictions[0].shape[0] - n_time_points) < (n_time_points * 0.1):  # Allow 10% tolerance
                    print(f"Predictions have the correct format with {all_predictions[0].shape[0]} time points")
                    print("Concatenating predictions along axis 1 (columns)")
                    predictions_df = pd.concat(all_predictions, axis=1)
                    print(f"Combined predictions shape: {predictions_df.shape}")
                else:
                    # The predictions are in the wrong format (batch_size, n_time_points)
                    # We need to transpose each batch before concatenating
                    print(f"Predictions have {all_predictions[0].shape[0]} rows but expected ~{n_time_points} time points")
                    print("Transposing each batch before concatenation")
                    transposed_predictions = [batch.T for batch in all_predictions]
                    predictions_df = pd.concat(transposed_predictions, axis=1)
                    print(f"Combined predictions shape after transposition: {predictions_df.shape}")
                
                # Verify the final shape
                if predictions_df.shape[0] != n_time_points:
                    print(f"WARNING: Expected {n_time_points} time points but got {predictions_df.shape[0]}")
                
                if predictions_df.shape[1] != n_samples:
                    print(f"WARNING: Expected {n_samples} samples but got {predictions_df.shape[1]}")
                    # If we have more columns than samples, truncate
                    if predictions_df.shape[1] > n_samples:
                        print(f"Truncating predictions to {n_samples} samples")
                        predictions_df = predictions_df.iloc[:, :n_samples]
                    # If we have fewer columns than samples, this is unexpected
                    else:
                        print(f"WARNING: Predictions have fewer columns ({predictions_df.shape[1]}) than samples ({n_samples})")
                
                print(f"Final predictions shape: {predictions_df.shape}")
            else:
                # For numpy arrays
                print(f"First batch predictions shape: {all_predictions[0].shape}")
                
                # Check if the predictions are already in the correct format (n_time_points, batch_size)
                
                # Check if the number of rows is close to the expected number of time points
                if abs(all_predictions[0].shape[0] - n_time_points) < (n_time_points * 0.1):  # Allow 10% tolerance
                    # This means the shape is already (n_time_points, batch_size)
                    # Concatenate arrays along axis 1 (columns)
                    print(f"Predictions have the correct format with {all_predictions[0].shape[0]} time points")
                    print("Concatenating predictions along axis 1 (columns)")
                    predictions_df = np.concatenate(all_predictions, axis=1)
                    print(f"Combined predictions shape: {predictions_df.shape}")
                else:
                    # The predictions are in the wrong format (batch_size, n_time_points)
                    # We need to transpose each batch before concatenating
                    print(f"Predictions have {all_predictions[0].shape[0]} rows but expected ~{n_time_points} time points")
                    print("Transposing each batch before concatenation")
                    transposed_predictions = [batch.T for batch in all_predictions]
                    predictions_df = np.concatenate(transposed_predictions, axis=1)
                    print(f"Combined predictions shape after transposition: {predictions_df.shape}")
                
                # Verify the final shape
                if predictions_df.shape[0] != n_time_points:
                    print(f"WARNING: Expected {n_time_points} time points but got {predictions_df.shape[0]}")
                
                if predictions_df.shape[1] != n_samples:
                    print(f"WARNING: Expected {n_samples} samples but got {predictions_df.shape[1]}")
                    # If we have more columns than samples, truncate
                    if predictions_df.shape[1] > n_samples:
                        print(f"Truncating predictions to {n_samples} samples")
                        predictions_df = predictions_df[:, :n_samples]
                    # If we have fewer columns than samples, this is unexpected
                    else:
                        print(f"WARNING: Predictions have fewer columns ({predictions_df.shape[1]}) than samples ({n_samples})")
                
                print(f"Final predictions shape: {predictions_df.shape}")
        else:
            predictions_df = None
        
        return predictions_df, durations, events
    
    # For DeepSurv, we need to compute baseline hazards
    if model_type.lower() == "deepsurv":
        print("\n=== Computing baseline hazards for CoxPH model ===")
        # Preferably use training data for baseline hazards computation
        if train_df_preprocessed is not None and not train_df_preprocessed.empty:
            test_df = train_df_preprocessed
            print("Using training dataset for baseline hazards computation")
        # Fall back to test datasets if training data is not available
        elif temporal_test_df_preprocessed is not None and not temporal_test_df_preprocessed.empty:
            test_df = temporal_test_df_preprocessed
            print("Using temporal test dataset for baseline hazards computation")
        elif spatial_test_df_preprocessed is not None and not spatial_test_df_preprocessed.empty:
            test_df = spatial_test_df_preprocessed
            print("Using spatial test dataset for baseline hazards computation")
        else:
            print("No datasets available for baseline hazards computation")
            test_df = None
        
        if test_df is not None:
            # Prepare the dataset
            # Get target endpoint for DeepSurv
            target_endpoint = model_endpoint if model_type.lower() == "deepsurv" else None
            
            # Prepare the dataset
            test_data = prepare_survival_dataset(
                test_df,
                feature_cols=feature_cols,
                target_endpoint=target_endpoint
            )
            
            # Extract features, durations, and events
            X_test, durations_test, events_test = test_data
            
            # Convert to torch tensors
            x_test = torch.tensor(X_test).float().to(device)
            durations_test_tensor = torch.tensor(durations_test).float()
            events_test_tensor = torch.tensor(events_test).float()
            
            # Create target as a tuple of (durations, events)
            target = (durations_test_tensor, events_test_tensor)
            
            # Compute baseline hazards
            model.compute_baseline_hazards(input=x_test, target=target)
            print("Baseline hazards computed successfully")
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize dictionaries to store prediction paths
    prediction_paths = {}
    metadata_paths = {}
    
    # Create directory for predictions
    predictions_dir = "results/test_predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Process temporal test dataset if provided
    temporal_test_predictions_path = None
    temporal_test_metadata_path = None
    
    if temporal_test_df_preprocessed is not None and not temporal_test_df_preprocessed.empty:
        print("\n=== Processing temporal test dataset for prediction ===")
        # Get target endpoint for DeepSurv
        target_endpoint = model_endpoint if model_type.lower() == "deepsurv" else None
        
        # Process dataset in batches
        temporal_test_surv, durations_temporal_test, events_temporal_test = process_dataset_in_batches(
            df=temporal_test_df_preprocessed,
            feature_cols=feature_cols,
            target_endpoint=target_endpoint,
            model=model,
            device=device,
            batch_size=batch_size
        )
        print(f"Temporal test predictions shape: {temporal_test_surv.shape}")
        
        # For DeepHit models, discretize the ground truth durations before saving
        if model_type.lower() == 'deephit' and time_grid is not None and labtrans is not None:
            print(f"\n=== Discretizing temporal test ground truth durations ===")
            
            # Ensure proper data types before transformation
            test_durations = np.asarray(durations_temporal_test, dtype=np.float64)
            test_events_raw = np.asarray(events_temporal_test, dtype=np.float64)
            
            # Clip durations to be within time grid bounds
            time_grid_array = np.array(time_grid)
            time_grid_min, time_grid_max = time_grid_array.min(), time_grid_array.max()
            test_durations = np.clip(test_durations, time_grid_min, time_grid_max)
            
            # For DeepHit competing risks, only discretize durations, keep original events
            test_idx_durations = labtrans.transform(test_durations, test_events_raw)[0]  # Only take discretized durations
            test_events = test_events_raw  # Keep original competing risk events (0, 1, 2)
            
            print(f"Temporal test discretization:")
            print(f"- Original durations range: {durations_temporal_test.min():.1f} to {durations_temporal_test.max():.1f}")
            print(f"- Discretized duration bins range: {test_idx_durations.min()} to {test_idx_durations.max()}")
            print(f"- Original events: {np.unique(events_temporal_test, return_counts=True)}")
            print(f"- Preserved events: {np.unique(test_events, return_counts=True)}")
            
            # Update the durations and events for saving
            durations_temporal_test = test_idx_durations
            events_temporal_test = test_events
            
            print(f"Updated temporal test data for DeepHit: durations (discrete bins), events (original)")
        
        # The expected shape is (n_time_points, n_samples)
        # Determine the expected number of time points based on model type
        from pycox.models import CoxPH, DeepHit
        
        if isinstance(model, DeepHit):
            # For DeepHit models, the number of time points is the length of the time grid
            # Default is 5 time points: [365, 730, 1095, 1460, 1825]
            n_time_points = len(model.duration_index)
            print(f"DeepHit model detected with {n_time_points} time points")
        else:
            # For DeepSurv models, the number of time points is the max value (1825)
            n_time_points = 1825
            print(f"DeepSurv model detected with max time point {n_time_points}")
        n_samples = len(durations_temporal_test)
        
        if hasattr(temporal_test_surv, 'shape') and len(temporal_test_surv.shape) == 2:
            # Check if the shape matches the expected format
            if temporal_test_surv.shape[0] == n_time_points and temporal_test_surv.shape[1] == n_samples:
                print(f"Predictions have the correct shape: ({n_time_points}, {n_samples})")
            else:
                print(f"WARNING: Predictions shape {temporal_test_surv.shape} does not match expected shape ({n_time_points}, {n_samples})")
                
                # Check if we need to transpose
                if temporal_test_surv.shape[0] == n_samples and temporal_test_surv.shape[1] == n_time_points:
                    print("Transposing predictions to match expected format")
                    if isinstance(temporal_test_surv, pd.DataFrame):
                        temporal_test_surv = temporal_test_surv.T
                    else:
                        temporal_test_surv = temporal_test_surv.T
                    print(f"Transposed predictions shape: {temporal_test_surv.shape}")
                
                # Check if we need to adjust the number of columns (samples)
                if temporal_test_surv.shape[1] != n_samples:
                    print(f"Adjusting predictions to match {n_samples} samples")
                    if isinstance(temporal_test_surv, pd.DataFrame):
                        if temporal_test_surv.shape[1] > n_samples:
                            temporal_test_surv = temporal_test_surv.iloc[:, :n_samples]
                        else:
                            # This case is less likely but handle it anyway
                            # Pad with zeros
                            pad_cols = n_samples - temporal_test_surv.shape[1]
                            pad_df = pd.DataFrame(0, index=temporal_test_surv.index, columns=range(temporal_test_surv.shape[1], n_samples))
                            temporal_test_surv = pd.concat([temporal_test_surv, pad_df], axis=1)
                    else:
                        if temporal_test_surv.shape[1] > n_samples:
                            temporal_test_surv = temporal_test_surv[:, :n_samples]
                        else:
                            # Pad with zeros
                            pad_cols = n_samples - temporal_test_surv.shape[1]
                            pad_array = np.zeros((temporal_test_surv.shape[0], pad_cols))
                            temporal_test_surv = np.hstack([temporal_test_surv, pad_array])
                    
                    print(f"Adjusted predictions shape: {temporal_test_surv.shape}")
        
        # Save predictions to HDF5 file
        temporal_test_predictions_path = f"{predictions_dir}/temporal_test_predictions_{timestamp}.h5"
        temporal_test_metadata_path = f"{predictions_dir}/temporal_test_metadata_{timestamp}.csv"
        
        if isinstance(model, DeepHit):
            # Use competing risks saving format for DeepHit
            from src.competing_risks_evaluation import save_competing_risks_predictions
            save_competing_risks_predictions(
                cif_predictions=temporal_test_surv,
                time_grid=time_grid,
                durations=durations_temporal_test,
                events=events_temporal_test,
                save_path=temporal_test_predictions_path,
                metadata_path=temporal_test_metadata_path
            )
        else:
            # Use standard saving for DeepSurv
            save_predictions_to_hdf5(
                temporal_test_surv,
                temporal_test_predictions_path,
                metadata={'durations': durations_temporal_test, 'events': events_temporal_test}
            )
            
            # Save metadata to CSV file
            save_metadata_to_csv(
                durations_temporal_test,
                events_temporal_test,
                temporal_test_metadata_path
            )
        
        # Store paths
        prediction_paths['temporal_test'] = temporal_test_predictions_path
        metadata_paths['temporal_test'] = temporal_test_metadata_path
        
        print("Temporal test dataset processed successfully")
    
    # Process spatial test dataset if provided
    spatial_test_predictions_path = None
    spatial_test_metadata_path = None
    
    if spatial_test_df_preprocessed is not None and not spatial_test_df_preprocessed.empty:
        print("\n=== Processing spatial test dataset for prediction ===")
        # Get target endpoint for DeepSurv
        target_endpoint = model_endpoint if model_type.lower() == "deepsurv" else None
        
        # Process dataset in batches
        spatial_test_surv, durations_spatial_test, events_spatial_test = process_dataset_in_batches(
            df=spatial_test_df_preprocessed,
            feature_cols=feature_cols,
            target_endpoint=target_endpoint,
            model=model,
            device=device,
            batch_size=batch_size
        )
        print(f"Spatial test predictions shape: {spatial_test_surv.shape}")
        
        # For DeepHit models, discretize the ground truth durations before saving
        if model_type.lower() == 'deephit' and time_grid is not None and labtrans is not None:
            print(f"\n=== Discretizing spatial test ground truth durations ===")
            
            # Ensure proper data types before transformation
            test_durations = np.asarray(durations_spatial_test, dtype=np.float64)
            test_events_raw = np.asarray(events_spatial_test, dtype=np.float64)
            
            # Clip durations to be within time grid bounds
            time_grid_array = np.array(time_grid)
            time_grid_min, time_grid_max = time_grid_array.min(), time_grid_array.max()
            test_durations = np.clip(test_durations, time_grid_min, time_grid_max)
            
            # For DeepHit competing risks, only discretize durations, keep original events
            test_idx_durations = labtrans.transform(test_durations, test_events_raw)[0]  # Only take discretized durations
            test_events = test_events_raw  # Keep original competing risk events (0, 1, 2)
            
            print(f"Spatial test discretization:")
            print(f"- Original durations range: {durations_spatial_test.min():.1f} to {durations_spatial_test.max():.1f}")
            print(f"- Discretized duration bins range: {test_idx_durations.min()} to {test_idx_durations.max()}")
            print(f"- Original events: {np.unique(events_spatial_test, return_counts=True)}")
            print(f"- Preserved events: {np.unique(test_events, return_counts=True)}")
            
            # Update the durations and events for saving
            durations_spatial_test = test_idx_durations
            events_spatial_test = test_events
            
            print(f"Updated spatial test data for DeepHit: durations (discrete bins), events (original)")
        
        # The expected shape is (n_time_points, n_samples)
        # Determine the expected number of time points based on model type
        from pycox.models import CoxPH, DeepHit
        
        if isinstance(model, DeepHit):
            # For DeepHit models, the number of time points is the length of the time grid
            # Default is 5 time points: [365, 730, 1095, 1460, 1825]
            n_time_points = len(model.duration_index)
            print(f"DeepHit model detected with {n_time_points} time points")
        else:
            # For DeepSurv models, the number of time points is the max value (1825)
            n_time_points = 1825
            print(f"DeepSurv model detected with max time point {n_time_points}")
        n_samples = len(durations_spatial_test)
        
        if hasattr(spatial_test_surv, 'shape') and len(spatial_test_surv.shape) == 2:
            # Check if the shape matches the expected format
            if spatial_test_surv.shape[0] == n_time_points and spatial_test_surv.shape[1] == n_samples:
                print(f"Predictions have the correct shape: ({n_time_points}, {n_samples})")
            else:
                print(f"WARNING: Predictions shape {spatial_test_surv.shape} does not match expected shape ({n_time_points}, {n_samples})")
                
                # Check if we need to transpose
                if spatial_test_surv.shape[0] == n_samples and spatial_test_surv.shape[1] == n_time_points:
                    print("Transposing predictions to match expected format")
                    if isinstance(spatial_test_surv, pd.DataFrame):
                        spatial_test_surv = spatial_test_surv.T
                    else:
                        spatial_test_surv = spatial_test_surv.T
                    print(f"Transposed predictions shape: {spatial_test_surv.shape}")
                
                # Check if we need to adjust the number of columns (samples)
                if spatial_test_surv.shape[1] != n_samples:
                    print(f"Adjusting predictions to match {n_samples} samples")
                    if isinstance(spatial_test_surv, pd.DataFrame):
                        if spatial_test_surv.shape[1] > n_samples:
                            spatial_test_surv = spatial_test_surv.iloc[:, :n_samples]
                        else:
                            # This case is less likely but handle it anyway
                            # Pad with zeros
                            pad_cols = n_samples - spatial_test_surv.shape[1]
                            pad_df = pd.DataFrame(0, index=spatial_test_surv.index, columns=range(spatial_test_surv.shape[1], n_samples))
                            spatial_test_surv = pd.concat([spatial_test_surv, pad_df], axis=1)
                    else:
                        if spatial_test_surv.shape[1] > n_samples:
                            spatial_test_surv = spatial_test_surv[:, :n_samples]
                        else:
                            # Pad with zeros
                            pad_cols = n_samples - spatial_test_surv.shape[1]
                            pad_array = np.zeros((spatial_test_surv.shape[0], pad_cols))
                            spatial_test_surv = np.hstack([spatial_test_surv, pad_array])
                    
                    print(f"Adjusted predictions shape: {spatial_test_surv.shape}")
        
        # Save predictions to HDF5 file
        spatial_test_predictions_path = f"{predictions_dir}/spatial_test_predictions_{timestamp}.h5"
        spatial_test_metadata_path = f"{predictions_dir}/spatial_test_metadata_{timestamp}.csv"
        
        if isinstance(model, DeepHit):
            # Use competing risks saving format for DeepHit
            from src.competing_risks_evaluation import save_competing_risks_predictions
            save_competing_risks_predictions(
                cif_predictions=spatial_test_surv,
                time_grid=time_grid,
                durations=durations_spatial_test,
                events=events_spatial_test,
                save_path=spatial_test_predictions_path,
                metadata_path=spatial_test_metadata_path
            )
        else:
            # Use standard saving for DeepSurv
            save_predictions_to_hdf5(
                spatial_test_surv,
                spatial_test_predictions_path,
                metadata={'durations': durations_spatial_test, 'events': events_spatial_test}
            )
            
            # Save metadata to CSV file
            save_metadata_to_csv(
                durations_spatial_test,
                events_spatial_test,
                spatial_test_metadata_path
            )
        
        # Store paths
        prediction_paths['spatial_test'] = spatial_test_predictions_path
        metadata_paths['spatial_test'] = spatial_test_metadata_path
        
        print("Spatial test dataset processed successfully")
    
    # 8. Save model weights
    deployed_model_weights_path = f"results/model_details/deployed_model_weights_{timestamp}.pt"
    os.makedirs(os.path.dirname(deployed_model_weights_path), exist_ok=True)
    torch.save(model.net.state_dict(), deployed_model_weights_path)
    
    # 9. Register model with MLflow if requested
    if register_model:
        print(f"\n=== Registering model with MLflow as {model_name} (Stage: {model_stage}) ===\n")
        
        # Set MLflow tracking URI if needed
        # mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Start MLflow run
        mlflow.end_run()
        with mlflow.start_run(run_name=f"{model_name}_{timestamp}"):
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("model_endpoint", model_endpoint)
            mlflow.log_param("input_dim", input_dim)
            mlflow.log_param("hidden_dims", hidden_dims)
            mlflow.log_param("output_dim", output_dim)
            mlflow.log_param("dropout", dropout)
            
            if model_type.lower() == "deephit":
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("sigma", sigma)
                mlflow.log_param("time_grid", time_grid)
            
            # Log model
            mlflow.pytorch.log_model(
                model.net,
                artifact_path="model",
                registered_model_name=model_name
            )
            
            # Log model weights file
            mlflow.log_artifact(deployed_model_weights_path)
            
            # Set model stage
            client = mlflow.tracking.MlflowClient()
            latest_version = 1  # Default to version 1
            
            try:
                # Get latest version
                latest_versions = client.get_latest_versions(model_name)
                if latest_versions:
                    latest_version = max([int(v.version) for v in latest_versions])
                
                # Transition model to specified stage
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version,
                    stage=model_stage
                )
                
                print(f"Model {model_name} version {latest_version} transitioned to {model_stage} stage")
            except Exception as e:
                print(f"Warning: Could not transition model to {model_stage} stage: {e}")
    
    # 10. Save deployment details
    deployment_details = {
        'model_type': model_type,
        'model_endpoint': model_endpoint,
        'original_model_path': model_path,
        'deployed_model_weights_path': deployed_model_weights_path,
        'hyperparameters': hyperparameters,
        'timestamp': timestamp,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dims': hidden_dims,
        'time_grid': time_grid.tolist() if model_type.lower() == "deephit" and time_grid is not None else None,
        'model_name': model_name,
        'model_stage': model_stage
    }
    
    # Add prediction file paths
    if prediction_paths:
        deployment_details['prediction_paths'] = prediction_paths
    
    if metadata_paths:
        deployment_details['metadata_paths'] = metadata_paths
    
    # Add specific paths for backward compatibility
    if temporal_test_predictions_path:
        deployment_details['temporal_test_predictions_path'] = temporal_test_predictions_path
    
    if temporal_test_metadata_path:
        deployment_details['temporal_test_metadata_path'] = temporal_test_metadata_path
    
    if spatial_test_predictions_path:
        deployment_details['spatial_test_predictions_path'] = spatial_test_predictions_path
    
    if spatial_test_metadata_path:
        deployment_details['spatial_test_metadata_path'] = spatial_test_metadata_path
    
    # Save deployment details
    deployed_model_details_path = f"results/model_details/deployed_model_details_{timestamp}.json"
    with open(deployed_model_details_path, 'w') as f:
        json.dump(deployment_details, f, indent=2)
    
    print(f"Deployed model weights saved to {deployed_model_weights_path}")
    print(f"Deployed model details saved to {deployed_model_details_path}")
    
    # Return deployment details
    return deployment_details