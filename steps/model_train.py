"""
Model Training Step for CKD Risk Prediction

This module contains the ZenML step for training the CKD risk prediction model
with hyperparameter optimization.
"""

import os
import yaml
import json
import numpy as np
import pandas as pd
import torch
import mlflow
import optuna
from zenml.steps import step
from zenml.client import Client
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import traceback

# Import callbacks from torchtuples
from torchtuples.callbacks import EarlyStopping

# Import neural network architectures
from src.nn_architectures import create_network

# Import PyCox models
from pycox.models import CoxPH, DeepHit
from pycox.evaluation import EvalSurv
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

# SciPy compatibility fix for PyCox
import scipy.integrate
try:
    from scipy.integrate import simps
except ImportError:
    # In newer SciPy versions, simps has been renamed to simpson
    from scipy.integrate import simpson
    scipy.integrate.simps = simpson

# Import utility functions
from src.util import extract_from_step_artifact, access_zenml_artifact_data, load_yaml_file
from steps.cv_utils import create_random_cv_splits, create_cv_datasets, time_based_patient_cv
from src.balance_data import balance_dataframe

# Import sequence utilities for LSTM (now in util.py)
from src.util import prepare_lstm_survival_dataset, print_sequence_summary


def prepare_survival_dataset(df, feature_cols=None, duration_col="duration", event_col="endpoint", target_endpoint=None):
    """Prepare a dataset for survival analysis
    
    Args:
        df: Input dataframe
        feature_cols: List of feature columns to include (optional)
        duration_col: Name of the duration column (default: "duration")
        event_col: Name of the event column (default: "endpoint")
        target_endpoint: Specific event type to focus on (optional)
            If specified, only this event type will be considered as an event (1),
            all other event types will be treated as censored (0)
    """
    print(f"[DEBUG] prepare_survival_dataset called with target_endpoint: {target_endpoint}")
    # Print column types to help with debugging
    print(f"DataFrame dtypes sample: {df.dtypes.head(10)}")
    
    # Columns to exclude from features
    exclude_cols = [
        duration_col, event_col, 'patient_id', 'key', 'date',
        'endpoint_date', 'first_sub_60_date', 'dob', 'icd10'
    ]
    
    # Only keep columns that exist in the dataframe
    exclude_cols = [col for col in exclude_cols if col in df.columns]
    print(f"Excluding columns: {exclude_cols}")
    
    # Convert categorical columns to numeric
    feature_df = df.drop(exclude_cols, axis=1, errors='ignore')
    
    # Filter features based on master dataframe mapping if provided
    if feature_cols:
        print(f"Filtering features based on master dataframe mapping: {feature_cols}")
        # Keep only columns that exist in the dataframe
        available_cols = [col for col in feature_cols if col in feature_df.columns]
        if len(available_cols) < len(feature_cols):
            missing_cols = set(feature_cols) - set(available_cols)
            print(f"Warning: Some columns from mapping not found in dataframe: {missing_cols}")
        
        if not available_cols:
            print("Warning: None of the specified feature columns found in dataframe. Using all available features.")
        else:
            feature_df = feature_df[available_cols]
    
    # Convert categorical columns to numeric
    for col in feature_df.select_dtypes(include=['object', 'category']).columns:
        print(f"Converting categorical column to numeric: {col}")
        # For binary categorical variables like gender, convert to 0/1
        if feature_df[col].nunique() <= 2:
            # Get the most common value to use as reference (0)
            most_common = feature_df[col].mode()[0]
            feature_df[col] = (feature_df[col] != most_common).astype(float)
        else:
            # For categorical variables with more than 2 categories, use one-hot encoding
            dummies = pd.get_dummies(feature_df[col], prefix=col, drop_first=True)
            feature_df = pd.concat([feature_df.drop(col, axis=1), dummies], axis=1)
    
    # Convert to numpy arrays
    X = feature_df.values.astype(float)
    
    # Get durations and events
    durations = df[duration_col].values.astype(float)
    events = df[event_col].values.astype(float)
    
    # If target_endpoint is specified, convert events to binary (1 for target event, 0 otherwise)
    if target_endpoint is not None:
        print(f"Converting events to binary for target endpoint: {target_endpoint}")
        # Show original event distribution
        event_counts = np.bincount(events.astype(int))
        print(f"Original event distribution: {event_counts}")
        
        # Create binary events: 1 if event matches target_endpoint, 0 otherwise
        binary_events = (events == target_endpoint).astype(float)
        
        # Show binary event distribution
        binary_counts = np.bincount(binary_events.astype(int))
        print(f"Binary event distribution: {binary_counts}")
        print(f"Target event rate: {binary_events.mean():.2%}")
        
        # Replace events with binary version
        events = binary_events
    
    print(f"Feature matrix shape: {X.shape}, with dtypes: {X.dtype}")
    print(f"Duration vector shape: {durations.shape}, with dtype: {durations.dtype}")
    print(f"Event vector shape: {events.shape}, with dtype: {events.dtype}")
    
    # Return as tuple
    return (X, durations, events)


experiment_tracker = Client().active_stack.experiment_tracker
@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def train_model(
    train_df_preprocessed: pd.DataFrame,
    master_df_mapping_path: str = "src/default_master_df_mapping.yml",
    hyperparameter_config_path: str = "src/hyperparameter_config.yml"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train a CKD risk prediction model with hyperparameter optimization.
    
    Args:
        train_df_preprocessed: Preprocessed training dataframe
        master_df_mapping_path: Path to the master dataframe mapping YAML file
        hyperparameter_config_path: Path to hyperparameter configuration YAML file
        
    Returns:
        Tuple containing:
        - model_details: Dictionary with optimized model weights and structure
        - optimization_metrics: Dictionary with hyperparameter optimization metrics
    """
    try:
        print("\n=== Training CKD Risk Prediction Model with Hyperparameter Optimization ===\n")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load master dataframe mapping
        master_df_mapping = load_yaml_file(master_df_mapping_path)
        if not master_df_mapping:
            raise ValueError(f"Failed to load master dataframe mapping from {master_df_mapping_path}")
        
        print(f"Loaded master dataframe mapping with {len(master_df_mapping.get('features', []))} features")
        
        # Load hyperparameter configuration
        hp_config = load_yaml_file(hyperparameter_config_path)
        if not hp_config:
            raise ValueError(f"Failed to load hyperparameter configuration from {hyperparameter_config_path}")
        
        print(f"Loaded hyperparameter configuration: {hp_config['model_type']} model")
        
        # Prepare training dataset
        print("\n=== Preparing survival dataset from preprocessed dataframe ===")
        print(f"Training dataframe shape: {train_df_preprocessed.shape}")
        print(f"Training dataframe columns: {train_df_preprocessed.columns[:10]}...")
        
        # Get features from the master dataframe mapping
        feature_cols = master_df_mapping.get("features", [])
        if not feature_cols:
            print("Warning: No features found in master dataframe mapping. Using all features from dataset.")
        
        # Get model type, network type, and target endpoint from config
        model_type = hp_config.get('model_type', 'deepsurv')
        network_type = hp_config.get('network', {}).get('type', 'ann')
        target_endpoint = None
        
        print(f"Network type: {network_type}")
        print(f"Model type: {model_type}")
        
        # If using DeepSurv, check if a specific endpoint is targeted
        if model_type.lower() == 'deepsurv':
            target_endpoint = hp_config.get('target_endpoint')
            if target_endpoint is not None:
                print(f"Training DeepSurv model for specific target endpoint: {target_endpoint}")
            else:
                print("Training DeepSurv model for all event types (any non-zero event)")
        
        # Get balancing configuration from hyperparameter config
        balance_cfg = hp_config.get("balance", {})
        orig_rows = len(train_df_preprocessed)
        
        # Apply balancing if enabled
        if balance_cfg.get("enable", False):
            print("\n=== Applying class balancing to training data ===")
            
            print(f"Model type: {model_type}")
            print(f"Target endpoint: {target_endpoint}")
            
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
                target_endpoint=target_endpoint,
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
        
        # Extract patient IDs (cluster column) and entry dates before preparing the dataset
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
        
        # Prepare training dataset based on network type
        print("\n=== Preparing training dataset ===")
        print(f"Training dataframe shape before preparation: {train_df_preprocessed.shape}")
        print(f"Using target_endpoint: {target_endpoint}")
        print(f"Network type: {network_type}")
        
        if network_type.lower() == 'lstm':
            # For LSTM, we need sequence data
            print("Preparing sequence data for LSTM...")
            
            # Get default sequence length from config (will be optimized later)
            default_sequence_length = hp_config.get('network', {}).get('lstm', {}).get('sequence_length', 5)
            print(f"Using default sequence length: {default_sequence_length}")
            
            # Prepare LSTM dataset
            X_sequences, durations, events = prepare_lstm_survival_dataset(
                df=train_df_preprocessed,
                sequence_length=default_sequence_length,
                feature_cols=feature_cols,
                cluster_col=cluster_col,
                date_col='date',
                duration_col='duration',
                event_col='endpoint',
                target_endpoint=target_endpoint
            )
            
            # Create train_data tuple compatible with existing code
            train_data = (X_sequences, durations, events)
            
            # Print sequence summary
            print_sequence_summary(X_sequences, durations, events, "Training")
            
            # Set input dimension for LSTM (features per timestep)
            input_dim = X_sequences.shape[2]
            sequence_length = X_sequences.shape[1]
            print(f"Input dimension (features per timestep): {input_dim}")
            print(f"Sequence length: {sequence_length}")
            
        else:
            # For ANN/MLP, use existing row-based data preparation
            print("Preparing row-based data for ANN/MLP...")
            
            train_data = prepare_survival_dataset(
                train_df_preprocessed,
                feature_cols=feature_cols,
                target_endpoint=target_endpoint
            )
            
            # Validate dataset
            validate_dataset(train_data, "train_dataset")
            
            # Log feature statistics
            X_train = train_data[0]
            print(f"\n=== Feature statistics for training dataset ===")
            print(f"Feature matrix shape: {X_train.shape}")
            print(f"Feature mean: {np.mean(X_train, axis=0)[:5]}...")
            print(f"Feature std: {np.std(X_train, axis=0)[:5]}...")
            print(f"Feature min: {np.min(X_train, axis=0)[:5]}...")
            print(f"Feature max: {np.max(X_train, axis=0)[:5]}...")
            
            # Set input dimension based on the actual number of features in the filtered data
            input_dim = train_data[0].shape[1]
            sequence_length = None  # Not used for ANN
            print(f"Input dimension (after filtering): {input_dim}")
        
        # Get optimization settings from config
        n_trials = hp_config.get('optimization', {}).get('n_trials', 10)
        patience = hp_config.get('optimization', {}).get('patience', 10)
        seed = hp_config.get('optimization', {}).get('seed', 42)
        
        # Get time grid for DeepHit model
        time_grid = None
        if model_type.lower() == 'deephit':
            time_grid = hp_config.get('network', {}).get('deephit', {}).get('time_grid', None)
            if time_grid is None:
                # Create time grid based on training data
                max_duration = train_data[1].max()
                time_grid = np.linspace(0, max_duration, 10)
                print(f"Created time grid with 10 points up to {max_duration:.1f} days")
            else:
                # Convert time_grid to numpy array for PyCox compatibility
                time_grid = np.array(time_grid)
                print(f"Using time grid from config: {time_grid}")
        
        # Pre-discretize data for DeepHit models BEFORE creating CV splits
        if model_type.lower() == 'deephit':
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
            
            # CRITICAL: Clip durations to be within time grid bounds BEFORE discretization
            # This prevents LabTransDiscreteTime from creating bins beyond the time_grid
            time_grid_array = np.array(time_grid)
            time_grid_min, time_grid_max = time_grid_array.min(), time_grid_array.max()
            
            print(f"Original duration range: {train_durations.min():.1f} to {train_durations.max():.1f}")
            print(f"Time grid range: {time_grid_min:.1f} to {time_grid_max:.1f}")
            
            # Clip durations to time_grid bounds to prevent out-of-bounds discretization
            train_durations = np.clip(train_durations, time_grid_min, time_grid_max)
            print(f"Clipped duration range: {train_durations.min():.1f} to {train_durations.max():.1f}")
            
            print(f"Pre-transform - Duration bounds: Min: {time_grid_min}, Max: {time_grid_max}")
            print(f"Pre-transform - Data types: durations {train_durations.dtype}, events {train_events_raw.dtype}")
            
            # CRITICAL: Force proper discretization with aggressive bounds checking
            # Skip LabTransDiscreteTime entirely and use direct manual discretization
            print(f"\n=== DIRECT MANUAL DISCRETIZATION (BYPASSING LABTRANSDISCRETETIME) ===")
            print(f"Time grid: {time_grid}")
            print(f"Input duration range: {train_durations.min():.1f} to {train_durations.max():.1f}")
            
            # Use np.digitize to properly map durations to time grid bins
            # digitize returns 1-based indices, so subtract 1 to get 0-based
            train_idx_durations = np.digitize(train_durations, time_grid) - 1
            
            # CRITICAL: Force all indices to be within valid bounds [0, len(time_grid)-1]
            max_valid_idx = len(time_grid) - 1
            train_idx_durations = np.clip(train_idx_durations, 0, max_valid_idx)
            train_events = train_events_raw.astype(np.int64)
            
            print(f"✓ Direct discretization result:")
            print(f"  Discretized duration range: {train_idx_durations.min()}-{train_idx_durations.max()}")
            print(f"  Unique discretized durations: {np.unique(train_idx_durations)}")
            print(f"  Events shape: {train_events.shape}, unique: {np.unique(train_events)}")
            print(f"  Valid range enforced: 0-{max_valid_idx}")
            
            # Convert to tensors
            train_idx_durations = torch.tensor(train_idx_durations).long()
            train_events = torch.tensor(train_events).long()
            print(f"✓ Converted to tensors successfully")
            
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
        cv_folds = 5
        print(f"\n=== Creating {cv_folds}-fold time-based cross-validation splits ===")
        
        # For LSTM, we need to use sequence data for CV splits, not original tabular data
        if network_type.lower() == 'lstm':
            print("LSTM detected: Using sequence data for cross-validation")
            cv_splits = create_random_cv_splits(train_data, n_splits=cv_folds, seed=seed)
        else:
            # Use time-based cross-validation if patient_ids and entry_dates are available
            if patient_ids is not None and entry_dates is not None:
                print("Using time-based patient cross-validation")
                cv_splits = time_based_patient_cv(
                    dataset=train_data,
                    patient_ids=patient_ids,
                    entry_dates=entry_dates,
                    cv_folds=cv_folds,
                    seed=seed
                )
            else:
                print("Warning: Patient IDs or entry dates not available. Falling back to random cross-validation.")
                cv_splits = create_random_cv_splits(train_data, n_splits=cv_folds, seed=seed)
            
        print(f"Created {len(cv_splits)} CV splits")
        
        # Define the objective function for Optuna
        def objective(trial):
            # Use nonlocal to access variables from outer scope
            nonlocal cv_splits, time_grid, original_continuous_data
            
            # Set random seeds for reproducibility
            torch.manual_seed(seed + trial.number)
            np.random.seed(seed + trial.number)
            
            # Sample hyperparameters from search space
            search_space = hp_config.get('search_space', {})
            common_space = search_space.get('common', {})
            network_config = hp_config.get('network', {})
            
            # Learning rate
            lr_config = common_space.get('learning_rate', {"type": "float", "min": 1e-4, "max": 1e-2, "log": True})
            lr = trial.suggest_float(
                "learning_rate",
                lr_config.get("min", 1e-4),
                lr_config.get("max", 1e-2),
                log=lr_config.get("log", True)
            )
            
            # Number of layers
            layers_config = common_space.get('num_layers', {"type": "int", "min": 2, "max": 4})
            num_layers = trial.suggest_int(
                "num_layers",
                layers_config.get("min", 2),
                layers_config.get("max", 4)
            )
            
            # Hidden units for each layer
            units_config_layer1 = common_space.get('hidden_units_layer1', {"type": "int", "min": 16, "max": 128})
            hidden_units_layer1 = trial.suggest_int(
                "hidden_units_layer1",
                units_config_layer1.get("min", 16),
                units_config_layer1.get("max", 128)
            )
            
            units_config_layer2 = common_space.get('hidden_units_layer2', {"type": "int", "min": 16, "max": 128})
            hidden_units_layer2 = trial.suggest_int(
                "hidden_units_layer2",
                units_config_layer2.get("min", 16),
                units_config_layer2.get("max", 128)
            )
            
            units_config_layer3 = common_space.get('hidden_units_layer3', {"type": "int", "min": 8, "max": 64})
            hidden_units_layer3 = trial.suggest_int(
                "hidden_units_layer3",
                units_config_layer3.get("min", 8),
                units_config_layer3.get("max", 64)
            )
            
            units_config_layer4 = common_space.get('hidden_units_layer4', {"type": "int", "min": 4, "max": 32})
            hidden_units_layer4 = trial.suggest_int(
                "hidden_units_layer4",
                units_config_layer4.get("min", 4),
                units_config_layer4.get("max", 32)
            )
            
            # Dropout
            dropout_config = common_space.get('dropout', {"type": "float", "min": 0.0, "max": 0.5})
            dropout = trial.suggest_float(
                "dropout",
                dropout_config.get("min", 0.0),
                dropout_config.get("max", 0.5)
            )
            
            # Optimizer
            optimizer_config = common_space.get('optimizer', {"type": "categorical", "values": ["Adam", "AdamW"]})
            optimizer_name = trial.suggest_categorical(
                "optimizer",
                optimizer_config.get("values", ["Adam", "AdamW"])
            )
            
            # Batch size
            batch_config = common_space.get('batch_size', {"type": "categorical", "values": [32, 64, 128, 256]})
            batch_size = trial.suggest_categorical(
                "batch_size",
                batch_config.get("values", [32, 64, 128, 256])
            )
            
            # LSTM-specific hyperparameters (if using LSTM)
            lstm_sequence_length = None
            lstm_hidden_dims = None
            lstm_num_layers = None
            lstm_bidirectional = None
            
            if network_type.lower() == 'lstm':
                lstm_space = search_space.get('lstm', {})
                
                # Sequence length
                seq_config = lstm_space.get('sequence', {"type": "int", "min": 3, "max": 10})
                lstm_sequence_length = trial.suggest_int(
                    "sequence",
                    seq_config.get("min", 3),
                    seq_config.get("max", 10)
                )
                
                # LSTM number of layers
                layers_config = lstm_space.get('lstm_num_layers', {"type": "int", "min": 1, "max": 3})
                lstm_num_layers = trial.suggest_int(
                    "lstm_num_layers",
                    layers_config.get("min", 1),
                    layers_config.get("max", 3)
                )
                
                # LSTM hidden dimensions (layer-specific, similar to ANN)
                lstm_hidden_dims = []
                
                # Layer 1
                if lstm_num_layers >= 1:
                    layer1_config = lstm_space.get('lstm_hidden_dim_layer1', {"type": "int", "min": 64, "max": 128})
                    hidden_dim_layer1 = trial.suggest_int(
                        "lstm_hidden_dim_layer1",
                        layer1_config.get("min", 64),
                        layer1_config.get("max", 128)
                    )
                    lstm_hidden_dims.append(hidden_dim_layer1)
                
                # Layer 2
                if lstm_num_layers >= 2:
                    layer2_config = lstm_space.get('lstm_hidden_dim_layer2', {"type": "int", "min": 32, "max": 96})
                    hidden_dim_layer2 = trial.suggest_int(
                        "lstm_hidden_dim_layer2",
                        layer2_config.get("min", 32),
                        layer2_config.get("max", 96)
                    )
                    lstm_hidden_dims.append(hidden_dim_layer2)
                
                # Layer 3
                if lstm_num_layers >= 3:
                    layer3_config = lstm_space.get('lstm_hidden_dim_layer3', {"type": "int", "min": 16, "max": 64})
                    hidden_dim_layer3 = trial.suggest_int(
                        "lstm_hidden_dim_layer3",
                        layer3_config.get("min", 16),
                        layer3_config.get("max", 64)
                    )
                    lstm_hidden_dims.append(hidden_dim_layer3)
                
                # Bidirectional
                bidir_config = lstm_space.get('bidirectional', {"type": "categorical", "values": [True, False]})
                lstm_bidirectional = trial.suggest_categorical(
                    "bidirectional",
                    bidir_config.get("values", [True, False])
                )
                
                print(f"LSTM hyperparameters - Sequence: {lstm_sequence_length}, "
                      f"Hidden dims: {lstm_hidden_dims}, Layers: {lstm_num_layers}, "
                      f"Bidirectional: {lstm_bidirectional}")
            
            # Model-specific hyperparameters
            if model_type.lower() == "deephit":
                # Get DeepHit-specific search space
                deephit_space = search_space.get("deephit", {})
                
                # Alpha
                alpha_config = deephit_space.get("alpha", {"type": "float", "min": 0.1, "max": 1.0})
                alpha = trial.suggest_float(
                    "alpha",
                    alpha_config.get("min", 0.1),
                    alpha_config.get("max", 1.0)
                )
                
                # Sigma
                sigma_config = deephit_space.get("sigma", {"type": "float", "min": 0.1, "max": 1.0})
                sigma = trial.suggest_float(
                    "sigma",
                    sigma_config.get("min", 0.1),
                    sigma_config.get("max", 1.0)
                )
                
                output_dim = len(time_grid)
            else:
                alpha = None
                sigma = None
                output_dim = 1
                
                # For DeepSurv, we still need time_grid for metric calculations
                # Use the default time grid from the DeepHit configuration if available
                if time_grid is None:
                    deephit_config = network_config.get('deephit', {})
                    time_grid = np.array(deephit_config.get('time_grid', [365, 730, 1095, 1460, 1825]))
                    print(f"Using default time grid for metric calculations: {time_grid}")
            
            # Prepare data for this trial (regenerate sequences if LSTM with different sequence length)
            trial_train_data = train_data
            trial_input_dim = input_dim
            trial_sequence_length = sequence_length
            
            if network_type.lower() == 'lstm' and lstm_sequence_length != sequence_length:
                # Regenerate sequences with the trial-specific sequence length
                print(f"Regenerating sequences with length {lstm_sequence_length} for this trial...")
                
                X_sequences_trial, durations_trial, events_trial = prepare_lstm_survival_dataset(
                    df=train_df_preprocessed,
                    sequence_length=lstm_sequence_length,
                    feature_cols=feature_cols,
                    cluster_col=cluster_col,
                    date_col='date',
                    duration_col='duration',
                    event_col='endpoint',
                    target_endpoint=target_endpoint
                )
                
                # CRITICAL: For DeepHit, apply discretization to the newly generated LSTM sequences
                if model_type.lower() == "deephit":
                    print(f"Applying discretization to newly generated LSTM sequences for DeepHit")
                    print(f"Original LSTM sequence durations range: {durations_trial.min():.1f} to {durations_trial.max():.1f}")
                    
                    # Apply the same discretization logic as before
                    time_grid_min = time_grid[0]
                    time_grid_max = time_grid[-1]
                    
                    # Clip durations to time_grid bounds
                    durations_trial_clipped = np.clip(durations_trial, time_grid_min, time_grid_max)
                    
                    # Use np.digitize to properly map durations to time grid bins
                    durations_trial_discrete = np.digitize(durations_trial_clipped, time_grid) - 1
                    
                    # Force all indices to be within valid bounds [0, len(time_grid)-1]
                    max_valid_idx = len(time_grid) - 1
                    durations_trial_discrete = np.clip(durations_trial_discrete, 0, max_valid_idx)
                    
                    print(f"Discretized LSTM sequence durations range: {durations_trial_discrete.min()} to {durations_trial_discrete.max()}")
                    print(f"Unique discretized bins: {np.unique(durations_trial_discrete)}")
                    
                    # Use discretized durations
                    durations_trial = durations_trial_discrete
                
                trial_train_data = (X_sequences_trial, durations_trial, events_trial)
                trial_input_dim = X_sequences_trial.shape[2]  # Features per timestep
                trial_sequence_length = lstm_sequence_length
                
                # Check if data size changed and regenerate CV splits if needed
                original_size = train_data[0].shape[0]
                new_size = X_sequences_trial.shape[0]
                
                if new_size != original_size:
                    print(f"Data size changed from {original_size} to {new_size} due to sequence generation.")
                    print(f"Regenerating CV splits for sequence data...")
                    
                    # Regenerate CV splits for the new sequence data
                    if patient_ids is not None and entry_dates is not None:
                        print("Using time-based patient cross-validation for sequence data")
                        # For sequence data, we need to extract patient info from the sequence data
                        # Since sequences are per-patient, we can use a simple approach
                        cv_splits = create_random_cv_splits(trial_train_data, n_splits=cv_folds, seed=seed)
                    else:
                        print("Using random cross-validation for sequence data")
                        cv_splits = create_random_cv_splits(trial_train_data, n_splits=cv_folds, seed=seed)
                    
                    print(f"Regenerated {len(cv_splits)} CV splits for sequence data")
            
            # Build network architecture
            if network_type.lower() == 'lstm':
                # For LSTM, use layer-specific hidden dimensions
                net = create_network(
                    model_type=model_type,
                    input_dim=trial_input_dim,
                    output_dim=output_dim,
                    dropout=dropout,
                    network_type='lstm',
                    sequence_length=trial_sequence_length,
                    lstm_hidden_dim=lstm_hidden_dims,  # Now a list of hidden dimensions
                    lstm_num_layers=lstm_num_layers,
                    bidirectional=lstm_bidirectional
                )
            else:
                # For ANN/MLP, use traditional hidden layer structure
                hidden_dims = []
                if num_layers >= 1:
                    hidden_dims.append(hidden_units_layer1)
                if num_layers >= 2:
                    hidden_dims.append(hidden_units_layer2)
                if num_layers >= 3:
                    hidden_dims.append(hidden_units_layer3)
                if num_layers >= 4:
                    hidden_dims.append(hidden_units_layer4)
                # Truncate to the actual number of layers if needed
                hidden_dims = hidden_dims[:num_layers]
                
                net = create_network(
                    model_type=model_type,
                    input_dim=trial_input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=output_dim,
                    dropout=dropout,
                    network_type='ann'
                )
            
            # Create optimizer
            if optimizer_name == "Adam":
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            else:
                optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
            
            # Create model and label transformer
            if model_type.lower() == "deepsurv":
                model = CoxPH(net, optimizer=optimizer)
                labtrans = None
                # For DeepSurv, keep original continuous data
                transformed_train_data = trial_train_data
            else:
                model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
                # For DeepHit, use the pre-discretized data that was created before CV splits
                transformed_train_data = trial_train_data
                print(f"Using pre-discretized data for DeepHit training")
            
            # Move model to device
            model.net = model.net.to(device)
            
            # Initialize list to store validation metrics
            val_metrics = []
            
            # Train and evaluate on each CV fold
            for fold, (train_indices, val_indices) in enumerate(cv_splits):
                print(f"\n--- Fold {fold+1}/{len(cv_splits)} ---")
                
                # Create training and validation datasets for this fold using pre-transformed data
                fold_train_data, fold_val_data = create_cv_datasets(transformed_train_data, train_indices, val_indices)
                
                # Convert numpy arrays to torch tensors and move to device
                x_train = torch.tensor(fold_train_data[0]).float().to(device)
                x_val = torch.tensor(fold_val_data[0]).float().to(device)
                
                # Create PyTorch datasets and dataloaders
                if model_type.lower() == "deepsurv":
                    # For DeepSurv, use continuous durations and events
                    durations_train = torch.tensor(fold_train_data[1]).float()
                    events_train = torch.tensor(fold_train_data[2]).float()
                    durations_val = torch.tensor(fold_val_data[1]).float()
                    events_val = torch.tensor(fold_val_data[2]).float()
                    
                    train_tuple = (x_train, (durations_train, events_train))
                    val_tuple = (x_val, (durations_val, events_val))
                    
                    print(f"DeepSurv labels - Train: durations shape {durations_train.shape}, events shape {events_train.shape}")
                    print(f"DeepSurv labels - Val: durations shape {durations_val.shape}, events shape {events_val.shape}")
                else:
                    # For DeepHit, use pre-transformed discrete labels (already processed above)
                    train_idx_durations = torch.tensor(fold_train_data[1]).long()
                    train_events = torch.tensor(fold_train_data[2]).long()
                    val_idx_durations = torch.tensor(fold_val_data[1]).long()
                    val_events = torch.tensor(fold_val_data[2]).long()
                    
                    train_tuple = (x_train, (train_idx_durations, train_events))
                    val_tuple = (x_val, (val_idx_durations, val_events))
                    print(f"DeepHit pre-transformed labels - Train: idx_durations shape {train_idx_durations.shape}, events shape {train_events.shape}")
                    print(f"DeepHit pre-transformed labels - Val: idx_durations shape {val_idx_durations.shape}, events shape {val_events.shape}")
                    
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
                    
                    
                
                # Create early stopping callback using torchtuples implementation
                callbacks = [EarlyStopping(patience=patience)]
                
                # Log hyperparameters
                if network_type.lower() == 'lstm':
                    print(f"Training LSTM with: lr={lr}, sequence_length={trial_sequence_length}, "
                          f"lstm_hidden={lstm_hidden_dims}, lstm_layers={lstm_num_layers}, "
                          f"bidirectional={lstm_bidirectional}, dropout={dropout}, "
                          f"optimizer={optimizer_name}, batch_size={batch_size}")
                else:
                    hidden_units_str = ", ".join([
                        f"layer1={hidden_units_layer1}",
                        f"layer2={hidden_units_layer2}" if num_layers >= 2 else "",
                        f"layer3={hidden_units_layer3}" if num_layers >= 3 else "",
                        f"layer4={hidden_units_layer4}" if num_layers >= 4 else ""
                    ])
                    hidden_units_str = hidden_units_str.replace(", ,", ",").replace(", ,", ",").rstrip(", ")
                    print(f"Training ANN with: lr={lr}, layers={num_layers}, units=[{hidden_units_str}], "
                          f"dropout={dropout}, optimizer={optimizer_name}, batch_size={batch_size}")
                
                # Train the model
                epochs = 100  # Maximum number of epochs
                model.fit(
                    *train_tuple,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    val_data=val_tuple,
                    verbose=True
                )
                
                # Evaluate on validation set
                # Compute baseline hazards for CoxPH model
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
                        model.compute_baseline_hazards()
                        print("Baseline hazards computed successfully")
                
                # Get optimization metric from config
                optimization_metric = hp_config.get('optimization', {}).get('metric', 'cidx')
                
                # Use different evaluation approaches for DeepHit vs DeepSurv
                if model_type.lower() == 'deephit':
                    # For DeepHit competing risks, use specialized evaluation
                    print(f"\n=== Using competing risks evaluation for DeepHit model (fold {fold+1}/{len(cv_splits)}) ===")
                    from src.competing_risks_evaluation import evaluate_competing_risks_model
                    
                    # For DeepHit evaluation, we need original continuous durations and events
                    # Use the stored original continuous data with validation indices
                    original_durations = original_continuous_data[0][val_indices]
                    original_events = original_continuous_data[1][val_indices]
                    
                    evaluation_results = evaluate_competing_risks_model(
                        model=model,
                        x_data=x_val,
                        durations=original_durations,
                        events=original_events,
                        time_grid=time_grid,
                        optimization_metric=optimization_metric
                    )
                    
                    # Use combined metric for model selection
                    c_index = evaluation_results['combined']['metric_value']
                    print(f"Fold {fold+1} combined competing risks metric: {c_index:.4f}")
                    print(f"  - Cause 1 (RRT/eGFR<15): {evaluation_results['cause_1']['metric_value']:.4f}")
                    print(f"  - Cause 2 (Mortality): {evaluation_results['cause_2']['metric_value']:.4f}")
                    
                else:
                    # For DeepSurv, use standard survival evaluation
                    print(f"\n=== Getting survival function predictions for validation fold {fold+1}/{len(cv_splits)} ===")
                    with torch.no_grad():
                        surv = model.predict_surv_df(x_val)
                        print(f"Survival function predictions shape: {surv.shape}")
                        if not surv.empty:
                            print(f"Survival function time points: {surv.columns[:5]}...")
                            print(f"Survival function values summary:")
                            print(f"- Mean: {surv.mean().mean():.4f}")
                            print(f"- Min: {surv.min().min():.4f}")
                            print(f"- Max: {surv.max().max():.4f}")
                            print(f"- First 3 rows sample: \n{surv.iloc[:3, :5]}")
                    
                    # Create EvalSurv object
                    print(f"\n=== Creating EvalSurv object for validation fold {fold+1}/{len(cv_splits)} ===")
                    ev = EvalSurv(
                        surv,
                        durations_val.numpy(),
                        events_val.numpy(),
                        censor_surv='km'
                    )
                    print(f"EvalSurv object created successfully")
                    
                    # Calculate metrics based on configuration
                    print(f"\n=== Calculating concordance index for validation fold {fold+1}/{len(cv_splits)} ===")
                    c_index = ev.concordance_td()
                    print(f"Validation C-index: {c_index:.4f}")
                
                # Log detailed information about the data used for C-index calculation
                print(f"Data summary for C-index calculation:")
                if model_type.lower() == 'deephit':
                    # For DeepHit, use the original continuous data we extracted
                    print(f"- Number of validation samples: {len(original_durations)}")
                    print(f"- Number of validation events: {int(np.sum(original_events))}")
                    print(f"- Validation event rate: {np.mean(original_events):.2%}")
                    print(f"- Validation duration range: {np.min(original_durations):.1f} to {np.max(original_durations):.1f} days")
                else:
                    # For DeepSurv, use the continuous variables
                    print(f"- Number of validation samples: {len(durations_val)}")
                    print(f"- Number of validation events: {int(torch.sum(events_val))}")
                    print(f"- Validation event rate: {torch.mean(events_val):.2%}")
                    print(f"- Validation duration range: {torch.min(durations_val):.1f} to {torch.max(durations_val):.1f} days")
                
                # Check for potential issues in the data
                print(f"Validation data quality checks:")
                if model_type.lower() == 'deephit':
                    # For DeepHit, use the original continuous data we extracted
                    print(f"- NaN in durations: {np.isnan(original_durations).any()}")
                    print(f"- NaN in events: {np.isnan(original_events).any()}")
                    print(f"- Zero durations: {(original_durations == 0).sum()}")
                    print(f"- Negative durations: {(original_durations < 0).sum()}")
                else:
                    # For DeepSurv, use the continuous variables
                    print(f"- NaN in durations: {torch.isnan(durations_val).any()}")
                    print(f"- NaN in events: {torch.isnan(events_val).any()}")
                    print(f"- Zero durations: {(durations_val == 0).sum()}")
                    print(f"- Negative durations: {(durations_val < 0).sum()}")
                
                # Calculate additional metrics only for DeepSurv models
                # For DeepHit, we already have the combined metric from competing risks evaluation
                if model_type.lower() != 'deephit':
                    # Calculate integrated Brier score
                    # time_grid should always be available now
                    ibs = ev.integrated_brier_score(time_grid)
                    print(f"Validation Integrated Brier Score: {ibs:.4f}")
                    
                    # Calculate integrated log-likelihood
                    loglik = ev.integrated_nbll(time_grid)
                    print(f"Validation Integrated Log-likelihood: {loglik:.4f}")
                    
                    # Determine which metric to use for optimization
                    if optimization_metric == 'brs' and ibs is not None:
                        # For Brier score, lower is better, so we negate it for maximization
                        metric_value = -ibs
                        print(f"Using Integrated Brier Score for optimization: {-metric_value:.4f}")
                    elif optimization_metric == 'loglik' and loglik is not None:
                        # For log-likelihood, lower is better, so we negate it for maximization
                        metric_value = -loglik
                        print(f"Using Integrated Log-likelihood for optimization: {-metric_value:.4f}")
                    else:
                        # Default to concordance index (higher is better)
                        metric_value = c_index
                        print(f"Using Concordance Index for optimization: {metric_value:.4f}")
                else:
                    # For DeepHit, use the combined competing risks metric
                    metric_value = c_index
                    print(f"Using Combined Competing Risks Metric for optimization: {metric_value:.4f}")
                
                # Store validation metric
                val_metrics.append(metric_value)
            
            # Calculate mean validation metric across folds
            mean_val_metric = np.mean(val_metrics)
            print(f"\nMean validation C-index across {len(cv_splits)} folds: {mean_val_metric:.4f}")
            
            return mean_val_metric
        
        # Create Optuna study
        print(f"\n=== Starting hyperparameter optimization with {n_trials} trials ===")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        # Get best hyperparameters
        best_params = study.best_params
        best_value = study.best_value
        print(f"\nBest hyperparameters: {best_params}")
        print(f"Best validation C-index: {best_value:.4f}")
        
        # Train final model with best hyperparameters
        print("\n=== Training final model with best hyperparameters ===")
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Extract hyperparameters
        lr = best_params["learning_rate"]
        num_layers = best_params["num_layers"]
        # Layer-specific hidden units are accessed directly in the network creation code
        dropout = best_params["dropout"]
        optimizer_name = best_params["optimizer"]
        batch_size = best_params["batch_size"]
        
        # Model-specific hyperparameters
        if model_type.lower() == "deephit":
            alpha = best_params["alpha"]
            sigma = best_params["sigma"]
            output_dim = len(time_grid)
        else:
            alpha = None
            sigma = None
            output_dim = 1
        
        # Build network architecture based on the optimized network type
        if network_type.lower() == 'lstm':
            # Extract LSTM hyperparameters from best_params
            lstm_num_layers = best_params.get("lstm_num_layers", 1)
            lstm_bidirectional = best_params.get("bidirectional", False)
            lstm_sequence_length = best_params.get("sequence", sequence_length)
            
            # Build LSTM hidden dimensions list
            lstm_hidden_dims = []
            for i in range(lstm_num_layers):
                layer_key = f"lstm_hidden_dim_layer{i+1}"
                if layer_key in best_params:
                    lstm_hidden_dims.append(best_params[layer_key])
            
            # Use sequence data for final training if LSTM
            final_input_dim = input_dim  # This should be features per timestep for LSTM
            final_sequence_length = lstm_sequence_length
            
            net = create_network(
                model_type=model_type,
                input_dim=final_input_dim,
                output_dim=output_dim,
                dropout=dropout,
                network_type='lstm',
                sequence_length=final_sequence_length,
                lstm_hidden_dim=lstm_hidden_dims,
                lstm_num_layers=lstm_num_layers,
                bidirectional=lstm_bidirectional
            )
        else:
            # Build ANN network architecture with different units per layer
            hidden_dims = []
            if num_layers >= 1:
                hidden_dims.append(best_params["hidden_units_layer1"])
            if num_layers >= 2:
                hidden_dims.append(best_params["hidden_units_layer2"])
            if num_layers >= 3:
                hidden_dims.append(best_params["hidden_units_layer3"])
            if num_layers >= 4:
                hidden_dims.append(best_params["hidden_units_layer4"])
            # Truncate to the actual number of layers if needed
            hidden_dims = hidden_dims[:num_layers]
            
            net = create_network(
                model_type=model_type,
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                dropout=dropout,
                network_type='ann'
            )
        
        # Create optimizer
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        else:
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
        
        # Create model and label transformer
        if model_type.lower() == "deepsurv":
            final_model = CoxPH(net, optimizer=optimizer)
            final_labtrans = None
        else:
            final_model = DeepHit(net, optimizer=optimizer, alpha=alpha, sigma=sigma, duration_index=time_grid)
            # Create label transformer for DeepHit
            final_labtrans = LabTransDiscreteTime(time_grid)
            # Fit the transformer on the training data
            final_labtrans.fit(train_data[1], train_data[2])
            print(f"Created final LabTransDiscreteTime with {len(time_grid)} time points")
        
        # Move model to device
        final_model.net = final_model.net.to(device)
        
        # Convert numpy arrays to torch tensors and move to device
        x_train = torch.tensor(train_data[0]).float().to(device)
        
        # Create PyTorch datasets and dataloaders
        if model_type.lower() == "deepsurv":
            durations_train = torch.tensor(train_data[1]).float()
            events_train = torch.tensor(train_data[2]).float()
            train_tuple = (x_train, (durations_train, events_train))
        else:
            # For DeepHit, transform the labels using LabTransDiscreteTime
            # Ensure proper data types before transformation
            final_durations = np.asarray(train_data[1], dtype=np.float64)
            final_events = np.asarray(train_data[2], dtype=np.float64)
            
            # Clip durations to be within time grid bounds to avoid discretization errors
            time_grid_array = np.array(time_grid)
            time_grid_min, time_grid_max = time_grid_array.min(), time_grid_array.max()
            final_durations = np.clip(final_durations, time_grid_min, time_grid_max)
            
            print(f"Final model - Data types before transform: durations {final_durations.dtype}, events {final_events.dtype}")
            print(f"Final model - Duration bounds: Min: {time_grid_min}, Max: {time_grid_max}")
            
            final_target = final_labtrans.transform(final_durations, final_events)
            
            # Convert to tensors
            train_idx_durations = torch.tensor(final_target[0]).long()
            train_events = torch.tensor(final_target[1]).long()
            
            train_tuple = (x_train, (train_idx_durations, train_events))
            
            print(f"Final model - Transformed labels: idx_durations shape {train_idx_durations.shape}, events shape {train_events.shape}")
        
        # Train the final model
        epochs = 100  # Maximum number of epochs
        
        # Create early stopping callback using torchtuples implementation
        callbacks = [EarlyStopping(patience=patience)]
        
        # Log hyperparameters based on network type
        if network_type.lower() == 'lstm':
            lstm_num_layers = best_params.get("lstm_num_layers", 1)
            lstm_hidden_dims = [best_params.get(f"lstm_hidden_dim_layer{i+1}") for i in range(lstm_num_layers)]
            lstm_bidirectional = best_params.get("bidirectional", False)
            lstm_sequence_length = best_params.get("sequence", sequence_length)
            
            print(f"Training final LSTM model with: lr={lr}, sequence_length={lstm_sequence_length}, "
                  f"lstm_hidden={lstm_hidden_dims}, lstm_layers={lstm_num_layers}, "
                  f"bidirectional={lstm_bidirectional}, dropout={dropout}, "
                  f"optimizer={optimizer_name}, batch_size={batch_size}")
        else:
            hidden_units_str = ", ".join([
                f"layer{i+1}={best_params[f'hidden_units_layer{i+1}']}"
                for i in range(min(4, num_layers))
            ])
            print(f"Training final ANN model with: lr={lr}, layers={num_layers}, "
                  f"units=[{hidden_units_str}], dropout={dropout}, "
                  f"optimizer={optimizer_name}, batch_size={batch_size}")
        
        # Train the model
        final_model.fit(
            *train_tuple,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=True
        )
        
        # Compute baseline hazards for CoxPH model
        if model_type.lower() == "deepsurv":
            print("Computing baseline hazards for final CoxPH model")
            # Check if any values are None
            if durations_train is None or events_train is None:
                print(f"Warning: durations or events are None. durations: {durations_train is None}, events: {events_train is None}")
                # Skip computing baseline hazards and use model directly
                print("Skipping baseline hazards computation due to None values")
            else:
                # Set the training data for the model
                final_model.training_data = (x_train, (durations_train, events_train))
                # Compute baseline hazards
                final_model.compute_baseline_hazards()
        
        # Evaluate on test sets
        print("\n=== Evaluating final model on test sets ===")
        
        # Function to evaluate on a dataset
        def evaluate_dataset(dataset, name):
            if dataset is None:
                print(f"No {name} dataset provided for evaluation")
                return None
            
            # Convert numpy arrays to torch tensors and move to device
            x_test = torch.tensor(dataset[0]).float().to(device)
            durations_test = torch.tensor(dataset[1]).float()
            events_test = torch.tensor(dataset[2]).float()
            
            # Get survival function predictions
            with torch.no_grad():
                surv = final_model.predict_surv_df(x_test)
            
            # Create EvalSurv object
            ev = EvalSurv(
                surv,
                durations_test.numpy(),
                events_test.numpy(),
                censor_surv='km'
            )
            
            # Calculate concordance index
            c_index = ev.concordance_td()
            print(f"{name} C-index: {c_index:.4f}")
            
            # Calculate integrated Brier score
            # time_grid should always be available now
            ibs = ev.integrated_brier_score(time_grid)
            print(f"{name} Integrated Brier Score: {ibs:.4f}")
            
            return {
                "c_index": float(c_index),
                "integrated_brier_score": float(ibs) if ibs is not None else None
            }
        
        # We don't have test datasets in this function anymore
        # Evaluation will be done in a separate step
        temporal_metrics = None
        spatial_metrics = None
        
        print("\n=== Note: Test set evaluation will be performed in a separate step ===")
        
        # Create results directory if it doesn't exist
        results_dir = "/mnt/dump/yard/projects/tarot2/results/model_details"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save model weights
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(results_dir, f"model_weights_{timestamp}.pt")
        torch.save(final_model.net.state_dict(), model_path)
        print(f"Saved model weights to {model_path}")
        
        # Create model details dictionary based on network type
        if network_type.lower() == 'lstm':
            # For LSTM, include LSTM-specific parameters
            lstm_num_layers = best_params.get("lstm_num_layers", 1)
            lstm_hidden_dims = [best_params.get(f"lstm_hidden_dim_layer{i+1}") for i in range(lstm_num_layers)]
            lstm_bidirectional = best_params.get("bidirectional", False)
            lstm_sequence_length = best_params.get("sequence", sequence_length)
            
            model_details = {
                "model_type": model_type,
                "network_type": "lstm",
                "input_dim": input_dim,
                "lstm_hidden_dims": lstm_hidden_dims,
                "lstm_num_layers": lstm_num_layers,
                "lstm_bidirectional": lstm_bidirectional,
                "sequence_length": lstm_sequence_length,
                "output_dim": output_dim,
                "dropout": dropout,
                "time_grid": time_grid.tolist() if time_grid is not None else None,
                "alpha": alpha,
                "sigma": sigma,
                "model_path": model_path,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            # For ANN, include traditional hidden dimensions
            model_details = {
                "model_type": model_type,
                "network_type": "ann",
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,
                "num_layers": num_layers,
                "output_dim": output_dim,
                "dropout": dropout,
                "time_grid": time_grid.tolist() if time_grid is not None else None,
                "alpha": alpha,
                "sigma": sigma,
                "model_path": model_path,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Get optimization metric from config
        optimization_metric = hp_config.get('optimization', {}).get('metric', 'cidx')
        
        # Create optimization metrics dictionary
        optimization_metrics = {
            "best_params": best_params,
            "best_value": float(best_value),
            "n_trials": n_trials,
            "optimization_metric": optimization_metric,
            "cv_metrics": {
                "mean_c_index": float(best_value) if optimization_metric == 'cidx' else None,
                "mean_brier_score": -float(best_value) if optimization_metric == 'brs' else None,
                "mean_log_likelihood": -float(best_value) if optimization_metric == 'loglik' else None,
                "fold_metrics": [float(t.value) if t.value is not None else None for t in study.trials[:cv_folds]]
            },
            "study_trials": [
                {
                    "number": t.number,
                    "params": t.params,
                    "value": float(t.value) if t.value is not None else None
                }
                for t in study.trials
            ]
        }
        
        # Save model details and optimization metrics as JSON files
        
        model_details_path = os.path.join(results_dir, f"model_details_{timestamp}.json")
        optimization_metrics_path = os.path.join(results_dir, f"optimization_metrics_{timestamp}.json")
        
        with open(model_details_path, 'w') as f:
            json.dump(model_details, f, indent=4)
        print(f"Saved model details to {model_details_path}")
        
        with open(optimization_metrics_path, 'w') as f:
            json.dump(optimization_metrics, f, indent=4)
        print(f"Saved optimization metrics to {optimization_metrics_path}")
        
        # Log to MLflow
        print("\n=== Logging to MLflow ===")
        
        # Check if there's an active MLflow run (managed by ZenML)
        active_run = mlflow.active_run()
        if active_run:
            print(f"Using existing MLflow run: {active_run.info.run_id}")
        
        # Log model parameters based on network type
        if network_type.lower() == 'lstm':
            # For LSTM, log LSTM-specific parameters
            lstm_num_layers = best_params.get("lstm_num_layers", 1)
            lstm_hidden_dims = [best_params.get(f"lstm_hidden_dim_layer{i+1}") for i in range(lstm_num_layers)]
            lstm_bidirectional = best_params.get("bidirectional", False)
            lstm_sequence_length = best_params.get("sequence", sequence_length)
            
            mlflow.log_params({
                "model_type": model_type,
                "network_type": "lstm",
                "input_dim": input_dim,
                "lstm_hidden_dims": str(lstm_hidden_dims),
                "lstm_num_layers": lstm_num_layers,
                "lstm_bidirectional": lstm_bidirectional,
                "sequence_length": lstm_sequence_length,
                "output_dim": output_dim,
                "dropout": dropout,
                "alpha": alpha if alpha is not None else "None",
                "sigma": sigma if sigma is not None else "None",
            })
        else:
            # For ANN, log traditional hidden dimensions
            mlflow.log_params({
                "model_type": model_type,
                "network_type": "ann",
                "input_dim": input_dim,
                "hidden_dims": str(hidden_dims),
                "num_layers": num_layers,
                "output_dim": output_dim,
                "dropout": dropout,
                "alpha": alpha if alpha is not None else "None",
                "sigma": sigma if sigma is not None else "None",
            })
        
        # Log balancing parameters and metrics
        mlflow.log_params({
            "balance_enabled": balance_cfg.get("enable", False),
            "balance_method": balance_cfg.get("method", "none"),
            "balance_strategy": balance_cfg.get("sampling_strategy", "majority"),
            "rows_before_balance": orig_rows,
            "rows_after_balance": balanced_rows,
            "rows_removed_pct": removed_pct,
        })
        
        # Log event distribution before and after balancing
        if balance_cfg.get("enable", False):
            # Get event distribution after balancing
            event_counts = train_df_preprocessed['endpoint'].value_counts().to_dict()
            for event_type, count in event_counts.items():
                mlflow.log_metric(f"event_{event_type}_count_after", count)
                mlflow.log_metric(f"event_{event_type}_pct_after", (count / balanced_rows) * 100)
        
        # Log best hyperparameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)
        
        # Log metrics
        # Get optimization metric from config
        optimization_metric = hp_config.get('optimization', {}).get('metric', 'cidx')
        
        # Log the appropriate metric based on what was optimized
        if optimization_metric == 'brs':
            mlflow.log_metric("best_brier_score", -float(best_value))
            mlflow.log_param("optimization_metric", "Integrated Brier Score")
        elif optimization_metric == 'loglik':
            mlflow.log_metric("best_log_likelihood", -float(best_value))
            mlflow.log_param("optimization_metric", "Integrated Log-likelihood")
        else:
            mlflow.log_metric("best_c_index", float(best_value))
            mlflow.log_param("optimization_metric", "Concordance Index")
        
        # Log additional metrics if available
        for fold_idx, metric_value in enumerate([float(t.value) if t.value is not None else 0.0
                                               for t in study.trials[:cv_folds]]):
            if metric_value is not None:
                if optimization_metric == 'brs':
                    mlflow.log_metric(f"fold_{fold_idx+1}_brier_score", -metric_value)
                elif optimization_metric == 'loglik':
                    mlflow.log_metric(f"fold_{fold_idx+1}_log_likelihood", -metric_value)
                else:
                    mlflow.log_metric(f"fold_{fold_idx+1}_c_index", metric_value)
        
        # Log model artifacts
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(model_details_path)
        mlflow.log_artifact(optimization_metrics_path)
        
        # Log the PyTorch model with input example for auto-inferring model signature
        # Create a small input example with the correct shape
        input_example = torch.zeros((1, input_dim), dtype=torch.float32)
        
        # Convert PyTorch tensor to numpy array for MLflow compatibility
        # MLflow doesn't accept PyTorch tensors directly as input examples
        numpy_input_example = input_example.numpy()
        
        # Log the model with input example
        mlflow.pytorch.log_model(
            final_model.net,
            "pytorch_model",
            input_example=numpy_input_example
        )
        
        print(f"Logged model details, parameters, and metrics to MLflow")
        
        return model_details, optimization_metrics
        
    except Exception as e:
        print(f"Error in train_model step: {e}")
        traceback.print_exc()
        raise


def extract_dataset(dataset: Any, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract dataset from ZenML StepArtifact.
    
    Args:
        dataset: Dataset or ZenML StepArtifact
        name: Name of the dataset for logging
        
    Returns:
        Tuple of (X, durations, events) arrays
    """
    print(f"Extracting {name} (type: {type(dataset).__name__})")
    
    # If dataset is already a tuple, return it directly
    if isinstance(dataset, tuple) and len(dataset) >= 3:
        print(f"{name} is already a tuple of length {len(dataset)}")
        return dataset
    
    # Try to access the raw data directly
    raw_data = access_zenml_artifact_data(dataset)
    if raw_data is not None and isinstance(raw_data, tuple) and len(raw_data) >= 3:
        print(f"Successfully accessed raw tuple data from {name}")
        return raw_data
    
    # Try standard extraction
    extracted_data = extract_from_step_artifact(
        dataset,
        expected_type=tuple,
        artifact_name=name
    )
    
    if extracted_data is not None and isinstance(extracted_data, tuple) and len(extracted_data) >= 3:
        print(f"Successfully extracted tuple data from {name}")
        return extracted_data
    
    raise ValueError(f"Failed to extract valid dataset from {name}")


def validate_dataset(dataset: Tuple[np.ndarray, np.ndarray, np.ndarray], name: str) -> None:
    """
    Validate dataset structure and contents.
    
    Args:
        dataset: Tuple of (X, durations, events) arrays
        name: Name of the dataset for logging
        
    Raises:
        ValueError: If dataset is invalid
    """
    if not isinstance(dataset, tuple) or len(dataset) < 3:
        raise ValueError(f"{name} must be a tuple of (X, durations, events)")
    
    X, durations, events = dataset
    
    # Check shapes
    print(f"{name} shapes: X={X.shape}, durations={durations.shape}, events={events.shape}")
    
    if len(X) != len(durations) or len(X) != len(events):
        raise ValueError(f"{name} has inconsistent lengths: X={len(X)}, durations={len(durations)}, events={len(events)}")
    
    # Check for NaN or Inf values - safely handle mixed data types
    try:
        # Check X for NaN values - only for numeric columns
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.number):
            if np.isnan(X).any():
                print(f"Warning: {name} X contains NaN values")
            if np.isinf(X).any():
                print(f"Warning: {name} X contains Inf values")
        else:
            # For mixed data types or object arrays, we need a different approach
            print(f"Note: {name} X contains non-numeric data (dtype: {X.dtype}), skipping NaN/Inf check")
            
        # Check durations for NaN values
        if np.isnan(durations).any():
            print(f"Warning: {name} durations contains NaN values")
        if np.isinf(durations).any():
            print(f"Warning: {name} durations contains Inf values")
            
        # Check events for NaN values
        if np.isnan(events).any():
            print(f"Warning: {name} events contains NaN values")
        if np.isinf(events).any():
            print(f"Warning: {name} events contains Inf values")
    except TypeError as e:
        print(f"Warning: Could not check for NaN/Inf values in {name}: {e}")
        print(f"X dtype: {X.dtype}, durations dtype: {durations.dtype}, events dtype: {events.dtype}")
    
    # Print event distribution
    try:
        event_counts = np.bincount(events.astype(int))
        print(f"{name} event distribution: {event_counts}")
        print(f"{name} event rate: {events.mean():.2%}")
    except Exception as e:
        print(f"Warning: Could not compute event distribution for {name}: {e}")