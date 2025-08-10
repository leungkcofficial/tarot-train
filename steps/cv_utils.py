"""
Cross-Validation Utilities for CKD Risk Prediction

This module provides functions for cross-validation in survival analysis,
particularly time-based cross-validation with patient grouping.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union


def time_based_patient_cv(
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray],
    patient_ids: Union[np.ndarray, List],
    entry_dates: Union[np.ndarray, List],
    cv_folds: int = 5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Perform time-based cross-validation with patient grouping.
    
    Args:
        dataset: Tuple of (X, durations, events) arrays
        patient_ids: Array or list of patient IDs corresponding to each row in X
        entry_dates: Array or list of patient entry dates (datetime objects or strings)
        cv_folds: Number of cross-validation folds (default: 5)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Convert to numpy arrays if they are lists
    if not isinstance(patient_ids, np.ndarray):
        patient_ids = np.array(patient_ids)
    
    # Create a DataFrame with patient IDs and entry dates
    patient_entry_df = pd.DataFrame({
        'patient_id': patient_ids,
        'entry_date': pd.to_datetime(entry_dates)
    }).drop_duplicates().sort_values('entry_date')
    
    # Get date range for patient entries
    min_entry_date = patient_entry_df['entry_date'].min()
    max_entry_date = patient_entry_df['entry_date'].max()
    entry_date_range = (max_entry_date - min_entry_date).days
    
    # Create folds based on patient entry dates
    fold_size = entry_date_range / cv_folds
    
    # Initialize list to store fold indices
    cv_splits = []
    
    # For each fold, train on patients who entered before fold_end and validate on patients in the next fold
    for fold in range(cv_folds - 1):  # Last fold is used only for validation
        # Calculate fold boundaries
        train_end_day = int((fold + 1) * fold_size)
        val_end_day = int((fold + 2) * fold_size)
        
        train_end_date = min_entry_date + pd.Timedelta(days=train_end_day)
        val_end_date = min_entry_date + pd.Timedelta(days=val_end_day)
        
        # Get patients for training and validation
        train_patients = patient_entry_df[patient_entry_df['entry_date'] <= train_end_date]['patient_id'].tolist()
        val_patients = patient_entry_df[(patient_entry_df['entry_date'] > train_end_date) &
                                       (patient_entry_df['entry_date'] <= val_end_date)]['patient_id'].tolist()
        
        # Create masks based on patient IDs
        train_mask = np.isin(patient_ids, train_patients)
        val_mask = np.isin(patient_ids, val_patients)
        
        # Get indices
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        
        # Check if we have enough data
        if len(train_indices) > 0 and len(val_indices) > 0:
            cv_splits.append((train_indices, val_indices))
    
    # If no valid folds were found or cv_folds is 1, use a simple temporal split
    if len(cv_splits) == 0:
        # Sort patients by entry date
        cutoff_idx = int(len(patient_entry_df) * 0.8)
        train_patients = patient_entry_df.iloc[:cutoff_idx]['patient_id'].tolist()
        val_patients = patient_entry_df.iloc[cutoff_idx:]['patient_id'].tolist()
        
        # Create masks based on patient IDs
        train_mask = np.isin(patient_ids, train_patients)
        val_mask = np.isin(patient_ids, val_patients)
        
        # Get indices
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        
        cv_splits.append((train_indices, val_indices))
    
    return cv_splits


def create_cv_datasets(
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray],
    train_indices: np.ndarray,
    val_indices: np.ndarray
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create training and validation datasets from indices.
    
    Args:
        dataset: Tuple of (X, durations, events) arrays
        train_indices: Indices for training set
        val_indices: Indices for validation set
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    X, durations, events = dataset
    
    # Validate indices compatibility with data size
    data_size = X.shape[0]
    max_train_idx = np.max(train_indices) if len(train_indices) > 0 else -1
    max_val_idx = np.max(val_indices) if len(val_indices) > 0 else -1
    
    if max_train_idx >= data_size:
        raise IndexError(f"Train index {max_train_idx} is out of bounds for axis 0 with size {data_size}")
    
    if max_val_idx >= data_size:
        raise IndexError(f"Val index {max_val_idx} is out of bounds for axis 0 with size {data_size}")
    
    # Create training dataset
    X_train = X[train_indices]
    durations_train = durations[train_indices]
    events_train = events[train_indices]
    train_dataset = (X_train, durations_train, events_train)
    
    # Create validation dataset
    X_val = X[val_indices]
    durations_val = durations[val_indices]
    events_val = events[val_indices]
    val_dataset = (X_val, durations_val, events_val)
    
    return train_dataset, val_dataset


def create_random_cv_splits(
    dataset: Any,
    n_splits: int = 5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create random cross-validation splits.
    
    Args:
        dataset: Tuple of (X, durations, events) arrays or ZenML StepArtifact
        n_splits: Number of cross-validation folds (default: 5)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    # Handle ZenML StepArtifact
    print(f"dataset type in create_random_cv_splits: {type(dataset)}")
    if hasattr(dataset, 'dir'):
        print(f"dataset attributes: {dir(dataset)}")
    
    import traceback
    from src.util import extract_from_step_artifact, access_zenml_artifact_data
    
    # Try to access common attributes for debugging
    try:
        print("DEBUG: Inspecting dataset attributes")
        for attr in ['data', '_data', 'value', '_value', 'artifact_data', 'materialize', 'read']:
            if hasattr(dataset, attr):
                attr_value = getattr(dataset, attr)
                print(f"DEBUG: dataset.{attr} type: {type(attr_value)}")
                if callable(attr_value):
                    print(f"DEBUG: {attr} is callable")
    except Exception as e:
        print(f"DEBUG: Error inspecting dataset: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
    
    # First try direct access to the raw data
    print("DEBUG: Attempting direct access to raw data with access_zenml_artifact_data")
    raw_data = access_zenml_artifact_data(dataset)
    if raw_data is not None:
        print(f"DEBUG: Raw data type: {type(raw_data)}")
        if isinstance(raw_data, tuple):
            print(f"Successfully accessed raw tuple data from ZenML artifact")
            print(f"Raw data is a tuple of length {len(raw_data)}")
            for i, item in enumerate(raw_data):
                print(f"Item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'unknown')}")
                # Check for NaN values in numeric arrays
                if hasattr(item, 'dtype') and np.issubdtype(item.dtype, np.number):
                    print(f"  Contains NaN: {np.isnan(item).any()}")
                    print(f"  Contains Inf: {np.isinf(item).any()}")
                    if np.isnan(item).any() or np.isinf(item).any():
                        print(f"DEBUG: Found problematic values in item {i}")
                        nan_count = np.isnan(item).sum() if hasattr(np.isnan(item), 'sum') else 'unknown'
                        inf_count = np.isinf(item).sum() if hasattr(np.isinf(item), 'sum') else 'unknown'
                        print(f"DEBUG: NaN count: {nan_count}, Inf count: {inf_count}")
            dataset = raw_data
        else:
            print(f"DEBUG: Raw data is not a tuple, type: {type(raw_data)}")
    else:
        print("DEBUG: Direct access returned None")
        
    if raw_data is not None and isinstance(raw_data, tuple) and len(raw_data) > 0:
        print("Successfully accessed raw tuple data from ZenML artifact")
        dataset = raw_data
    else:
        print("Direct access didn't return a valid tuple, trying standard extraction")
        # Try to extract the data using standard methods
        print("DEBUG: Attempting extraction with extract_from_step_artifact")
        dataset = extract_from_step_artifact(
            dataset,
            expected_type=tuple,  # Explicitly specify we expect a tuple
            artifact_name="dataset"
        )
        print(f"DEBUG: After extraction, dataset type: {type(dataset)}")
        
    # If dataset is still a StepArtifact, try one more approach
    if 'StepArtifact' in str(type(dataset)):
        print("Dataset is still a StepArtifact after extraction attempts")
        try:
            # Try to access the artifact directly as a tuple
            if hasattr(dataset, '__getitem__'):
                print("Trying to access StepArtifact as a tuple-like object")
                # Check if it has a length
                if hasattr(dataset, '__len__'):
                    length = len(dataset)
                    print(f"StepArtifact has length {length}")
                    
                    # Try to extract the first few items
                    try:
                        items = [dataset[i] for i in range(min(3, length))]
                        print(f"Successfully extracted {len(items)} items from StepArtifact")
                        
                        # If it looks like a tuple of arrays, convert it
                        if all(hasattr(item, 'shape') for item in items):
                            print("Items appear to be arrays, converting to tuple")
                            dataset = tuple(dataset[i] for i in range(length))
                            print(f"Successfully converted to tuple of length {len(dataset)}")
                    except Exception as e:
                        print(f"Error extracting items from StepArtifact: {e}")
                        
                        # Try a different approach - maybe it's a single-item container
                        try:
                            print("Trying to access as a single-item container")
                            single_item = dataset[0]
                            print(f"Successfully accessed single item of type: {type(single_item).__name__}")
                            
                            # If it's a tuple, use it directly
                            if isinstance(single_item, tuple) and len(single_item) >= 3:
                                print(f"Single item is a tuple of length {len(single_item)}, using directly")
                                dataset = single_item
                            # Otherwise, wrap it in a tuple if it has a shape (likely an array)
                            elif hasattr(single_item, 'shape'):
                                print(f"Single item has shape {single_item.shape}, wrapping in tuple")
                                dataset = (single_item,)
                        except Exception as e2:
                            print(f"Error accessing as single-item container: {e2}")
        except Exception as e:
            print(f"Error accessing StepArtifact as tuple-like object: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    # Print information about extracted data
    print(f"Extracted dataset type: {type(dataset)}")
    if isinstance(dataset, tuple):
        print(f"dataset is a tuple of length {len(dataset)}")
        for i, item in enumerate(dataset):
            print(f"Item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'unknown')}")
            # Check for NaN values in numeric arrays
            if hasattr(item, 'dtype') and np.issubdtype(item.dtype, np.number):
                print(f"Item {i} contains NaN: {np.isnan(item).any()}")
                print(f"Item {i} contains Inf: {np.isinf(item).any()}")
                if np.isnan(item).any() or np.isinf(item).any():
                    print(f"DEBUG: Found problematic values in item {i}")
                    nan_count = np.isnan(item).sum() if hasattr(np.isnan(item), 'sum') else 'unknown'
                    inf_count = np.isinf(item).sum() if hasattr(np.isinf(item), 'sum') else 'unknown'
                    print(f"DEBUG: NaN count: {nan_count}, Inf count: {inf_count}")
    else:
        print(f"Warning: Extracted dataset is not a tuple, type: {type(dataset)}")
        print(f"DEBUG: Dataset attributes: {dir(dataset)[:20]}")
    
    # If extraction failed or returned None, use a default approach
    if dataset is None:
        print("Warning: Could not extract data from StepArtifact")
        # Use a default number of samples
        n_samples = 100
        np.random.seed(seed)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Create splits
        fold_size = n_samples // n_splits
        cv_splits = []
        
        for i in range(n_splits - 1):
            val_indices = indices[i*fold_size:(i+1)*fold_size]
            train_indices = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
            cv_splits.append((train_indices, val_indices))
            
        return cv_splits
    
    # Try to get the number of samples
    try:
        n_samples = len(dataset[0])
        print(f"DEBUG: Successfully determined n_samples={n_samples} from dataset[0]")
    except (TypeError, IndexError) as e:
        print(f"Warning: Could not determine number of samples from dataset: {e}")
        print(f"DEBUG: Error traceback: {traceback.format_exc()}")
        # Use a default number of samples
        n_samples = 100
        print(f"DEBUG: Using default n_samples={n_samples}")
    
    np.random.seed(seed)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Create splits
    fold_size = n_samples // n_splits
    cv_splits = []
    
    for i in range(n_splits - 1):
        val_indices = indices[i*fold_size:(i+1)*fold_size]
        train_indices = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
        cv_splits.append((train_indices, val_indices))
        
    return cv_splits