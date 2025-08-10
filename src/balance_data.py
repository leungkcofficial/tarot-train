"""
Class Imbalance Handling for Deep Learning Survival Models

This module provides functions for handling class imbalance in survival analysis datasets
through under-sampling of the majority class.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

# Import imbalanced-learn classes
from imblearn.under_sampling import (
    RandomUnderSampler,
    NearMiss,
    AllKNN,
    TomekLinks
)

# Define allowed under-sampling methods
ALLOWED_METHODS = {
    "random_under_sampler": RandomUnderSampler,
    "near_miss": NearMiss,
    "enn": AllKNN,
    "tomek_links": TomekLinks,
}

def balance_dataframe(
    df: pd.DataFrame,
    duration_col: str = "duration",
    event_col: str = "endpoint",
    method: str = "random_under_sampler",
    sampling_strategy: str = "majority",
    model_type: str = "deepsurv",
    target_endpoint: Optional[int] = None,
    near_miss_version: int = 1,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Balance a dataframe by under-sampling the majority class.
    
    Args:
        df: Input dataframe
        duration_col: Name of the duration column (default: "duration")
        event_col: Name of the event column (default: "endpoint")
        method: Under-sampling method to use (default: "random_under_sampler")
            Options: "random_under_sampler", "near_miss", "cluster_centroids", "tomek_links"
        sampling_strategy: Sampling strategy (default: "majority")
            Fixed to "majority" to ensure no synthetic rows are created
        model_type: Type of model being trained (default: "deepsurv")
            Options: "deepsurv" (binary), "deephit" (multi-class)
        target_endpoint: For DeepSurv, the specific endpoint to focus on (default: None)
            If specified, only this endpoint will be considered as the minority class
        near_miss_version: Version of NearMiss algorithm to use (default: 1)
            Only used when method="near_miss"
        feature_cols: List of feature columns to include (default: None)
            If None, all columns except duration_col and event_col will be used
            
    Returns:
        Balanced dataframe with only rows removed (no new rows added)
    """
    # Validate method
    if method not in ALLOWED_METHODS:
        raise ValueError(
            f"balance.method '{method}' not supported. "
            f"Choose one of {list(ALLOWED_METHODS.keys())}."
        )
    
    # Validate sampling_strategy
    if sampling_strategy != "majority":
        raise ValueError(
            f"sampling_strategy must be 'majority' to ensure no synthetic rows are created. "
            f"Got '{sampling_strategy}' instead."
        )
    
    # Store original row count
    orig_rows = len(df)
    
    # Get event distribution
    event_counts = df[event_col].value_counts()
    print(f"Original event distribution: {event_counts.to_dict()}")
    
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Create a new column with a unique identifier for each row
    # Reset the index first to ensure it's contiguous, then use index values as row_id
    df_copy = df_copy.reset_index(drop=True)
    df_copy['row_id'] = df_copy.index
    
    # Handle different model types
    binary_event_col = None
    if model_type.lower() == "deepsurv" and target_endpoint is not None:
        # For DeepSurv with target_endpoint, we need to convert to binary classification
        print(f"Converting to binary classification for target endpoint: {target_endpoint}")
        
        # Create a binary version of the event column
        binary_event_col = f"{event_col}_binary"
        df_copy[binary_event_col] = (df_copy[event_col] == target_endpoint).astype(int)
        
        # Use the binary event column for balancing
        event_col_for_balancing = binary_event_col
        
        # Get binary event distribution
        binary_event_counts = df_copy[binary_event_col].value_counts()
        print(f"Binary event distribution: {binary_event_counts.to_dict()}")
    else:
        # Use the original event column
        event_col_for_balancing = event_col
    
    # Determine which columns to use as features
    if feature_cols is None:
        # If no specific features are provided, use all columns except event, duration, and row_id
        feature_cols = [col for col in df_copy.columns if col != event_col and col != duration_col and col != 'row_id']
        if binary_event_col is not None and binary_event_col in df_copy.columns:
            feature_cols = [col for col in feature_cols if col != binary_event_col]
    else:
        # Use only the specified feature columns that exist in the dataframe
        feature_cols = [col for col in feature_cols if col in df_copy.columns]
        print(f"Using {len(feature_cols)} features from master dataframe mapping: {feature_cols}")
    
    # Filter out datetime columns that would cause issues with the undersampling algorithms
    numeric_feature_cols = []
    datetime_cols = []
    for col in feature_cols:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            print(f"WARNING: Excluding datetime column from balancing: {col} (type: {df_copy[col].dtype})")
            datetime_cols.append(col)
        else:
            numeric_feature_cols.append(col)
    
    if len(numeric_feature_cols) < len(feature_cols):
        print(f"Excluded {len(feature_cols) - len(numeric_feature_cols)} datetime columns from balancing")
    
    # Create X (numeric features + duration + row_id) and y (event)
    X_cols = numeric_feature_cols + [duration_col, 'row_id']
    X = df_copy[X_cols]
    y = df_copy[event_col_for_balancing]
    
    # Check for non-numeric columns and handle NaN values
    for col in X.columns:
        # Skip row_id column for NaN handling
        if col == 'row_id':
            continue
            
        # Check for non-numeric columns
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"WARNING: Converting non-numeric column to numeric: {col} (type: {X[col].dtype})")
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Check for NaN values
        if X[col].isna().any():
            print(f"WARNING: NaN values detected in column: {col} (count: {X[col].isna().sum()})")
            if X[col].notna().any():
                # Fill NaN with column mean if there are non-NaN values
                mean_val = X[col].mean()
                X[col] = X[col].fillna(mean_val)
                print(f"  Filled NaN values in column {col} with mean: {mean_val}")
            else:
                # Fill with 0 if all values are NaN
                X[col] = X[col].fillna(0)
                print(f"  Filled NaN values in column {col} with 0")
    
    # Initialize under-sampler with appropriate parameters
    if method == "near_miss":
        under_sampler = NearMiss(
            sampling_strategy=sampling_strategy,
            version=near_miss_version,
        )
    elif method == "enn":
        under_sampler = AllKNN(sampling_strategy=sampling_strategy,
                                n_neighbors=100,
                                n_jobs= -1)
    else:
        under_sampler = ALLOWED_METHODS[method](sampling_strategy=sampling_strategy)
    
    # Apply under-sampling
    print(f"Applying {method} with sampling_strategy='{sampling_strategy}'")
    X_resampled, y_resampled = under_sampler.fit_resample(X, y)
    
    # Create balanced dataframe with only the resampled rows
    df_balanced = pd.DataFrame(X_resampled, columns=X_cols)
    
    # Get the row_ids from the resampled data
    row_ids = df_balanced['row_id'].astype(int).values
    
    # Create a new dataframe by selecting rows from the original dataframe using row_ids
    # Use .loc with a list of row_ids to select rows by their row_id value
    df_balanced_full = df_copy.loc[df_copy['row_id'].isin(row_ids)].copy()
    
    # Add the event column back (in case it was modified for binary classification)
    if model_type.lower() == "deepsurv" and target_endpoint is not None:
        # For rows where binary_event_col is 1, set event_col to target_endpoint
        # For rows where binary_event_col is 0, set event_col to 0 (censored)
        # Create a mapping from row_id to y_resampled
        row_id_to_y = dict(zip(df_balanced['row_id'], y_resampled))
        
        # Update the event column based on the mapping
        df_balanced_full[event_col] = df_balanced_full['row_id'].map(
            lambda x: target_endpoint if row_id_to_y.get(x, 0) == 1 else 0
        )
    
    # Drop the temporary row_id column
    if 'row_id' in df_balanced_full.columns:
        df_balanced_full = df_balanced_full.drop('row_id', axis=1)
    
    # Get new event distribution
    new_event_counts = df_balanced_full[event_col].value_counts()
    print(f"Balanced event distribution: {new_event_counts.to_dict()}")
    
    # Verify no new rows were created
    if len(df_balanced_full) > orig_rows:
        raise ValueError(
            f"Under-sampling created new rows, which should not happen. "
            f"Original rows: {orig_rows}, Balanced rows: {len(df_balanced_full)}"
        )
    
    # Calculate reduction percentage
    reduction_pct = ((orig_rows - len(df_balanced_full)) / orig_rows) * 100
    print(f"Reduced dataset by {reduction_pct:.2f}% (from {orig_rows} to {len(df_balanced_full)} rows)")
    
    # Reset index to ensure consistency
    df_balanced_full = df_balanced_full.reset_index(drop=True)
    
    return df_balanced_full