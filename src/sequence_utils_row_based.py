"""
Row-based Sequence Utilities for LSTM-based Survival Analysis

This module provides functions for converting tabular survival data into
sequences suitable for LSTM training, where each row generates a sequence
by looking back at previous observations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
import warnings


def create_sequences_for_all_rows(
    df: pd.DataFrame,
    sequence_length: int,
    cluster_col: str = 'key',
    date_col: str = 'date',
    feature_cols: List[str] = None,
    duration_col: str = 'duration',
    event_col: str = 'endpoint',
    target_endpoint: Optional[int] = None,
    pad_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences from a dataframe where each row generates a sequence.
    
    This function creates a sequence for EACH row in the dataset by looking
    back at previous observations for that patient. If a patient doesn't have
    enough previous observations, the sequence is padded with zeros.
    
    Args:
        df: Input dataframe with patient data
        sequence_length: Desired sequence length
        cluster_col: Column name for patient/cluster ID (default: 'key')
        date_col: Column name for date/time (default: 'date')
        feature_cols: List of feature columns to include
        duration_col: Column name for survival duration (default: 'duration')
        event_col: Column name for event indicator (default: 'endpoint')
        target_endpoint: Specific event type to focus on (optional)
        pad_value: Value to use for padding short sequences (default: 0.0)
        
    Returns:
        Tuple of (X_sequences, durations, events, row_indices)
        - X_sequences: (n_rows, sequence_length, n_features)
        - durations: (n_rows,)
        - events: (n_rows,)
        - row_indices: (n_rows,) - original row indices for tracking
    """
    print(f"\n=== Creating sequences for all rows ===")
    print(f"Input dataframe shape: {df.shape}")
    print(f"Sequence length: {sequence_length}")
    print(f"This will create {df.shape[0]} sequences (one per row)")
    
    # Validate required columns
    required_cols = [cluster_col, date_col, duration_col, event_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Prepare feature columns
    if feature_cols is None:
        # Exclude non-feature columns
        exclude_cols = [cluster_col, date_col, duration_col, event_col, 
                       'patient_id', 'endpoint_date', 'first_sub_60_date', 'dob', 'icd10']
        exclude_cols = [col for col in exclude_cols if col in df.columns]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} feature columns: {feature_cols[:5]}...")
    
    # Convert categorical columns to numeric
    df_processed = df.copy()
    for col in feature_cols:
        if col in df_processed.columns and df_processed[col].dtype == 'object':
            print(f"Converting categorical column to numeric: {col}")
            if df_processed[col].nunique() <= 2:
                # Binary categorical
                most_common = df_processed[col].mode()[0]
                df_processed[col] = (df_processed[col] != most_common).astype(float)
            else:
                # Multi-category - use label encoding for simplicity
                df_processed[col] = pd.Categorical(df_processed[col]).codes.astype(float)
    
    # Sort by patient and date to ensure chronological order
    df_sorted = df_processed.sort_values([cluster_col, date_col]).reset_index(drop=True)
    print(f"Sorted dataframe by {cluster_col} and {date_col}")
    
    # Initialize output arrays
    n_rows = len(df_sorted)
    n_features = len(feature_cols)
    X_sequences = np.zeros((n_rows, sequence_length, n_features), dtype=np.float32)
    durations = np.zeros(n_rows, dtype=np.float32)
    events = np.zeros(n_rows, dtype=np.float32)
    row_indices = np.arange(n_rows)
    
    # Track statistics
    rows_with_full_history = 0
    rows_with_partial_history = 0
    rows_with_no_history = 0
    
    # Process each row
    for idx in range(n_rows):
        if idx % 10000 == 0:
            print(f"Processing row {idx}/{n_rows} ({idx/n_rows*100:.1f}%)")
        
        current_row = df_sorted.iloc[idx]
        patient_id = current_row[cluster_col]
        current_date = current_row[date_col]
        
        # Get duration and event for this row
        duration = current_row[duration_col]
        event = current_row[event_col]
        
        # Apply target endpoint filtering if specified
        if target_endpoint is not None:
            event = 1.0 if event == target_endpoint else 0.0
        
        durations[idx] = duration
        events[idx] = event
        
        # Get all observations for this patient up to and including current date
        patient_mask = (df_sorted[cluster_col] == patient_id) & (df_sorted[date_col] <= current_date)
        patient_history = df_sorted[patient_mask]
        
        # Extract features from patient history
        history_features = patient_history[feature_cols].values.astype(np.float32)
        n_history = len(history_features)
        
        if n_history >= sequence_length:
            # Use the most recent sequence_length observations
            X_sequences[idx] = history_features[-sequence_length:]
            rows_with_full_history += 1
        elif n_history > 0:
            # Pad with zeros at the beginning
            padding_needed = sequence_length - n_history
            X_sequences[idx, :padding_needed] = pad_value
            X_sequences[idx, padding_needed:] = history_features
            rows_with_partial_history += 1
        else:
            # No history - all padding (shouldn't happen if data is sorted correctly)
            X_sequences[idx] = pad_value
            rows_with_no_history += 1
            warnings.warn(f"Row {idx} has no history for patient {patient_id}")
    
    print(f"\n=== Sequence generation summary ===")
    print(f"Total rows processed: {n_rows}")
    print(f"Rows with full history (>= {sequence_length} obs): {rows_with_full_history}")
    print(f"Rows with partial history (padded): {rows_with_partial_history}")
    print(f"Rows with no history: {rows_with_no_history}")
    print(f"Final sequences shape: {X_sequences.shape}")
    print(f"Durations shape: {durations.shape}")
    print(f"Events shape: {events.shape}")
    print(f"Event rate: {events.mean():.2%}")
    print(f"Duration range: {durations.min():.0f} - {durations.max():.0f} days")
    
    return X_sequences, durations, events, row_indices


def create_sequences_from_dataframe(
    df: pd.DataFrame,
    sequence_length: int,
    cluster_col: str = 'key',
    date_col: str = 'date',
    feature_cols: List[str] = None,
    duration_col: str = 'duration',
    event_col: str = 'endpoint',
    target_endpoint: Optional[int] = None,
    pad_value: float = 0.0,
    min_sequence_length: int = 1,
    use_all_rows: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences from a dataframe for LSTM training.
    
    This is a wrapper function that can either create sequences for all rows
    (new behavior) or one sequence per patient (old behavior).
    
    Args:
        df: Input dataframe with patient data
        sequence_length: Desired sequence length
        cluster_col: Column name for patient/cluster ID (default: 'key')
        date_col: Column name for date/time (default: 'date')
        feature_cols: List of feature columns to include
        duration_col: Column name for survival duration (default: 'duration')
        event_col: Column name for event indicator (default: 'endpoint')
        target_endpoint: Specific event type to focus on (optional)
        pad_value: Value to use for padding short sequences (default: 0.0)
        min_sequence_length: Minimum sequence length to include (default: 1)
        use_all_rows: If True, create sequence for each row; if False, one per patient
        
    Returns:
        Tuple of (X_sequences, durations, events, identifiers)
        - X_sequences: (n_samples, sequence_length, n_features)
        - durations: (n_samples,)
        - events: (n_samples,)
        - identifiers: (n_samples,) - row indices or patient IDs
    """
    if use_all_rows:
        # New behavior: create sequence for each row
        return create_sequences_for_all_rows(
            df=df,
            sequence_length=sequence_length,
            cluster_col=cluster_col,
            date_col=date_col,
            feature_cols=feature_cols,
            duration_col=duration_col,
            event_col=event_col,
            target_endpoint=target_endpoint,
            pad_value=pad_value
        )
    else:
        # Old behavior: one sequence per patient (import from original module)
        from sequence_utils import create_sequences_from_dataframe as create_sequences_original
        return create_sequences_original(
            df=df,
            sequence_length=sequence_length,
            cluster_col=cluster_col,
            date_col=date_col,
            feature_cols=feature_cols,
            duration_col=duration_col,
            event_col=event_col,
            target_endpoint=target_endpoint,
            pad_value=pad_value,
            min_sequence_length=min_sequence_length
        )