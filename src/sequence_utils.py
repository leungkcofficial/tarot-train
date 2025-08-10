"""
Sequence Utilities for LSTM-based Survival Analysis

This module provides functions for converting tabular survival data into
sequences suitable for LSTM training, including sequence generation,
padding, and validation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
import warnings


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
    min_sequence_length: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences from a dataframe for LSTM training.
    
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
        
    Returns:
        Tuple of (X_sequences, durations, events, patient_ids)
        - X_sequences: (n_samples, sequence_length, n_features)
        - durations: (n_samples,)
        - events: (n_samples,)
        - patient_ids: (n_samples,) - patient IDs for each sequence
    """
    print(f"\n=== Creating sequences from dataframe ===")
    print(f"Input dataframe shape: {df.shape}")
    print(f"Sequence length: {sequence_length}")
    print(f"Cluster column: {cluster_col}")
    print(f"Date column: {date_col}")
    
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
    
    # Sort by patient and date
    df_sorted = df_processed.sort_values([cluster_col, date_col]).reset_index(drop=True)
    print(f"Sorted dataframe by {cluster_col} and {date_col}")
    
    # Group by patient
    patient_groups = df_sorted.groupby(cluster_col)
    print(f"Found {len(patient_groups)} unique patients")
    
    sequences = []
    durations = []
    events = []
    patient_ids = []
    
    patients_with_sufficient_data = 0
    patients_padded = 0
    
    for patient_id, group in patient_groups:
        group_sorted = group.sort_values(date_col).reset_index(drop=True)
        n_observations = len(group_sorted)
        
        if n_observations < min_sequence_length:
            continue
        
        # Extract features, duration, and event from the last observation
        last_obs = group_sorted.iloc[-1]
        duration = last_obs[duration_col]
        event = last_obs[event_col]
        
        # Apply target endpoint filtering if specified
        if target_endpoint is not None:
            event = 1.0 if event == target_endpoint else 0.0
        
        # Extract feature matrix for this patient
        feature_matrix = group_sorted[feature_cols].values.astype(float)
        
        if n_observations >= sequence_length:
            # Use the last sequence_length observations
            sequence = feature_matrix[-sequence_length:]
            patients_with_sufficient_data += 1
        else:
            # Pad with zeros at the beginning (older timestamps)
            padding_needed = sequence_length - n_observations
            padding = np.full((padding_needed, len(feature_cols)), pad_value, dtype=float)
            sequence = np.vstack([padding, feature_matrix])
            patients_padded += 1
        
        sequences.append(sequence)
        durations.append(duration)
        events.append(event)
        patient_ids.append(patient_id)
    
    # Convert to numpy arrays
    X_sequences = np.array(sequences, dtype=float)
    durations_array = np.array(durations, dtype=float)
    events_array = np.array(events, dtype=float)
    patient_ids_array = np.array(patient_ids)
    
    print(f"\n=== Sequence generation summary ===")
    print(f"Total patients processed: {len(patient_groups)}")
    print(f"Patients with sufficient data (>= {sequence_length} obs): {patients_with_sufficient_data}")
    print(f"Patients requiring padding: {patients_padded}")
    print(f"Final sequences shape: {X_sequences.shape}")
    print(f"Durations shape: {durations_array.shape}")
    print(f"Events shape: {events_array.shape}")
    print(f"Event rate: {events_array.mean():.2%}")
    
    return X_sequences, durations_array, events_array, patient_ids_array


def validate_sequences(
    X_sequences: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    patient_ids: np.ndarray,
    sequence_length: int,
    n_features: int
) -> None:
    """
    Validate sequence data for LSTM training.
    
    Args:
        X_sequences: Sequence features array
        durations: Duration array
        events: Event array
        patient_ids: Patient ID array
        sequence_length: Expected sequence length
        n_features: Expected number of features
    """
    print(f"\n=== Validating sequence data ===")
    
    # Check shapes
    expected_shape = (len(durations), sequence_length, n_features)
    if X_sequences.shape != expected_shape:
        raise ValueError(f"X_sequences shape {X_sequences.shape} != expected {expected_shape}")
    
    if len(durations) != len(events) or len(durations) != len(patient_ids):
        raise ValueError("Inconsistent array lengths")
    
    # Check for NaN values
    if np.isnan(X_sequences).any():
        nan_count = np.isnan(X_sequences).sum()
        warnings.warn(f"Found {nan_count} NaN values in X_sequences")
    
    if np.isnan(durations).any():
        raise ValueError("Found NaN values in durations")
    
    if np.isnan(events).any():
        raise ValueError("Found NaN values in events")
    
    # Check value ranges
    if (durations < 0).any():
        raise ValueError("Found negative durations")
    
    if not np.all(np.isin(events, [0, 1])):
        unique_events = np.unique(events)
        warnings.warn(f"Events contain non-binary values: {unique_events}")
    
    # Check for infinite values
    if np.isinf(X_sequences).any():
        inf_count = np.isinf(X_sequences).sum()
        warnings.warn(f"Found {inf_count} infinite values in X_sequences")
    
    print(f"Validation passed for {len(durations)} sequences")
    print(f"Sequence shape: {X_sequences.shape}")
    print(f"Duration range: [{durations.min():.1f}, {durations.max():.1f}]")
    print(f"Event distribution: {np.bincount(events.astype(int))}")


def prepare_lstm_survival_dataset(
    df: pd.DataFrame,
    sequence_length: int,
    feature_cols: List[str] = None,
    cluster_col: str = 'key',
    date_col: str = 'date',
    duration_col: str = 'duration',
    event_col: str = 'endpoint',
    target_endpoint: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare a dataset for LSTM-based survival analysis.
    
    This is a convenience function that combines sequence creation and validation.
    
    Args:
        df: Input dataframe
        sequence_length: Desired sequence length
        feature_cols: List of feature columns (optional)
        cluster_col: Patient ID column (default: 'key')
        date_col: Date column (default: 'date')
        duration_col: Duration column (default: 'duration')
        event_col: Event column (default: 'endpoint')
        target_endpoint: Specific event type to focus on (optional)
        
    Returns:
        Tuple of (X_sequences, durations, events)
    """
    print(f"\n=== Preparing LSTM survival dataset ===")
    
    # Create sequences
    X_sequences, durations, events, patient_ids = create_sequences_from_dataframe(
        df=df,
        sequence_length=sequence_length,
        cluster_col=cluster_col,
        date_col=date_col,
        feature_cols=feature_cols,
        duration_col=duration_col,
        event_col=event_col,
        target_endpoint=target_endpoint
    )
    
    # Validate sequences
    n_features = len(feature_cols) if feature_cols else X_sequences.shape[2]
    validate_sequences(
        X_sequences=X_sequences,
        durations=durations,
        events=events,
        patient_ids=patient_ids,
        sequence_length=sequence_length,
        n_features=n_features
    )
    
    return X_sequences, durations, events


def get_sequence_statistics(
    X_sequences: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray
) -> Dict[str, Any]:
    """
    Get statistics about sequence data.
    
    Args:
        X_sequences: Sequence features array
        durations: Duration array
        events: Event array
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'n_samples': len(durations),
        'sequence_length': X_sequences.shape[1],
        'n_features': X_sequences.shape[2],
        'event_rate': events.mean(),
        'duration_stats': {
            'mean': durations.mean(),
            'std': durations.std(),
            'min': durations.min(),
            'max': durations.max(),
            'median': np.median(durations)
        },
        'feature_stats': {
            'mean': X_sequences.mean(axis=(0, 1)),
            'std': X_sequences.std(axis=(0, 1)),
            'min': X_sequences.min(axis=(0, 1)),
            'max': X_sequences.max(axis=(0, 1))
        }
    }
    
    return stats


def print_sequence_summary(
    X_sequences: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    name: str = "Dataset"
) -> None:
    """
    Print a summary of sequence data.
    
    Args:
        X_sequences: Sequence features array
        durations: Duration array
        events: Event array
        name: Name for the dataset
    """
    stats = get_sequence_statistics(X_sequences, durations, events)
    
    print(f"\n=== {name} Summary ===")
    print(f"Samples: {stats['n_samples']}")
    print(f"Sequence length: {stats['sequence_length']}")
    print(f"Features: {stats['n_features']}")
    print(f"Event rate: {stats['event_rate']:.2%}")
    print(f"Duration - Mean: {stats['duration_stats']['mean']:.1f}, "
          f"Std: {stats['duration_stats']['std']:.1f}, "
          f"Range: [{stats['duration_stats']['min']:.1f}, {stats['duration_stats']['max']:.1f}]")
    print(f"Feature means (first 5): {stats['feature_stats']['mean'][:5]}")