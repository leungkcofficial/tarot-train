"""
Fixed sequence utilities for LSTM models.

This version ensures predictions are made for ALL patients in the test set,
padding with zeros for patients with insufficient sequential data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any


def create_sequences_from_dataframe_fixed(
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
    include_all_patients: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences from a dataframe for LSTM training/prediction.
    
    This fixed version includes ALL patients, even those with insufficient data,
    by padding their sequences appropriately.
    
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
        include_all_patients: If True, include all patients regardless of sequence length
        
    Returns:
        Tuple of (X_sequences, durations, events, patient_ids)
        - X_sequences: (n_samples, sequence_length, n_features)
        - durations: (n_samples,)
        - events: (n_samples,)
        - patient_ids: (n_samples,) - patient IDs for each sequence
    """
    print(f"\n=== Creating sequences from dataframe (Fixed Version) ===")
    print(f"Input dataframe shape: {df.shape}")
    print(f"Sequence length: {sequence_length}")
    print(f"Include all patients: {include_all_patients}")
    
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
    
    print(f"Using {len(feature_cols)} feature columns")
    
    # Convert categorical columns to numeric
    df_processed = df.copy()
    for col in feature_cols:
        if col in df_processed.columns and df_processed[col].dtype == 'object':
            if df_processed[col].nunique() <= 2:
                # Binary categorical
                most_common = df_processed[col].mode()[0] if not df_processed[col].mode().empty else df_processed[col].iloc[0]
                df_processed[col] = (df_processed[col] != most_common).astype(float)
            else:
                # Multi-category - use label encoding
                df_processed[col] = pd.Categorical(df_processed[col]).codes.astype(float)
    
    # Sort by patient and date
    df_sorted = df_processed.sort_values([cluster_col, date_col]).reset_index(drop=True)
    
    # Group by patient
    patient_groups = df_sorted.groupby(cluster_col)
    print(f"Found {len(patient_groups)} unique patients")
    
    sequences = []
    durations = []
    events = []
    patient_ids = []
    
    patients_with_sufficient_data = 0
    patients_padded = 0
    patients_with_single_observation = 0
    
    for patient_id, group in patient_groups:
        group_sorted = group.sort_values(date_col).reset_index(drop=True)
        n_observations = len(group_sorted)
        
        # For test set predictions, we MUST include all patients
        if not include_all_patients and n_observations < min_sequence_length:
            continue
        
        # Extract duration and event from the last observation
        last_obs = group_sorted.iloc[-1]
        duration = last_obs[duration_col]
        event = last_obs[event_col]
        
        # Apply target endpoint filtering if specified
        if target_endpoint is not None:
            event = 1.0 if event == target_endpoint else 0.0
        
        # Handle different cases for sequence creation
        if n_observations >= sequence_length:
            # Use the last sequence_length observations
            feature_matrix = group_sorted[feature_cols].values.astype(float)
            sequence = feature_matrix[-sequence_length:]
            patients_with_sufficient_data += 1
            
        elif n_observations > 1:
            # Pad with zeros at the beginning
            feature_matrix = group_sorted[feature_cols].values.astype(float)
            padding_needed = sequence_length - n_observations
            padding = np.full((padding_needed, len(feature_cols)), pad_value, dtype=float)
            sequence = np.vstack([padding, feature_matrix])
            patients_padded += 1
            
        else:  # n_observations == 1
            # Special case: only one observation
            # Create a sequence by repeating the single observation
            single_obs_features = group_sorted[feature_cols].values.astype(float)
            
            if include_all_patients:
                # Option 1: Pad entire sequence with zeros except last position
                sequence = np.full((sequence_length, len(feature_cols)), pad_value, dtype=float)
                sequence[-1] = single_obs_features[0]  # Put the single observation at the end
                
                # Option 2 (alternative): Repeat the single observation
                # sequence = np.repeat(single_obs_features, sequence_length, axis=0)
                
                patients_with_single_observation += 1
            else:
                # Skip if not including all patients
                continue
        
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
    print(f"Total patients in input: {len(patient_groups)}")
    print(f"Total patients processed: {len(sequences)}")
    print(f"Patients with sufficient data (>= {sequence_length} obs): {patients_with_sufficient_data}")
    print(f"Patients requiring padding (2 to {sequence_length-1} obs): {patients_padded}")
    print(f"Patients with single observation: {patients_with_single_observation}")
    print(f"Final sequences shape: {X_sequences.shape}")
    print(f"Event rate: {events_array.mean():.2%}")
    
    # Verify we included all patients if requested
    if include_all_patients and len(sequences) != len(patient_groups):
        print(f"WARNING: Not all patients were included! Expected {len(patient_groups)}, got {len(sequences)}")
    
    return X_sequences, durations_array, events_array, patient_ids_array


def prepare_lstm_survival_dataset_fixed(
    df: pd.DataFrame,
    sequence_length: int,
    feature_cols: List[str] = None,
    cluster_col: str = 'key',
    date_col: str = 'date',
    duration_col: str = 'duration',
    event_col: str = 'endpoint',
    target_endpoint: Optional[int] = None,
    include_all_patients: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare a dataset for LSTM-based survival analysis.
    
    This fixed version ensures all patients are included for test set predictions.
    
    Args:
        df: Input dataframe
        sequence_length: Desired sequence length
        feature_cols: List of feature columns (optional)
        cluster_col: Patient ID column (default: 'key')
        date_col: Date column (default: 'date')
        duration_col: Duration column (default: 'duration')
        event_col: Event column (default: 'endpoint')
        target_endpoint: Specific event type to focus on (optional)
        include_all_patients: If True, include all patients (for test sets)
        
    Returns:
        Tuple of (X_sequences, durations, events)
    """
    print(f"\n=== Preparing LSTM survival dataset (Fixed) ===")
    print(f"Include all patients: {include_all_patients}")
    
    # Create sequences with the fixed function
    X_sequences, durations, events, patient_ids = create_sequences_from_dataframe_fixed(
        df=df,
        sequence_length=sequence_length,
        cluster_col=cluster_col,
        date_col=date_col,
        feature_cols=feature_cols,
        duration_col=duration_col,
        event_col=event_col,
        target_endpoint=target_endpoint,
        include_all_patients=include_all_patients
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


def validate_sequences(
    X_sequences: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    patient_ids: np.ndarray,
    sequence_length: int,
    n_features: int
) -> None:
    """
    Validate the generated sequences.
    
    Args:
        X_sequences: Sequence data
        durations: Duration values
        events: Event indicators
        patient_ids: Patient IDs
        sequence_length: Expected sequence length
        n_features: Expected number of features
    """
    assert X_sequences.shape[0] == len(durations) == len(events) == len(patient_ids), \
        "Mismatch in number of samples"
    
    assert X_sequences.shape[1] == sequence_length, \
        f"Expected sequence length {sequence_length}, got {X_sequences.shape[1]}"
    
    assert X_sequences.shape[2] == n_features, \
        f"Expected {n_features} features, got {X_sequences.shape[2]}"
    
    # Check for NaN values
    if np.any(np.isnan(X_sequences)):
        print("WARNING: NaN values found in sequences")
    
    if np.any(np.isnan(durations)):
        print("WARNING: NaN values found in durations")
    
    if np.any(np.isnan(events)):
        print("WARNING: NaN values found in events")
    
    print("Sequence validation passed!")