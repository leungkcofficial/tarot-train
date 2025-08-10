"""
Survival Dataset Builder Step for CKD Risk Prediction

This module contains the ZenML step for converting pandas DataFrames to PyCox SurvDataset objects
for deep learning survival analysis.
"""

import pandas as pd
import numpy as np
from zenml.steps import step
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import utility functions
from src.survival_utils import prepare_survival_data, create_pycox_dataset


@step
def survival_dataset_builder(
    train_df: pd.DataFrame,
    features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    duration_col: str = "duration",
    event_col: str = "endpoint",
) -> Tuple[Any, Dict[str, Any], Any, int]:
    """
    Convert pandas DataFrame to PyCox SurvDataset object for deep learning survival analysis.
    
    Args:
        train_df: Training DataFrame
        features: List of feature columns to use (default: None, uses all columns except special ones)
        categorical_features: List of categorical feature columns (default: None)
        duration_col: Name of the duration column (default: "duration")
        event_col: Name of the event column (default: "endpoint")
        
    Returns:
        Tuple containing:
        - train_ds: Training SurvDataset
        - feature_metadata: Dictionary with feature metadata
        - labeller: Fitted label encoder
        - num_features: Number of features
    """
    try:
        print("\n=== Building Survival Datasets ===\n")
        
        # Ensure train_df is a pandas DataFrame (handle ZenML StepArtifact)
        print(f"train_df type: {type(train_df)}")
        if hasattr(train_df, 'dir'):
            print(f"train_df attributes: {dir(train_df)[:10]}...")
        
        from src.util import extract_from_step_artifact, access_zenml_artifact_data
        
        # First try direct access to the raw data
        raw_data = access_zenml_artifact_data(train_df)
        if raw_data is not None and isinstance(raw_data, pd.DataFrame):
            print("Successfully accessed raw DataFrame data from ZenML artifact")
            train_df = raw_data
        else:
            print("Direct access didn't return a DataFrame, trying standard extraction")
            # Try to extract the data using standard methods
            train_df = extract_from_step_artifact(
                train_df,
                expected_type=pd.DataFrame,
                artifact_name="train_df"
            )
        
        # Print information about extracted data
        print(f"Extracted train_df type: {type(train_df)}")
        if hasattr(train_df, 'shape'):
            print(f"train_df shape: {train_df.shape}")
        if hasattr(train_df, 'columns'):
            print(f"train_df columns: {train_df.columns.tolist()[:5]}...")
        
        if train_df is None:
            raise ValueError("Failed to extract DataFrame from train_df")
            
        if not isinstance(train_df, pd.DataFrame):
            try:
                # Try to convert to DataFrame if it's not already
                train_df = pd.DataFrame(train_df)
            except Exception as e:
                print(f"Error converting train_df to DataFrame: {e}")
                print(f"Type of train_df: {type(train_df)}")
                raise ValueError("train_df must be a pandas DataFrame or convertible to one")
        
        # If features not provided, use all columns except special ones
        if features is None:
            exclude_cols = [duration_col, event_col, 'patient_id', 'key', 'date',
                           'endpoint_date', 'first_sub_60_date']
            features = [col for col in train_df.columns if col not in exclude_cols]
            print(f"Using {len(features)} features from DataFrame")
        else:
            print(f"Using {len(features)} user-provided features")
        
        # Check if all features exist in the dataframe
        missing_features = [f for f in features if f not in train_df.columns]
        if missing_features:
            print(f"Warning: The following features are not in the dataframe: {missing_features}")
            # Remove missing features from the list
            features = [f for f in features if f in train_df.columns]
            print(f"Proceeding with {len(features)} available features")
        
        # Print feature list
        print(f"Features: {features}")
        
        # Check if categorical features are provided
        if categorical_features is None:
            print("No categorical features specified")
            categorical_features = []
        else:
            # Filter to only include categorical features that exist in features
            categorical_features = [f for f in categorical_features if f in features]
            print(f"Using {len(categorical_features)} categorical features: {categorical_features}")
        
        # Check if duration and event columns exist
        if duration_col not in train_df.columns:
            print(f"Warning: Duration column '{duration_col}' not found in DataFrame")
            print("Calculating duration from endpoint_date and date columns")
            
            # Calculate duration if endpoint_date and date columns exist
            if 'endpoint_date' in train_df.columns and 'date' in train_df.columns:
                train_df[duration_col] = (pd.to_datetime(train_df['endpoint_date']) - 
                                         pd.to_datetime(train_df['date'])).dt.days
                temporal_test_df[duration_col] = (pd.to_datetime(temporal_test_df['endpoint_date']) - 
                                                pd.to_datetime(temporal_test_df['date'])).dt.days
                spatial_test_df[duration_col] = (pd.to_datetime(spatial_test_df['endpoint_date']) - 
                                               pd.to_datetime(spatial_test_df['date'])).dt.days
                
                # Ensure duration is positive
                train_df[duration_col] = train_df[duration_col].clip(lower=1)
                temporal_test_df[duration_col] = temporal_test_df[duration_col].clip(lower=1)
                spatial_test_df[duration_col] = spatial_test_df[duration_col].clip(lower=1)
                
                print(f"Duration statistics: min={train_df[duration_col].min()}, "
                      f"max={train_df[duration_col].max()}, "
                      f"mean={train_df[duration_col].mean():.2f}")
            else:
                raise ValueError("Cannot calculate duration: endpoint_date or date columns missing")
        
        # Check if event column exists
        if event_col not in train_df.columns:
            raise ValueError(f"Event column '{event_col}' not found in DataFrame")
        
        # Print event distribution
        event_counts = train_df[event_col].value_counts()
        print(f"Event distribution in training set:")
        for event_value, count in event_counts.items():
            print(f"  {event_value}: {count} ({count/len(train_df)*100:.2f}%)")
        
        # Create PyCox dataset
        print("\nCreating PyCox dataset...")
        
        # Create PyCox dataset without additional scaling since data is already preprocessed
        # in the preprocess_data step of the pipeline
        train_ds, _ = create_pycox_dataset(
            df=train_df,
            duration_col=duration_col,
            event_col=event_col,
            feature_cols=features,
            categorical_cols=categorical_features,
            fit_scaler=False  # No need to fit scaler as data is already preprocessed
        )
        
        # Create label encoder for event values
        labeller = LabelEncoder().fit(train_df[event_col])
        
        # Print dataset information
        print(f"Dataset: {len(train_ds[0])} samples")
        
        # Create feature metadata
        feature_metadata = {
            'feature_names': features,
            'categorical_features': categorical_features,
            'scaler': None,  # No scaler used as data is already preprocessed
            'num_features': len(features),
            'duration_col': duration_col,
            'event_col': event_col
        }
        
        # Return the number of features as a separate output
        num_features = len(features)
        return train_ds, feature_metadata, labeller, num_features
        
    except Exception as e:
        print(f"Error building survival datasets: {e}")
        import traceback
        traceback.print_exc()
        raise