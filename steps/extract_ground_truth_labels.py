"""
Step to extract ground truth labels from test datasets.
"""

import pandas as pd
import yaml
from typing import Tuple, Annotated
from zenml import step


@step
def extract_ground_truth_labels(
    temporal_test_df: pd.DataFrame,
    spatial_test_df: pd.DataFrame
) -> Tuple[
    Annotated[pd.DataFrame, "y_temporal_test"],
    Annotated[pd.DataFrame, "y_spatial_test"]
]:
    """
    Extract ground truth duration and event labels from test datasets.
    
    Args:
        temporal_test_df: Temporal test dataframe
        spatial_test_df: Spatial test dataframe
        
    Returns:
        Tuple of (y_temporal_test, y_spatial_test) with 'time' and 'event' columns
    """
    # Load master dataframe mapping
    with open('src/default_master_df_mapping.yml', 'r') as f:
        mapping = yaml.safe_load(f)
    
    # Extract column names
    duration_col = mapping['duration']  # 'duration'
    event_col = mapping['event']  # 'endpoint'
    
    # Extract ground truth from temporal test set
    y_temporal_test = pd.DataFrame({
        'time': temporal_test_df[duration_col],
        'event': temporal_test_df[event_col]
    })
    
    # Extract ground truth from spatial test set
    y_spatial_test = pd.DataFrame({
        'time': spatial_test_df[duration_col],
        'event': spatial_test_df[event_col]
    })
    
    print(f"Extracted ground truth labels:")
    print(f"  Temporal test samples: {len(y_temporal_test)}")
    print(f"  Temporal event distribution:\n{y_temporal_test['event'].value_counts().sort_index()}")
    print(f"  Spatial test samples: {len(y_spatial_test)}")
    print(f"  Spatial event distribution:\n{y_spatial_test['event'].value_counts().sort_index()}")
    
    # Save to CSV for reference
    y_temporal_test.to_csv('results/final_deploy/temporal_test_labels.csv', index=False)
    y_spatial_test.to_csv('results/final_deploy/spatial_test_labels.csv', index=False)
    
    return y_temporal_test, y_spatial_test