"""
Save test labels for temporal and spatial datasets.
"""

import os
import pickle
import pandas as pd
import numpy as np
from steps.generate_all_predictions_with_baseline import load_preprocessed_data

def main():
    """Save test labels from preprocessed data."""
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    temporal_data, spatial_data = load_preprocessed_data()
    
    # Extract test labels
    y_temporal_test = temporal_data['y_test']
    y_spatial_test = spatial_data['y_test']
    
    print(f"Temporal test samples: {len(y_temporal_test)}")
    print(f"Spatial test samples: {len(y_spatial_test)}")
    
    # Save labels
    output_dir = "results/final_deploy"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save temporal test labels
    temporal_path = os.path.join(output_dir, "temporal_test_labels.pkl")
    with open(temporal_path, 'wb') as f:
        pickle.dump(y_temporal_test, f)
    print(f"Saved temporal test labels to: {temporal_path}")
    
    # Save spatial test labels
    spatial_path = os.path.join(output_dir, "spatial_test_labels.pkl")
    with open(spatial_path, 'wb') as f:
        pickle.dump(y_spatial_test, f)
    print(f"Saved spatial test labels to: {spatial_path}")
    
    # Print sample of labels
    print("\nSample of temporal test labels:")
    print(y_temporal_test.head())
    print(f"\nEvent distribution (temporal):")
    print(y_temporal_test['event'].value_counts())
    
    print("\nSample of spatial test labels:")
    print(y_spatial_test.head())
    print(f"\nEvent distribution (spatial):")
    print(y_spatial_test['event'].value_counts())


if __name__ == "__main__":
    main()