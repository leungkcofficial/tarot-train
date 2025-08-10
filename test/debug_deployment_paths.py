"""
Debug script to extract prediction paths from deployment details
"""

from zenml.client import Client

print("Checking prediction paths in deployment details...")

# Load deployment details
client = Client()
artifact1 = client.get_artifact_version("ea12d9e2-1225-4a5e-8c49-e8443047fe21")
deployment_details = artifact1.load()

print(f"\nTotal deployment details: {len(deployment_details)}")

# Check the structure of deployment details
if deployment_details:
    print("\n=== First Deployment Detail Structure ===")
    first = deployment_details[0]
    for key, value in first.items():
        if 'path' in key.lower():
            print(f"{key}: {value}")
    
    print("\n=== Extracting Prediction Paths ===")
    temporal_paths = []
    spatial_paths = []
    
    for i, details in enumerate(deployment_details):
        # Check different possible keys for prediction paths
        temporal_path = (
            details.get('temporal_test_predictions_path') or
            details.get('temporal_predictions_path') or
            details.get('prediction_paths', {}).get('temporal', '')
        )
        spatial_path = (
            details.get('spatial_test_predictions_path') or
            details.get('spatial_predictions_path') or
            details.get('prediction_paths', {}).get('spatial', '')
        )
        
        temporal_paths.append(temporal_path)
        spatial_paths.append(spatial_path)
        
        if i < 5:  # Show first 5
            print(f"\nModel {i+1} ({details.get('model_config', {}).get('model_no', 'N/A')}):")
            print(f"  Temporal: {temporal_path}")
            print(f"  Spatial: {spatial_path}")
    
    # Count non-empty paths
    non_empty_temporal = sum(1 for path in temporal_paths if path and path != '')
    non_empty_spatial = sum(1 for path in spatial_paths if path and path != '')
    
    print(f"\n=== Path Summary ===")
    print(f"Non-empty temporal paths: {non_empty_temporal}/{len(temporal_paths)}")
    print(f"Non-empty spatial paths: {non_empty_spatial}/{len(spatial_paths)}")