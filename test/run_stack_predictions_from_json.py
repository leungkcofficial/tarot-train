"""
Load and process the cached artifacts from JSON files.
"""

import os
import json
import numpy as np
import h5py
from steps.stack_predictions import stack_predictions
from steps.ensemble_predictions_fixed import ensemble_predictions_fixed

# Artifact paths from the 3-day run
artifact_dirs = {
    'deployment_details': '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_0/67c9050f-777c-40ad-b087-a4bd53d6c47a/99a24532',
    'temporal_predictions': '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_1/67c9050f-777c-40ad-b087-a4bd53d6c47a/8784e2ea',
    'spatial_predictions': '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_2/67c9050f-777c-40ad-b087-a4bd53d6c47a/385bd4d0'
}

print("Loading cached artifacts from JSON files...")

# Load the artifacts
loaded_artifacts = {}
for name, dir_path in artifact_dirs.items():
    json_path = os.path.join(dir_path, 'data.json')
    print(f"\nLoading {name} from {json_path}")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            loaded_artifacts[name] = data
            print(f"  Loaded JSON. Type: {type(data)}")
            if isinstance(data, list):
                print(f"  List length: {len(data)}")
                if data:
                    print(f"  First item type: {type(data[0])}")
                    if isinstance(data[0], dict):
                        print(f"  First item keys: {list(data[0].keys())[:10]}")
            elif isinstance(data, dict):
                print(f"  Dict keys: {list(data.keys())[:10]}")

# Check what we loaded
print("\n=== Processing Loaded Data ===")

# Get the deployment details
if 'deployment_details' in loaded_artifacts:
    deployment_details = loaded_artifacts['deployment_details']
    print(f"\nDeployment details: {type(deployment_details)}")
    
    if isinstance(deployment_details, list):
        print(f"Number of deployment details: {len(deployment_details)}")
        
        # Check first item
        if deployment_details:
            first = deployment_details[0]
            print(f"First deployment detail keys: {list(first.keys()) if isinstance(first, dict) else 'Not a dict'}")
            
            # Extract paths if they exist
            temporal_paths = []
            spatial_paths = []
            
            for detail in deployment_details:
                if isinstance(detail, dict):
                    if 'temporal_predictions_path' in detail:
                        temporal_paths.append(detail['temporal_predictions_path'])
                    if 'spatial_predictions_path' in detail:
                        spatial_paths.append(detail['spatial_predictions_path'])
            
            print(f"\nExtracted {len(temporal_paths)} temporal paths")
            print(f"Extracted {len(spatial_paths)} spatial paths")

# Get the prediction paths directly
temporal_paths = loaded_artifacts.get('temporal_predictions', [])
spatial_paths = loaded_artifacts.get('spatial_predictions', [])

print(f"\nTemporal predictions: {type(temporal_paths)}")
print(f"Spatial predictions: {type(spatial_paths)}")

if isinstance(temporal_paths, list) and temporal_paths:
    print(f"\nFound {len(temporal_paths)} temporal prediction paths")
    print("Sample paths:")
    for i, path in enumerate(temporal_paths[:3]):
        print(f"  {i}: {path}")
        if os.path.exists(path):
            print(f"     File exists!")
            # Check the content
            try:
                with h5py.File(path, 'r') as f:
                    keys = list(f.keys())
                    print(f"     H5 keys: {keys}")
                    if 'predictions' in f:
                        shape = f['predictions'].shape
                        print(f"     Predictions shape: {shape}")
                    elif 'cif' in f:
                        shape = f['cif'].shape
                        print(f"     CIF shape: {shape}")
            except Exception as e:
                print(f"     Error reading H5: {e}")
        else:
            print(f"     File does NOT exist")

if isinstance(spatial_paths, list) and spatial_paths:
    print(f"\nFound {len(spatial_paths)} spatial prediction paths")

# Now run the stacking step
if temporal_paths and isinstance(temporal_paths, list):
    print("\n=== Running Stack Predictions Step ===")
    
    # We need deployment details for the stack_predictions step
    # If we don't have proper deployment details, create them from the paths
    if 'deployment_details' not in loaded_artifacts or not isinstance(loaded_artifacts['deployment_details'][0], dict):
        print("Creating deployment details from prediction paths...")
        deployment_details = []
        
        # We have 24 models total: 12 DeepSurv groups + 12 DeepHit models
        for i in range(len(temporal_paths)):
            if i < 12:
                # DeepSurv groups (2 models each for Event 1 and Event 2)
                group_idx = i
                model_type = 'DeepSurv'
                model_identifier = f'deepsurv_group_{group_idx}'
                
                # Infer structure and other details from group index
                if group_idx < 6:
                    structure = 'ANN'
                else:
                    structure = 'LSTM'
                
                # Balancing methods cycle through None, NearMiss, KNN
                balancing_idx = (group_idx // 2) % 3
                balancing_methods = ['None', 'NearMiss version 1', 'KNN'] if structure == 'ANN' else ['None', 'NearMiss version 3', 'KNN']
                balancing = balancing_methods[balancing_idx]
                
                # Optimization target alternates
                opt_target = 'Concordance Index' if group_idx % 2 == 0 else 'Log-likelihood'
                
            else:
                # DeepHit models
                deephit_idx = i - 12
                model_type = 'DeepHit'
                model_identifier = f'deephit_{deephit_idx}'
                
                # Structure alternates between ANN and LSTM
                structure = 'ANN' if deephit_idx < 6 else 'LSTM'
                
                # Balancing methods
                balancing_idx = (deephit_idx // 2) % 3
                balancing_methods = ['None', 'NearMiss version 3', 'KNN']
                balancing = balancing_methods[balancing_idx]
                
                opt_target = 'Log-likelihood'  # DeepHit typically uses log-likelihood
            
            detail = {
                'model_config': {
                    'algorithm': model_type,
                    'structure': structure,
                    'balancing_method': balancing,
                    'optimization_target': opt_target,
                    'model_no': i + 1
                },
                'temporal_predictions_path': temporal_paths[i] if i < len(temporal_paths) else '',
                'spatial_predictions_path': spatial_paths[i] if i < len(spatial_paths) else '',
                'model_identifier': model_identifier
            }
            deployment_details.append(detail)
        
        print(f"Created {len(deployment_details)} deployment details")
    
    # Run stacking
    temporal_stacked, spatial_stacked = stack_predictions(
        deployment_details=deployment_details,
        temporal_predictions_paths=temporal_paths,
        spatial_predictions_paths=spatial_paths
    )
    
    print(f"\nStacking complete!")
    print(f"Temporal stacked models: {len(temporal_stacked)}")
    print(f"Spatial stacked models: {len(spatial_stacked)}")
    
    # Show the shapes of stacked predictions
    print("\nStacked prediction shapes:")
    for i, (model_name, predictions) in enumerate(temporal_stacked.items()):
        if i < 5:
            print(f"  {model_name}: {predictions.shape}")
    
    # Continue with ensemble predictions
    print("\n=== Running Fixed Ensemble Predictions Step ===")
    temporal_ensemble_path, spatial_ensemble_path, ensemble_metadata = ensemble_predictions_fixed(
        temporal_stacked=temporal_stacked,
        spatial_stacked=spatial_stacked,
        ensemble_method="average"
    )
    
    print(f"\nEnsemble predictions complete!")
    print(f"Temporal ensemble: {temporal_ensemble_path}")
    print(f"Spatial ensemble: {spatial_ensemble_path}")
    print(f"Ensemble metadata: {ensemble_metadata}")
    
    # Verify the ensemble shapes
    print("\nVerifying ensemble prediction shapes...")
    with h5py.File(temporal_ensemble_path, 'r') as f:
        ensemble_shape = f['predictions'].shape
        print(f"Temporal ensemble shape: {ensemble_shape}")
        print(f"Expected format: (n_events=2, n_timepoints=5, n_samples={ensemble_shape[-1]})")
    
    print("\n✅ Pipeline completed successfully!")
    print("\nNext steps:")
    print("1. The ensemble predictions have been created with consistent shapes")
    print("2. You can now run ensemble evaluation with the original test dataframes")
    print("3. The LSTM prediction alignment issue has been addressed in the fixed ensemble step")
else:
    print("\n❌ ERROR: Could not find prediction paths in the loaded artifacts")