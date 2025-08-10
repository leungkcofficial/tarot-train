import pandas as pd
import json
import h5py
import numpy as np
from collections import defaultdict

def check_group_prediction_shapes():
    """Check the CIF shapes for each DeepSurv group's predictions"""
    
    # Read the model configuration
    config_df = pd.read_csv('results/final_deploy/model_config/model_config.csv')
    
    # Filter only DeepSurv models
    deepsurv_df = config_df[config_df['Algorithm'] == 'DeepSurv'].copy()
    
    # Handle NaN values in Balancing Method
    deepsurv_df['Balancing Method'] = deepsurv_df['Balancing Method'].fillna('None')
    
    # Create grouping key (excluding endpoint)
    deepsurv_df['group_key'] = (
        deepsurv_df['Algorithm'] + '_' + 
        deepsurv_df['Structure'] + '_' + 
        deepsurv_df['Balancing Method'] + '_' + 
        deepsurv_df['Optimization target']
    )
    
    # Group models
    groups = defaultdict(list)
    for _, row in deepsurv_df.iterrows():
        groups[row['group_key']].append({
            'model_no': row['Model No.'],
            'endpoint': row['Prediction Endpoint']
        })
    
    # Load the model data from JSON
    json_file = '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_0/67c9050f-777c-40ad-b087-a4bd53d6c47a/99a24532/data.json'
    with open(json_file, 'r') as f:
        model_data = json.load(f)
    
    # Create mapping of model number to prediction paths
    model_paths = {}
    for model in model_data:
        if model['model_type'] == 'deepsurv':
            model_name = model['model_name']
            model_num = int(model_name.split('_')[1].replace('model', ''))
            model_paths[model_num] = {
                'name': model_name,
                'temporal_test': model['temporal_test_predictions_path'],
                'spatial_test': model['spatial_test_predictions_path']
            }
    
    # Check shapes for each group
    print("### DEEPSURV GROUP PREDICTION SHAPES ###")
    print("=" * 80)
    
    group_num = 1
    for group_key, models in sorted(groups.items()):
        parts = group_key.split('_')
        structure = parts[1]
        balancing = '_'.join(parts[2:-2]) if len(parts) > 4 else parts[2]
        optimization = '_'.join(parts[-2:]) if parts[-2] in ['Concordance', 'Log'] else parts[-1]
        
        print(f"\nGroup {group_num}: {structure} + {balancing} + {optimization}")
        print("-" * 60)
        
        # Sort models by endpoint
        event1_models = [m for m in models if m['endpoint'] == 'Event 1']
        event2_models = [m for m in models if m['endpoint'] == 'Event 2']
        
        # Check Event 1 models
        print("\n  Event 1 Models:")
        for m in event1_models:
            model_no = m['model_no']
            if model_no in model_paths:
                model_info = model_paths[model_no]
                print(f"    Model {model_no}: {model_info['name']}")
                
                # Check temporal test predictions
                try:
                    with h5py.File(model_info['temporal_test'], 'r') as f:
                        cif_shape = f['cif'].shape
                        print(f"      Temporal Test CIF shape: {cif_shape}")
                except Exception as e:
                    print(f"      Error reading temporal test: {e}")
                
                # Check spatial test predictions
                try:
                    with h5py.File(model_info['spatial_test'], 'r') as f:
                        cif_shape = f['cif'].shape
                        print(f"      Spatial Test CIF shape: {cif_shape}")
                except Exception as e:
                    print(f"      Error reading spatial test: {e}")
        
        # Check Event 2 models
        print("\n  Event 2 Models:")
        for m in event2_models:
            model_no = m['model_no']
            if model_no in model_paths:
                model_info = model_paths[model_no]
                print(f"    Model {model_no}: {model_info['name']}")
                
                # Check temporal test predictions
                try:
                    with h5py.File(model_info['temporal_test'], 'r') as f:
                        cif_shape = f['cif'].shape
                        print(f"      Temporal Test CIF shape: {cif_shape}")
                except Exception as e:
                    print(f"      Error reading temporal test: {e}")
                
                # Check spatial test predictions
                try:
                    with h5py.File(model_info['spatial_test'], 'r') as f:
                        cif_shape = f['cif'].shape
                        print(f"      Spatial Test CIF shape: {cif_shape}")
                except Exception as e:
                    print(f"      Error reading spatial test: {e}")
        
        group_num += 1
    
    # Also check DeepHit models
    print("\n" + "=" * 80)
    print("### DEEPHIT MODEL PREDICTION SHAPES ###")
    print("=" * 80)
    
    deephit_models = []
    for model in model_data:
        if model['model_type'] == 'deephit':
            deephit_models.append(model)
    
    # Sort by model name
    deephit_models.sort(key=lambda x: x['model_name'])
    
    for model in deephit_models:
        print(f"\n{model['model_name']}:")
        
        # Check temporal test predictions
        try:
            with h5py.File(model['temporal_test_predictions_path'], 'r') as f:
                cif_shape = f['cif'].shape
                print(f"  Temporal Test CIF shape: {cif_shape}")
        except Exception as e:
            print(f"  Error reading temporal test: {e}")
        
        # Check spatial test predictions
        try:
            with h5py.File(model['spatial_test_predictions_path'], 'r') as f:
                cif_shape = f['cif'].shape
                print(f"  Spatial Test CIF shape: {cif_shape}")
        except Exception as e:
            print(f"  Error reading spatial test: {e}")

if __name__ == "__main__":
    check_group_prediction_shapes()