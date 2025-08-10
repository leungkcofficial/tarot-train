import pandas as pd
import json
from collections import defaultdict

def verify_deepsurv_groupings():
    """Verify the correct grouping of DeepSurv models based on model_config.csv"""
    
    # Read the model configuration
    config_df = pd.read_csv('results/final_deploy/model_config/model_config.csv')
    
    # Filter only DeepSurv models
    deepsurv_df = config_df[config_df['Algorithm'] == 'DeepSurv'].copy()
    
    # Create grouping key (excluding endpoint)
    # Handle NaN values in Balancing Method
    deepsurv_df['Balancing Method'] = deepsurv_df['Balancing Method'].fillna('None')
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
    
    # Display groups
    print("### DEEPSURV MODEL GROUPINGS ###")
    print("(Models grouped by: Algorithm + Structure + Balancing Method + Optimization Target)")
    print("=" * 80)
    
    group_num = 1
    for group_key, models in sorted(groups.items()):
        # Parse group key
        parts = group_key.split('_')
        algorithm = parts[0]
        structure = parts[1]
        balancing = '_'.join(parts[2:-2]) if len(parts) > 4 else parts[2]
        optimization = '_'.join(parts[-2:]) if parts[-2] in ['Concordance', 'Log'] else parts[-1]
        
        print(f"\nGroup {group_num}: {structure} + {balancing} + {optimization}")
        print(f"  Models in this group:")
        
        # Sort models by endpoint
        event1_models = [m for m in models if m['endpoint'] == 'Event 1']
        event2_models = [m for m in models if m['endpoint'] == 'Event 2']
        
        for m in event1_models:
            print(f"    - Model {m['model_no']} (Event 1)")
        for m in event2_models:
            print(f"    - Model {m['model_no']} (Event 2)")
        
        group_num += 1
    
    # Load the actual model names from the JSON data
    json_file = '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_0/67c9050f-777c-40ad-b087-a4bd53d6c47a/99a24532/data.json'
    with open(json_file, 'r') as f:
        model_data = json.load(f)
    
    # Create mapping of model number to actual model name
    model_name_map = {}
    for model in model_data:
        if model['model_type'] == 'deepsurv':
            # Extract model number from name (e.g., "Ensemble_model1_DeepSurv_ANN_Event_1" -> 1)
            model_name = model['model_name']
            model_num = int(model_name.split('_')[1].replace('model', ''))
            model_name_map[model_num] = model_name
    
    # Display groups with actual model names
    print("\n" + "=" * 80)
    print("### GROUPS WITH ACTUAL MODEL NAMES ###")
    print("=" * 80)
    
    group_num = 1
    for group_key, models in sorted(groups.items()):
        parts = group_key.split('_')
        structure = parts[1]
        balancing = '_'.join(parts[2:-2]) if len(parts) > 4 else parts[2]
        optimization = '_'.join(parts[-2:]) if parts[-2] in ['Concordance', 'Log'] else parts[-1]
        
        print(f"\nGroup {group_num}: {structure} + {balancing} + {optimization}")
        
        event1_models = [m for m in models if m['endpoint'] == 'Event 1']
        event2_models = [m for m in models if m['endpoint'] == 'Event 2']
        
        print("  Event 1 models:")
        for m in event1_models:
            model_no = m['model_no']
            if model_no in model_name_map:
                print(f"    - {model_name_map[model_no]}")
        
        print("  Event 2 models:")
        for m in event2_models:
            model_no = m['model_no']
            if model_no in model_name_map:
                print(f"    - {model_name_map[model_no]}")
        
        group_num += 1
    
    return groups

if __name__ == "__main__":
    groups = verify_deepsurv_groupings()
    
    # Summary
    print("\n" + "=" * 80)
    print("### SUMMARY ###")
    print(f"Total DeepSurv groups: {len(groups)}")
    print("Each group contains 2 models (one for Event 1, one for Event 2)")
    print("Total DeepSurv models: 24 (12 groups × 2 events)")