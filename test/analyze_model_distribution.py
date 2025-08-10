import json
from collections import defaultdict

def analyze_model_distribution():
    """Analyze the distribution of models by type, architecture, and event."""
    
    # Load the model paths data
    with open('all_model_paths.json', 'r') as f:
        all_paths = json.load(f)
    
    # Load the full model data
    json_file = '/home/goma/.config/zenml/local_stores/f53346f0-79a8-49e4-b4ed-1cfb14dfa3fb/process_models_sequentially/output_0/67c9050f-777c-40ad-b087-a4bd53d6c47a/99a24532/data.json'
    with open(json_file, 'r') as f:
        models = json.load(f)
    
    # Analyze distribution
    distribution = defaultdict(lambda: defaultdict(int))
    model_details = []
    
    for model in models:
        model_type = model['model_type']
        model_name = model['model_name']
        
        # Extract architecture from model name
        if 'ANN' in model_name:
            architecture = 'ANN'
        elif 'LSTM' in model_name:
            architecture = 'LSTM'
        else:
            architecture = 'Unknown'
        
        # Extract event
        if model_type == 'deepsurv':
            event = f"Event_{model['model_endpoint']}"
        else:  # deephit
            event = 'Both_Events'
        
        distribution[model_type][f"{architecture}_{event}"] += 1
        
        model_details.append({
            'name': model_name,
            'type': model_type,
            'architecture': architecture,
            'event': event,
            'model_path': model['original_model_path']
        })
    
    # Print distribution
    print("### MODEL DISTRIBUTION ANALYSIS ###\n")
    
    print("Overall:")
    print(f"- Total models: {len(models)}")
    print(f"- DeepSurv models: {sum(distribution['deepsurv'].values())}")
    print(f"- DeepHit models: {sum(distribution['deephit'].values())}")
    
    print("\nDetailed Distribution:")
    print("\nDeepSurv Models (24 total):")
    for key, count in sorted(distribution['deepsurv'].items()):
        print(f"  - {key}: {count}")
    
    print("\nDeepHit Models (12 total):")
    for key, count in sorted(distribution['deephit'].items()):
        print(f"  - {key}: {count}")
    
    # Group models by characteristics
    print("\n### MODELS GROUPED BY CHARACTERISTICS ###")
    
    # Group DeepSurv by architecture and event
    deepsurv_groups = defaultdict(list)
    for model in model_details:
        if model['type'] == 'deepsurv':
            key = f"{model['architecture']}_{model['event']}"
            deepsurv_groups[key].append(model['name'])
    
    print("\nDeepSurv Groups:")
    for group, names in sorted(deepsurv_groups.items()):
        print(f"\n{group} ({len(names)} models):")
        for name in sorted(names):
            print(f"  - {name}")
    
    # Group DeepHit by architecture
    deephit_groups = defaultdict(list)
    for model in model_details:
        if model['type'] == 'deephit':
            key = model['architecture']
            deephit_groups[key].append(model['name'])
    
    print("\nDeepHit Groups:")
    for group, names in sorted(deephit_groups.items()):
        print(f"\n{group} ({len(names)} models):")
        for name in sorted(names):
            print(f"  - {name}")
    
    return distribution, model_details

if __name__ == "__main__":
    distribution, model_details = analyze_model_distribution()