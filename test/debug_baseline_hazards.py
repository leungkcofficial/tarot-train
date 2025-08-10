import pickle
import numpy as np

# Load baseline hazards for model 13
with open('results/final_deploy/models/baseline_hazards_model13_20250804.pkl', 'rb') as f:
    baseline_hazards = pickle.load(f)

print("Baseline hazards type:", type(baseline_hazards))
print("Baseline hazards shape:", baseline_hazards.shape if hasattr(baseline_hazards, 'shape') else 'N/A')

if isinstance(baseline_hazards, dict):
    print("\nBaseline hazards keys:", baseline_hazards.keys())
    for key, value in baseline_hazards.items():
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if hasattr(value, 'shape'):
            print(f"  Shape: {value.shape}")
        elif hasattr(value, '__len__'):
            print(f"  Length: {len(value)}")
        if isinstance(value, np.ndarray):
            print(f"  First few values: {value[:5]}")
            print(f"  Last few values: {value[-5:]}")

# Also check the time grid from model config
import json
with open('results/final_deploy/model_config/model13_details_20250720_005439.json', 'r') as f:
    config = json.load(f)
    
print("\n\nModel configuration:")
print(f"Sequence length: {config['sequence_length']}")
print(f"Time grid: {config['time_grid']}")
print(f"Time grid length: {len(config['time_grid'])}")