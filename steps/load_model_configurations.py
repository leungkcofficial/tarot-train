"""
Load Model Configurations Step for Ensemble Deployment

This module contains the ZenML step for loading model configurations from CSV and JSON files.
"""

import os
import json
import pandas as pd
import numpy as np
from zenml.steps import step
from typing import List, Dict, Any, Tuple
from datetime import datetime


@step(enable_cache=True)
def load_model_configurations(
    config_csv_path: str = "results/final_deploy/model_config/model_config.csv",
    config_dir: str = "results/final_deploy/model_config",
    output_dir: str = "results/final_deploy/temp"
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load model configurations from CSV and corresponding JSON files.
    
    This step reads the model_config.csv file to get model metadata and loads
    the corresponding JSON files containing model details and optimization metrics.
    It also saves the configurations to a JSON file for use in subsequent steps.
    
    Args:
        config_csv_path: Path to model_config.csv file
        config_dir: Directory containing model JSON files
        output_dir: Directory to save the configurations JSON file
        
    Returns:
        Tuple of (model_configs list, path to saved JSON file)
    """
    print(f"\n=== Loading Model Configurations ===")
    print(f"CSV path: {config_csv_path}")
    print(f"Config directory: {config_dir}")
    
    # Check if CSV file exists
    if not os.path.exists(config_csv_path):
        raise FileNotFoundError(f"Model configuration CSV not found at {config_csv_path}")
    
    # Check if config directory exists
    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"Model configuration directory not found at {config_dir}")
    
    # Read CSV file
    model_df = pd.read_csv(config_csv_path)
    print(f"Found {len(model_df)} models in configuration CSV")
    
    # Display CSV structure
    print("\nModel configuration CSV columns:")
    for col in model_df.columns:
        print(f"  - {col}")
    
    model_configs = []
    
    for idx, row in model_df.iterrows():
        model_no = row['Model No.']
        algorithm = row['Algorithm']
        structure = row['Structure']
        balancing = row['Balancing Method']
        endpoint = row['Prediction Endpoint']
        optimization = row['Optimization target']
        
        print(f"\nProcessing Model {model_no}: {algorithm} - {structure}")
        
        # Find corresponding JSON files
        details_json_files = [
            f for f in os.listdir(config_dir) 
            if f.startswith(f"model{model_no}_details_") and f.endswith('.json')
        ]
        optim_json_files = [
            f for f in os.listdir(config_dir) 
            if f.startswith(f"model{model_no}_optimization_metrics_") and f.endswith('.json')
        ]
        
        if not details_json_files:
            print(f"Warning: No details JSON file found for model {model_no}, skipping...")
            continue
            
        # Use the first matching file (should only be one)
        details_json_file = details_json_files[0]
        details_json_path = os.path.join(config_dir, details_json_file)
        
        # Load model details JSON
        try:
            with open(details_json_path, 'r') as f:
                model_details = json.load(f)
            print(f"  Loaded details from: {details_json_file}")
        except Exception as e:
            print(f"  Error loading details JSON for model {model_no}: {e}")
            continue
        
        # Load optimization metrics JSON if available
        optimization_details = {}
        if optim_json_files:
            optim_json_file = optim_json_files[0]
            optim_json_path = os.path.join(config_dir, optim_json_file)
            try:
                with open(optim_json_path, 'r') as f:
                    optimization_details = json.load(f)
                print(f"  Loaded optimization metrics from: {optim_json_file}")
            except Exception as e:
                print(f"  Warning: Could not load optimization metrics for model {model_no}: {e}")
        
        # Handle NaN balancing method
        if pd.isna(balancing) or str(balancing).lower() == 'nan':
            balancing = 'None'
        
        # Create combined configuration
        config = {
            'model_no': model_no,
            'algorithm': algorithm,
            'structure': structure,
            'balancing_method': balancing,
            'prediction_endpoint': endpoint,
            'optimization_target': optimization,
            'model_details': model_details,
            'optimization_details': optimization_details,
            'details_json_file': details_json_file,
            'optim_json_file': optim_json_files[0] if optim_json_files else None
        }
        
        # Add some derived information
        config['model_type'] = algorithm.lower()
        config['network_type'] = structure.lower()
        
        # Determine target endpoint for DeepSurv models
        if algorithm.lower() == 'deepsurv':
            if endpoint == 'Event 1':
                config['target_endpoint'] = 1
            elif endpoint == 'Event 2':
                config['target_endpoint'] = 2
            else:
                print(f"  Warning: Unknown endpoint '{endpoint}' for DeepSurv model")
                config['target_endpoint'] = 1  # Default
        else:
            # DeepHit models predict both endpoints
            config['target_endpoint'] = None
        
        model_configs.append(config)
        
        # Display key configuration details
        print(f"  Model type: {config['model_type']}")
        print(f"  Network type: {config['network_type']}")
        print(f"  Target endpoint: {config['target_endpoint']}")
        print(f"  Balancing method: {config['balancing_method']}")
        print(f"  Optimization target: {config['optimization_target']}")
    
    print(f"\n=== Successfully loaded {len(model_configs)} model configurations ===")
    
    # Summary statistics
    algorithms = pd.Series([c['algorithm'] for c in model_configs]).value_counts()
    structures = pd.Series([c['structure'] for c in model_configs]).value_counts()
    
    print("\nModel distribution:")
    print("\nBy algorithm:")
    for alg, count in algorithms.items():
        print(f"  {alg}: {count}")
    
    print("\nBy structure:")
    for struct, count in structures.items():
        print(f"  {struct}: {count}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configurations to JSON file for use in subsequent steps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"model_configs_{timestamp}.json")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_configs = convert_to_serializable(model_configs)
    
    with open(json_path, 'w') as f:
        json.dump(serializable_configs, f, indent=2)
    
    print(f"\nSaved model configurations to: {json_path}")
    
    return model_configs, json_path