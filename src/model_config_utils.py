"""
Utility functions for model configuration management in ensemble deployment.
"""

import pandas as pd
from typing import Dict, Any, Optional


def create_model_specific_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a model-specific configuration dictionary that mimics hyperparameter_config.yml structure.
    
    This function takes a model configuration loaded from CSV/JSON and transforms it into
    the format expected by the training pipeline.
    
    Args:
        model_config: Model configuration from load_model_configurations step
        
    Returns:
        Configuration dictionary with model-specific settings
    """
    # Extract information from model config
    algorithm = model_config['algorithm'].lower()
    structure = model_config['structure'].lower()
    balancing_method = model_config['balancing_method']
    optimization_target = model_config['optimization_target']
    model_details = model_config['model_details']
    optimization_details = model_config.get('optimization_details', {})
    
    # Handle NaN balancing method
    if pd.isna(balancing_method) or str(balancing_method).lower() == 'nan' or balancing_method == 'None':
        balancing_method = None
    
    # Parse balancing configuration
    balance_config = {
        'enable': False,
        'method': 'random_under_sampler',
        'sampling_strategy': 'majority'
    }
    
    if balancing_method:
        balance_config['enable'] = True
        
        if 'NearMiss' in str(balancing_method):
            balance_config['method'] = 'near_miss'
            # Extract version number from "NearMiss version X"
            if 'version' in str(balancing_method):
                try:
                    version = int(balancing_method.split()[-1])
                    balance_config['near_miss_version'] = version
                except:
                    balance_config['near_miss_version'] = 1
        elif balancing_method == 'KNN':
            balance_config['method'] = 'enn'  # KNN maps to Edited Nearest Neighbors
        else:
            balance_config['method'] = 'random_under_sampler'
    
    # Get sequence length from optimization details for LSTM
    sequence_length = 5  # default
    if structure.lower() == 'lstm' and optimization_details:
        if 'best_params' in optimization_details and 'sequence' in optimization_details['best_params']:
            sequence_length = optimization_details['best_params']['sequence']
    
    # Extract hyperparameters from optimization details
    best_params = optimization_details.get('best_params', {}) if optimization_details else {}
    
    # Create configuration structure
    config = {
        'model_type': algorithm,
        'target_endpoint': model_config.get('target_endpoint'),
        'network': {
            'type': structure,
            'default': {
                'hidden_dims': model_details.get('hidden_dims', [128, 64, 32]),
                'num_layers': len(model_details.get('hidden_dims', [128, 64, 32])),
                'dropout': model_details.get('dropout', 0.2),
                'batch_size': best_params.get('batch_size', 64),
                'learning_rate': best_params.get('learning_rate', 0.001),
                'epochs': model_details.get('epochs', 100)
            },
            'lstm': {
                'hidden_dims': model_details.get('lstm_hidden_dims', [128, 64, 32]),
                'num_layers': model_details.get('lstm_num_layers', 2),
                'bidirectional': model_details.get('bidirectional', True),
                'sequence_length': sequence_length
            },
            'deephit': {
                'alpha': model_details.get('alpha', 0.2),
                'sigma': model_details.get('sigma', 0.1),
                'time_grid': model_details.get('time_grid', [365, 730, 1095, 1460, 1825])
            }
        },
        'optimization': {
            'n_trials': 1,  # We're not optimizing, just using the found parameters
            'patience': 10,
            'seed': 42,
            'metric': 'cidx' if 'Concordance' in optimization_target else 'loglik'
        },
        'balance': balance_config
    }
    
    # Add optimizer information
    if best_params.get('optimizer'):
        config['optimizer'] = best_params['optimizer']
    
    return config


def save_model_specific_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save a model-specific configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the YAML file
    """
    import yaml
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Saved model configuration to {output_path}")


def get_model_identifier(model_config: Dict[str, Any]) -> str:
    """
    Generate a unique identifier for a model based on its configuration.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        String identifier for the model
    """
    model_no = model_config['model_no']
    algorithm = model_config['algorithm']
    structure = model_config['structure']
    endpoint = model_config.get('prediction_endpoint', 'Both')
    
    # Create a clean identifier
    identifier = f"model{model_no}_{algorithm}_{structure}_{endpoint}".replace(' ', '_')
    
    return identifier