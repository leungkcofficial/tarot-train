"""
Model Initialization Step for CKD Risk Prediction

This module contains the ZenML step for initializing PyTorch neural networks for
deep learning survival analysis models (DeepSurv and DeepHit).
"""

import torch
import mlflow
from zenml.steps import step
from typing import Dict, List, Any, Optional

# Import neural network architectures
from src.nn_architectures import create_network


@step
def model_init(
    model_type: str,
    input_dim: int,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.2,
    output_dim: int = 1,
    activation: str = 'relu',
    batch_norm: bool = True,
    num_causes: int = 2,
    log_to_mlflow: bool = True
) -> torch.nn.Module:
    """
    Initialize a PyTorch neural network for survival analysis.
    
    Args:
        model_type: Type of model ("deepsurv" or "deephit")
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions (default: [128, 64, 32])
        dropout: Dropout rate (default: 0.2)
        output_dim: Output dimension (default: 1 for DeepSurv, num_durations for DeepHit)
        activation: Activation function (default: 'relu')
        batch_norm: Whether to use batch normalization (default: True)
        num_causes: Number of competing risks for DeepHit (default: 2)
        log_to_mlflow: Whether to log model architecture to MLflow (default: True)
        
    Returns:
        Initialized PyTorch neural network
    """
    try:
        print(f"\n=== Initializing {model_type.upper()} model ===\n")
        
        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
            print(f"Using default hidden dimensions: {hidden_dims}")
        else:
            print(f"Using provided hidden dimensions: {hidden_dims}")
        
        # Validate model type
        if model_type.lower() not in ["deepsurv", "deephit"]:
            raise ValueError(f"Unsupported model type: {model_type}. Must be 'deepsurv' or 'deephit'.")
        
        # For DeepHit, output_dim should be the number of time intervals
        if model_type.lower() == "deephit" and output_dim == 1:
            print("Warning: output_dim=1 is too small for DeepHit. Setting to default value of 10.")
            output_dim = 10
        
        # Print model configuration
        print(f"Model type: {model_type}")
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dimensions: {hidden_dims}")
        print(f"Output dimension: {output_dim}")
        print(f"Dropout rate: {dropout}")
        print(f"Activation function: {activation}")
        print(f"Batch normalization: {batch_norm}")
        if model_type.lower() == "deephit":
            print(f"Number of competing risks: {num_causes}")
        
        # Create network
        net = create_network(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
            num_causes=num_causes
        )
        
        # Print model summary
        print("\nModel architecture:")
        print(net)
        
        # Count number of parameters
        num_params = sum(p.numel() for p in net.parameters())
        print(f"\nTotal number of parameters: {num_params:,}")
        
        # Log model architecture to MLflow
        if log_to_mlflow:
            # Get model configuration
            model_config = {
                'model_type': model_type,
                'input_dim': input_dim,
                'hidden_dims': hidden_dims,
                'output_dim': output_dim,
                'dropout': dropout,
                'activation': activation,
                'batch_norm': batch_norm,
                'num_causes': num_causes if model_type.lower() == "deephit" else None,
                'num_parameters': num_params
            }
            
            # Log model configuration as parameters
            for key, value in model_config.items():
                if value is not None:
                    mlflow.log_param(f"model_{key}", value)
            
            # Log model architecture as text
            mlflow.log_text(str(net), "model_architecture.txt")
        
        return net
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        raise