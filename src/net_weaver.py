"""
Neural Network Architecture Module for CKD Risk Prediction

This module provides a flexible way to create different neural network architectures
for the CKD risk prediction task. It includes various network architectures and
factory methods to create them based on configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np


# Note: MLPNetwork is equivalent to a basic ANN (Artificial Neural Network)
# The LSTM network is added for sequential data processing

class BaseNetwork(nn.Module):
    """Base class for all neural network architectures."""
    
    def __init__(self, input_dim: int, output_dim: int = 1):
        """
        Initialize the base network.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes (default: 1 for binary classification)
        """
        super(BaseNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the network.
        
        Returns:
            Dictionary containing the network configuration
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "type": self.__class__.__name__
        }


class MLPNetwork(BaseNetwork):
    """Multi-Layer Perceptron (MLP) network."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        output_dim: int = 1,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        use_batch_norm: bool = True
    ):
        """
        Initialize the MLP network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output classes (default: 1 for binary classification)
            dropout_rate: Dropout rate (default: 0.2)
            activation: Activation function to use (default: "relu")
            use_batch_norm: Whether to use batch normalization (default: True)
        """
        super(MLPNetwork, self).__init__(input_dim, output_dim)
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm
        
        # Create the network layers
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            # Add linear layer
            layers.append(nn.Linear(prev_dim, dim))
            
            # Add batch normalization if specified
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            
            # Add activation function
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "selu":
                layers.append(nn.SELU())
            else:
                layers.append(nn.ReLU())  # Default to ReLU
            
            # Add dropout
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # If binary classification, add sigmoid activation
        if output_dim == 1:
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the network.
        
        Returns:
            Dictionary containing the network configuration
        """
        config = super().get_config()
        config.update({
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation_name,
            "use_batch_norm": self.use_batch_norm
        })
        return config


class ResidualBlock(nn.Module):
    """Residual block for deep networks."""
    
    def __init__(
        self, 
        dim: int, 
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Initialize the residual block.
        
        Args:
            dim: Dimension of the input and output
            dropout_rate: Dropout rate (default: 0.2)
            use_batch_norm: Whether to use batch normalization (default: True)
        """
        super(ResidualBlock, self).__init__()
        
        # First layer
        layers = []
        layers.append(nn.Linear(dim, dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Second layer
        layers.append(nn.Linear(dim, dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        identity = x
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out


class ResNetNetwork(BaseNetwork):
    """Residual Network (ResNet) for tabular data."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int,
        num_blocks: int,
        output_dim: int = 1,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Initialize the ResNet network.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Dimension of the hidden layers
            num_blocks: Number of residual blocks
            output_dim: Number of output classes (default: 1 for binary classification)
            dropout_rate: Dropout rate (default: 0.2)
            use_batch_norm: Whether to use batch normalization (default: True)
        """
        super(ResNetNetwork, self).__init__(input_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate, use_batch_norm)
            for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid() if output_dim == 1 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        x = self.input_layer(x)
        x = F.relu(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_layer(x)
        
        if self.sigmoid is not None:
            x = self.sigmoid(x)
        
        return x
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the network.
        
        Returns:
            Dictionary containing the network configuration
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm
        })
        return config


class LSTMNetwork(BaseNetwork):
    """Long Short-Term Memory (LSTM) network for sequential data."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout_rate: float = 0.2,
        bidirectional: bool = False,
        sequence_length: int = 10
    ):
        """
        Initialize the LSTM network.
        
        Args:
            input_dim: Number of input features per time step
            hidden_dim: Dimension of the hidden state
            num_layers: Number of LSTM layers (default: 2)
            output_dim: Number of output classes (default: 1 for binary classification)
            dropout_rate: Dropout rate (default: 0.2)
            bidirectional: Whether to use bidirectional LSTM (default: False)
            sequence_length: Length of input sequences (default: 10)
        """
        super(LSTMNetwork, self).__init__(input_dim, output_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid() if output_dim == 1 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
                Note: For LSTM, we reshape this to (batch_size, sequence_length, input_features)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # Reshape input for LSTM if it's not already in the right shape
        if len(x.shape) == 2:
            # Determine how to reshape based on input dimensions
            if x.size(1) % self.sequence_length == 0:
                # If input_dim is divisible by sequence_length, reshape accordingly
                features_per_step = x.size(1) // self.sequence_length
                x = x.view(batch_size, self.sequence_length, features_per_step)
            else:
                # Otherwise, use a default approach: treat each feature as a time step
                # and pad or truncate to sequence_length
                if x.size(1) >= self.sequence_length:
                    # Truncate
                    x = x[:, :self.sequence_length].unsqueeze(2)
                else:
                    # Pad with zeros
                    padding = torch.zeros(batch_size, self.sequence_length - x.size(1), device=x.device)
                    x = torch.cat([x, padding], dim=1).unsqueeze(2)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(lstm_out)
        
        # Apply sigmoid for binary classification
        if self.sigmoid is not None:
            out = self.sigmoid(out)
        
        return out
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the network.
        
        Returns:
            Dictionary containing the network configuration
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "bidirectional": self.bidirectional,
            "sequence_length": self.sequence_length
        })
        return config


class TabNetNetwork(BaseNetwork):
    """TabNet network for tabular data."""
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int = 1,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 2,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        virtual_batch_size: int = 128,
        momentum: float = 0.02
    ):
        """
        Initialize the TabNet network.
        
        This is a simplified implementation of TabNet for demonstration purposes.
        For a full implementation, consider using the official TabNet implementation
        or a library like pytorch-tabnet.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes (default: 1 for binary classification)
            n_d: Dimension of the decision prediction layer (default: 64)
            n_a: Dimension of the attention embedding for each step (default: 64)
            n_steps: Number of sequential attention steps (default: 3)
            gamma: Coefficient for feature reusage (default: 1.3)
            n_independent: Number of independent GLU layers (default: 2)
            n_shared: Number of shared GLU layers (default: 2)
            epsilon: Avoid log(0) (default: 1e-15)
            virtual_batch_size: Batch size for Ghost Batch Normalization (default: 128)
            momentum: Momentum for Ghost Batch Normalization (default: 0.02)
        """
        super(TabNetNetwork, self).__init__(input_dim, output_dim)
        
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.epsilon = epsilon
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        
        # For simplicity, we'll use a simpler network structure here
        # In a real implementation, you would implement the full TabNet architecture
        
        self.feature_transformer = nn.Sequential(
            nn.Linear(input_dim, 2 * (n_d + n_a)),
            nn.BatchNorm1d(2 * (n_d + n_a)),
            nn.ReLU()
        )
        
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_d + n_a, n_a),
                nn.BatchNorm1d(n_a),
                nn.ReLU()
            )
            for _ in range(n_steps)
        ])
        
        self.output_layer = nn.Linear(n_d * n_steps, output_dim)
        
        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid() if output_dim == 1 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # Initial feature transformation
        x_transformed = self.feature_transformer(x)
        
        # Split into decision and attention components
        d_a = x_transformed[:, :self.n_d + self.n_a]
        
        # Collect outputs from each step
        step_outputs = []
        
        for step in range(self.n_steps):
            # Get attention weights
            a = self.attention_layers[step](d_a)
            a = torch.sigmoid(a)
            
            # Apply attention weights
            masked_x = x * a
            
            # Transform masked input
            masked_x_transformed = self.feature_transformer(masked_x)
            
            # Get decision output for this step
            d = masked_x_transformed[:, :self.n_d]
            step_outputs.append(d)
            
            # Update d_a for next step
            if step < self.n_steps - 1:
                d_a = masked_x_transformed[:, self.n_d:self.n_d + self.n_a]
        
        # Concatenate outputs from all steps
        out = torch.cat(step_outputs, dim=1)
        
        # Final output layer
        out = self.output_layer(out)
        
        if self.sigmoid is not None:
            out = self.sigmoid(out)
        
        return out
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the network.
        
        Returns:
            Dictionary containing the network configuration
        """
        config = super().get_config()
        config.update({
            "n_d": self.n_d,
            "n_a": self.n_a,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "n_independent": self.n_independent,
            "n_shared": self.n_shared,
            "virtual_batch_size": self.virtual_batch_size,
            "momentum": self.momentum
        })
        return config


def create_network(
    network_type: str,
    input_dim: int,
    output_dim: int = 1,
    **kwargs
) -> BaseNetwork:
    """
    Create a neural network based on the specified type and parameters.
    
    Args:
        network_type: Type of network to create ("mlp", "ann", "resnet", "lstm", "tabnet")
        input_dim: Number of input features
        output_dim: Number of output classes (default: 1 for binary classification)
        **kwargs: Additional parameters for the specific network type
        
    Returns:
        Neural network instance
        
    Raises:
        ValueError: If the network type is not supported
    """
    if network_type.lower() == "mlp" or network_type.lower() == "ann":
        # MLP is equivalent to ANN (Artificial Neural Network)
        hidden_dims = kwargs.get("hidden_dims", [128, 64, 32])
        dropout_rate = kwargs.get("dropout_rate", 0.2)
        activation = kwargs.get("activation", "relu")
        use_batch_norm = kwargs.get("use_batch_norm", True)
        
        return MLPNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
    
    elif network_type.lower() == "resnet":
        hidden_dim = kwargs.get("hidden_dim", 128)
        num_blocks = kwargs.get("num_blocks", 3)
        dropout_rate = kwargs.get("dropout_rate", 0.2)
        use_batch_norm = kwargs.get("use_batch_norm", True)
        
        return ResNetNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        )
    
    elif network_type.lower() == "lstm":
        hidden_dim = kwargs.get("hidden_dim", 128)
        num_layers = kwargs.get("num_layers", 2)
        dropout_rate = kwargs.get("dropout_rate", 0.2)
        bidirectional = kwargs.get("bidirectional", False)
        sequence_length = kwargs.get("sequence_length", 10)
        
        return LSTMNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            sequence_length=sequence_length
        )
    
    elif network_type.lower() == "tabnet":
        n_d = kwargs.get("n_d", 64)
        n_a = kwargs.get("n_a", 64)
        n_steps = kwargs.get("n_steps", 3)
        gamma = kwargs.get("gamma", 1.3)
        n_independent = kwargs.get("n_independent", 2)
        n_shared = kwargs.get("n_shared", 2)
        virtual_batch_size = kwargs.get("virtual_batch_size", 128)
        momentum = kwargs.get("momentum", 0.02)
        
        return TabNetNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum
        )
    
    else:
        raise ValueError(f"Unsupported network type: {network_type}")


def get_available_networks() -> List[str]:
    """
    Get a list of available network types.
    
    Returns:
        List of available network types
    """
    return ["mlp", "ann", "resnet", "lstm", "tabnet"]