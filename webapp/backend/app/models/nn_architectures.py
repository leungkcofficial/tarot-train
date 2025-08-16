"""
Neural Network Architectures for TAROT CKD Risk Prediction Models
Implements MLP and LSTM networks compatible with PyCox survival models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
import numpy as np


class MLP(nn.Module):
    """Multi-Layer Perceptron network for survival analysis"""
    
    def __init__(self, 
                 in_features: int,
                 hidden_dims: List[int],
                 out_features: int = 1,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 batch_norm: bool = True):
        """
        Args:
            in_features: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            out_features: Output dimension (1 for survival, 2+ for competing risks)
            dropout: Dropout rate
            activation: Activation function ('relu', 'elu', 'selu')
            batch_norm: Whether to use batch normalization
        """
        super(MLP, self).__init__()
        
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.out_features = out_features
        self.dropout_rate = dropout
        
        # Build layers
        layers = []
        prev_dim = in_features
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'elu':
                layers.append(nn.ELU(inplace=True))
            elif activation == 'selu':
                layers.append(nn.SELU(inplace=True))
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, out_features))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)


class LSTMSurvival(nn.Module):
    """LSTM network for sequential survival analysis"""
    
    def __init__(self,
                 input_dim: int,
                 sequence_length: int,
                 lstm_hidden_dims: List[int],
                 output_dim: int = 1,
                 dropout: float = 0.1,
                 bidirectional: bool = False,
                 num_layers: int = None):
        """
        Args:
            input_dim: Input feature dimension
            sequence_length: Length of input sequences
            lstm_hidden_dims: List of LSTM hidden dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            num_layers: Number of LSTM layers (if None, use len(lstm_hidden_dims))
        """
        super(LSTMSurvival, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.lstm_hidden_dims = lstm_hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        
        # Determine number of LSTM layers
        if num_layers is None:
            num_layers = len(lstm_hidden_dims) if lstm_hidden_dims else 1
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(lstm_hidden_dims):
            lstm = nn.LSTM(
                input_size=prev_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0,  # We'll add dropout separately
                bidirectional=bidirectional
            )
            self.lstm_layers.append(lstm)
            
            # Account for bidirectional doubling
            prev_dim = hidden_dim * (2 if bidirectional else 1)
            
            # Add dropout after LSTM (except last layer)
            if dropout > 0 and i < len(lstm_hidden_dims) - 1:
                self.lstm_layers.append(nn.Dropout(dropout))
        
        # If no LSTM hidden dims specified, create single LSTM
        if not lstm_hidden_dims:
            hidden_dim = 64  # Default
            lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
            self.lstm_layers.append(lstm)
            prev_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize LSTM and linear layer weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize output layer
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_out', nonlinearity='relu')
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
               or (batch_size, input_dim) which will be unsqueezed
        """
        # Handle 2D input by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Pass through LSTM layers
        for layer in self.lstm_layers:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)  # x: (batch_size, seq_len, hidden_dim)
            elif isinstance(layer, nn.Dropout):
                x = layer(x)
        
        # Use last time step for prediction
        if x.size(1) > 1:  # If sequence length > 1
            x = x[:, -1, :]  # Take last time step: (batch_size, hidden_dim)
        else:
            x = x.squeeze(1)  # Remove sequence dimension: (batch_size, hidden_dim)
        
        # Output layer
        output = self.output_layer(x)  # (batch_size, output_dim)
        
        return output


def create_network_from_state_dict(state_dict: dict, model_path: str = None) -> nn.Module:
    """
    Create neural network by inferring architecture from state dict
    
    Args:
        state_dict: PyTorch model state dictionary
        model_path: Optional model file path for additional context
        
    Returns:
        Neural network module
    """
    # Check if LSTM or MLP
    is_lstm = any('lstm' in key.lower() for key in state_dict.keys())
    
    if is_lstm:
        return _create_lstm_from_state_dict(state_dict)
    else:
        return _create_mlp_from_state_dict(state_dict)


def _create_mlp_from_state_dict(state_dict: dict) -> MLP:
    """Create MLP network from state dict"""
    # Find all linear layer weights (excluding batch norm)
    layer_keys = []
    for key in state_dict.keys():
        if '.weight' in key and not any(x in key.lower() for x in ['batch_norm', 'bn']):
            layer_keys.append(key)
    
    layer_keys.sort()
    
    if len(layer_keys) < 1:
        raise ValueError("No linear layers found in state dict")
    
    # Get dimensions
    first_weight = state_dict[layer_keys[0]]
    input_dim = first_weight.shape[1]
    
    hidden_dims = []
    for i, key in enumerate(layer_keys[:-1]):  # Exclude output layer
        hidden_dims.append(state_dict[key].shape[0])
    
    # Output dimension
    last_weight = state_dict[layer_keys[-1]]
    output_dim = last_weight.shape[0]
    
    # Check for batch norm
    has_batch_norm = any('batch_norm' in key or 'bn' in key for key in state_dict.keys())
    
    # Create MLP
    mlp = MLP(
        in_features=input_dim,
        hidden_dims=hidden_dims,
        out_features=output_dim,
        dropout=0.1,  # Default, will be overridden by state dict
        batch_norm=has_batch_norm
    )
    
    return mlp


def _create_lstm_from_state_dict(state_dict: dict) -> LSTMSurvival:
    """Create LSTM network from state dict"""
    # Find LSTM layers
    lstm_keys = [k for k in state_dict.keys() if 'lstm' in k.lower() and 'weight_ih_l0' in k]
    
    if not lstm_keys:
        # Try alternative pattern
        lstm_keys = [k for k in state_dict.keys() if 'lstm' in k.lower() and 'weight_ih' in k]
    
    if not lstm_keys:
        raise ValueError("No LSTM layers found in state dict")
    
    # Get first LSTM layer dimensions
    first_lstm_key = lstm_keys[0]
    weight_ih = state_dict[first_lstm_key]
    
    input_dim = weight_ih.shape[1]
    hidden_dim = weight_ih.shape[0] // 4  # LSTM has 4 gates (i, f, g, o)
    
    # For simplicity, assume single LSTM layer with the detected hidden dimension
    lstm_hidden_dims = [hidden_dim]
    
    # Find output layer
    output_keys = [k for k in state_dict.keys() if 'output_layer' in k and '.weight' in k]
    if not output_keys:
        output_keys = [k for k in state_dict.keys() if k.endswith('.weight') and 'lstm' not in k.lower()]
    
    output_dim = 1  # Default
    if output_keys:
        output_weight = state_dict[output_keys[0]]
        output_dim = output_weight.shape[0]
    
    # Create LSTM network
    lstm_net = LSTMSurvival(
        input_dim=input_dim,
        sequence_length=1,  # Default for single time point
        lstm_hidden_dims=lstm_hidden_dims,
        output_dim=output_dim,
        dropout=0.1  # Default, will be overridden by state dict
    )
    
    return lstm_net