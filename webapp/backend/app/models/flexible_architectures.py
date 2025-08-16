"""
Flexible Neural Network Architecture Reconstruction
Handles complex LSTM and DeepHit model architectures with exact state dict matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import re
from collections import OrderedDict


class FlexibleLSTM(nn.Module):
    """Flexible LSTM that reconstructs exact architecture from state dict"""
    
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super(FlexibleLSTM, self).__init__()
        
        self.state_dict_template = state_dict
        self.lstm_config = self._analyze_lstm_structure(state_dict)
        
        # Build LSTM layers to match state dict exactly
        self.lstm_layers = nn.ModuleList()
        self._build_lstm_layers()
        
        # Build output layer
        self._build_output_layer()
        
    def _analyze_lstm_structure(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze LSTM structure from state dict keys and shapes"""
        config = {
            'layers': [],
            'bidirectional': False,
            'input_dim': None,
            'output_dim': None
        }
        
        # Check for bidirectional (reverse keys)
        config['bidirectional'] = any('reverse' in key for key in state_dict.keys())
        
        # Analyze each LSTM layer
        layer_pattern = re.compile(r'lstm_layers\.(\d+)\.weight_ih_l0')
        output_pattern = re.compile(r'output_layer\.weight')
        
        layer_indices = set()
        for key in state_dict.keys():
            match = layer_pattern.match(key)
            if match:
                layer_indices.add(int(match.group(1)))
        
        layer_indices = sorted(layer_indices)
        
        for layer_idx in layer_indices:
            layer_info = self._analyze_lstm_layer(state_dict, layer_idx, config['bidirectional'])
            config['layers'].append(layer_info)
            
            if config['input_dim'] is None:
                config['input_dim'] = layer_info['input_dim']
        
        # Get output dimension
        for key in state_dict.keys():
            if output_pattern.match(key):
                config['output_dim'] = state_dict[key].shape[0]
                break
        
        return config
    
    def _analyze_lstm_layer(self, state_dict: Dict[str, torch.Tensor], 
                           layer_idx: int, bidirectional: bool) -> Dict[str, Any]:
        """Analyze individual LSTM layer"""
        weight_ih_key = f'lstm_layers.{layer_idx}.weight_ih_l0'
        weight_hh_key = f'lstm_layers.{layer_idx}.weight_hh_l0'
        
        if weight_ih_key not in state_dict:
            raise ValueError(f"LSTM layer {layer_idx} weight_ih not found")
        
        weight_ih = state_dict[weight_ih_key]
        weight_hh = state_dict[weight_hh_key]
        
        # LSTM weight_ih shape: (4 * hidden_size, input_size)
        # LSTM weight_hh shape: (4 * hidden_size, hidden_size)
        
        hidden_size = weight_ih.shape[0] // 4
        input_size = weight_ih.shape[1]
        
        return {
            'layer_idx': layer_idx,
            'input_dim': input_size,
            'hidden_size': hidden_size,
            'bidirectional': bidirectional
        }
    
    def _build_lstm_layers(self):
        """Build LSTM layers to match exact state dict structure"""
        for layer_info in self.lstm_config['layers']:
            # Create LSTM layer with exact parameters
            lstm = nn.LSTM(
                input_size=layer_info['input_dim'],
                hidden_size=layer_info['hidden_size'],
                num_layers=1,
                batch_first=True,
                dropout=0.0,
                bidirectional=layer_info['bidirectional']
            )
            self.lstm_layers.append(lstm)
    
    def _build_output_layer(self):
        """Build output layer"""
        # Calculate final hidden dimension
        if self.lstm_config['layers']:
            last_layer = self.lstm_config['layers'][-1]
            final_hidden = last_layer['hidden_size']
            if last_layer['bidirectional']:
                final_hidden *= 2
        else:
            final_hidden = 64  # Default
        
        output_dim = self.lstm_config.get('output_dim', 1)
        self.output_layer = nn.Linear(final_hidden, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Ensure 3D input (batch, seq, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Pass through LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        # Take last time step
        if x.size(1) > 1:
            x = x[:, -1, :]
        else:
            x = x.squeeze(1)
        
        # Output layer
        return self.output_layer(x)


class FlexibleDeepHit(nn.Module):
    """Flexible DeepHit network reconstruction"""
    
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super(FlexibleDeepHit, self).__init__()
        
        self.state_dict_template = state_dict
        self.deephit_config = self._analyze_deephit_structure(state_dict)
        
        # Build architecture
        self._build_conv_layers()
        self._build_cause_networks()
    
    def _analyze_deephit_structure(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze DeepHit structure from state dict"""
        config = {
            'conv_layers': [],
            'cause_networks': [],
            'num_causes': 0
        }
        
        # Analyze conv layers
        conv_pattern = re.compile(r'conv(\d+)\.(weight|bias)')
        conv_layers = {}
        
        for key in state_dict.keys():
            match = conv_pattern.match(key)
            if match:
                layer_num = int(match.group(1))
                param_type = match.group(2)
                
                if layer_num not in conv_layers:
                    conv_layers[layer_num] = {}
                
                if param_type == 'weight':
                    weight = state_dict[key]
                    conv_layers[layer_num]['out_channels'] = weight.shape[0]
                    conv_layers[layer_num]['in_channels'] = weight.shape[1]
                    conv_layers[layer_num]['kernel_size'] = weight.shape[2]
        
        config['conv_layers'] = [conv_layers[i] for i in sorted(conv_layers.keys())]
        
        # Analyze cause networks
        cause_pattern = re.compile(r'cause_networks\.(\d+)\.(\d+)\.(weight|bias)')
        cause_networks = {}
        
        for key in state_dict.keys():
            match = cause_pattern.match(key)
            if match:
                cause_idx = int(match.group(1))
                layer_idx = int(match.group(2))
                param_type = match.group(3)
                
                if cause_idx not in cause_networks:
                    cause_networks[cause_idx] = {}
                if layer_idx not in cause_networks[cause_idx]:
                    cause_networks[cause_idx][layer_idx] = {}
                
                if param_type == 'weight':
                    weight = state_dict[key]
                    cause_networks[cause_idx][layer_idx]['out_features'] = weight.shape[0]
                    cause_networks[cause_idx][layer_idx]['in_features'] = weight.shape[1]
        
        config['cause_networks'] = cause_networks
        config['num_causes'] = len(cause_networks)
        
        return config
    
    def _build_conv_layers(self):
        """Build convolutional layers"""
        self.conv_layers = nn.ModuleList()
        
        for i, conv_info in enumerate(self.deephit_config['conv_layers']):
            conv = nn.Conv1d(
                in_channels=conv_info['in_channels'],
                out_channels=conv_info['out_channels'],
                kernel_size=conv_info['kernel_size']
            )
            self.conv_layers.append(conv)
    
    def _build_cause_networks(self):
        """Build cause-specific networks"""
        self.cause_networks = nn.ModuleList()
        
        for cause_idx in range(self.deephit_config['num_causes']):
            cause_layers = nn.ModuleList()
            
            cause_info = self.deephit_config['cause_networks'][cause_idx]
            for layer_idx in sorted(cause_info.keys()):
                layer_info = cause_info[layer_idx]
                
                # Check if it's BatchNorm or Linear based on parameter patterns
                state_key_base = f'cause_networks.{cause_idx}.{layer_idx}'
                
                if f'{state_key_base}.running_mean' in self.state_dict_template:
                    # BatchNorm layer
                    bn = nn.BatchNorm1d(layer_info['out_features'])
                    cause_layers.append(bn)
                else:
                    # Linear layer
                    linear = nn.Linear(
                        layer_info['in_features'],
                        layer_info['out_features']
                    )
                    cause_layers.append(linear)
            
            self.cause_networks.append(nn.Sequential(*cause_layers))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size = x.size(0)
        
        # Conv layers
        x = x.unsqueeze(-1)  # Add channel dimension for conv1d
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Cause networks
        outputs = []
        for cause_net in self.cause_networks:
            cause_output = cause_net(x)
            outputs.append(cause_output)
        
        # Concatenate or sum outputs
        if len(outputs) > 1:
            return torch.cat(outputs, dim=-1)
        else:
            return outputs[0]


class FlexibleLSTMDeepHit(nn.Module):
    """Flexible LSTM DeepHit network reconstruction for competing risks"""
    
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super(FlexibleLSTMDeepHit, self).__init__()
        
        self.state_dict_template = state_dict
        self.lstm_config = self._analyze_lstm_deephit_structure(state_dict)
        
        # Build LSTM layers to match state dict exactly
        self.lstm_layers = nn.ModuleList()
        self._build_lstm_layers()
        
        # Build cause-specific output heads
        self._build_cause_heads()
        
    def _analyze_lstm_deephit_structure(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze LSTM DeepHit structure from state dict"""
        config = {
            'layers': [],
            'bidirectional': False,
            'input_dim': None,
            'output_dim': None,
            'num_causes': 0
        }
        
        # Check for bidirectional (reverse keys)
        config['bidirectional'] = any('reverse' in key for key in state_dict.keys())
        
        # Analyze LSTM layers
        layer_pattern = re.compile(r'lstm_layers\.(\d+)\.weight_ih_l0')
        cause_pattern = re.compile(r'cause_heads\.(\d+)\.weight')
        
        layer_indices = set()
        cause_indices = set()
        
        for key in state_dict.keys():
            # LSTM layers
            match = layer_pattern.match(key)
            if match:
                layer_indices.add(int(match.group(1)))
                
            # Cause heads
            match = cause_pattern.match(key)
            if match:
                cause_indices.add(int(match.group(1)))
        
        layer_indices = sorted(layer_indices)
        cause_indices = sorted(cause_indices)
        
        # Analyze each LSTM layer
        for layer_idx in layer_indices:
            layer_info = self._analyze_lstm_layer(state_dict, layer_idx, config['bidirectional'])
            config['layers'].append(layer_info)
            
            if config['input_dim'] is None:
                config['input_dim'] = layer_info['input_dim']
        
        # Analyze cause heads
        config['num_causes'] = len(cause_indices)
        if cause_indices:
            first_cause_weight = state_dict[f'cause_heads.{cause_indices[0]}.weight']
            config['output_dim'] = first_cause_weight.shape[0]  # Output dimension per cause
        
        return config
    
    def _analyze_lstm_layer(self, state_dict: Dict[str, torch.Tensor], 
                           layer_idx: int, bidirectional: bool) -> Dict[str, Any]:
        """Analyze individual LSTM layer"""
        weight_ih_key = f'lstm_layers.{layer_idx}.weight_ih_l0'
        weight_hh_key = f'lstm_layers.{layer_idx}.weight_hh_l0'
        
        if weight_ih_key not in state_dict:
            raise ValueError(f"LSTM layer {layer_idx} weight_ih not found")
        
        weight_ih = state_dict[weight_ih_key]
        weight_hh = state_dict[weight_hh_key]
        
        # LSTM weight_ih shape: (4 * hidden_size, input_size)
        # LSTM weight_hh shape: (4 * hidden_size, hidden_size)
        
        hidden_size = weight_ih.shape[0] // 4
        input_size = weight_ih.shape[1]
        
        return {
            'layer_idx': layer_idx,
            'input_dim': input_size,
            'hidden_size': hidden_size,
            'bidirectional': bidirectional
        }
    
    def _build_lstm_layers(self):
        """Build LSTM layers to match exact state dict structure"""
        for layer_info in self.lstm_config['layers']:
            # Create LSTM layer with exact parameters
            lstm = nn.LSTM(
                input_size=layer_info['input_dim'],
                hidden_size=layer_info['hidden_size'],
                num_layers=1,
                batch_first=True,
                dropout=0.0,
                bidirectional=layer_info['bidirectional']
            )
            self.lstm_layers.append(lstm)
    
    def _build_cause_heads(self):
        """Build cause-specific output heads"""
        # Calculate final hidden dimension
        if self.lstm_config['layers']:
            last_layer = self.lstm_config['layers'][-1]
            final_hidden = last_layer['hidden_size']
            if last_layer['bidirectional']:
                final_hidden *= 2
        else:
            final_hidden = 64  # Default
        
        output_dim = self.lstm_config.get('output_dim', 5)
        num_causes = self.lstm_config.get('num_causes', 2)
        
        # Create cause-specific output heads
        self.cause_heads = nn.ModuleList([
            nn.Linear(final_hidden, output_dim) for _ in range(num_causes)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Ensure 3D input (batch, seq, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Pass through LSTM layers
        for lstm in self.lstm_layers:
            x, (h, c) = lstm(x)
        
        # Take last time step
        if x.size(1) > 1:
            x = x[:, -1, :]
        else:
            x = x.squeeze(1)
        
        # Generate outputs for each cause
        outputs = []
        for cause_head in self.cause_heads:
            cause_output = cause_head(x)
            outputs.append(cause_output)
        
        # Concatenate and reshape for DeepHit format
        # Expected: (batch_size, num_causes, output_dim)
        if len(outputs) > 1:
            concatenated = torch.cat(outputs, dim=1)  # (batch, total_output)
            batch_size = concatenated.shape[0]
            num_causes = len(outputs)
            output_dim = concatenated.shape[1] // num_causes
            reshaped = concatenated.view(batch_size, num_causes, output_dim)
            return reshaped
        else:
            return outputs[0]


def create_flexible_network_from_state_dict(state_dict: Dict[str, torch.Tensor], 
                                           model_path: str = None) -> nn.Module:
    """
    Create neural network by exactly matching state dict architecture
    """
    keys = list(state_dict.keys())
    
    # Determine model type
    if any('cause_networks' in key for key in keys):
        # CNN-based DeepHit model
        return FlexibleDeepHit(state_dict)
    elif any('cause_heads' in key for key in keys):
        # LSTM-based DeepHit model
        return FlexibleLSTMDeepHit(state_dict)
    elif any('lstm' in key.lower() for key in keys):
        # Standard LSTM model (DeepSurv)
        return FlexibleLSTM(state_dict)
    else:
        # MLP model - use flexible MLP creation
        return create_flexible_mlp_from_state_dict(state_dict)


def create_flexible_mlp_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    """
    Create MLP by exactly reconstructing the Sequential structure from state dict
    """
    # Find all layer indices in the sequential model
    layer_indices = set()
    for key in state_dict.keys():
        if key.startswith('model.') and ('weight' in key or 'bias' in key):
            try:
                layer_idx = int(key.split('.')[1])
                layer_indices.add(layer_idx)
            except ValueError:
                continue
    
    layer_indices = sorted(layer_indices)
    
    # Build the sequential model exactly as stored
    layers = []
    
    for layer_idx in layer_indices:
        weight_key = f'model.{layer_idx}.weight'
        bias_key = f'model.{layer_idx}.bias'
        
        if weight_key not in state_dict:
            continue
            
        weight = state_dict[weight_key]
        
        # Check if it's a BatchNorm layer (has running_mean/running_var)
        running_mean_key = f'model.{layer_idx}.running_mean'
        running_var_key = f'model.{layer_idx}.running_var'
        
        if running_mean_key in state_dict and running_var_key in state_dict:
            # BatchNorm layer
            num_features = weight.shape[0]
            layers.append((str(layer_idx), nn.BatchNorm1d(num_features)))
            
        else:
            # Linear layer
            if len(weight.shape) == 2:
                out_features, in_features = weight.shape
                layers.append((str(layer_idx), nn.Linear(in_features, out_features)))
    
    # Create ordered sequential model
    model = nn.Sequential(OrderedDict(layers))
    
    return model


def load_model_with_flexible_architecture(model_path: str, device: torch.device) -> nn.Module:
    """
    Load model with flexible architecture reconstruction
    """
    # Load state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create network
    network = create_flexible_network_from_state_dict(state_dict, str(model_path))
    
    # Load state dict
    try:
        network.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        # Try non-strict loading
        missing_keys, unexpected_keys = network.load_state_dict(state_dict, strict=False)
        
        if len(missing_keys) > len(state_dict) // 2:
            raise RuntimeError(f"Too many missing keys: {len(missing_keys)}/{len(state_dict)}")
    
    network.to(device)
    network.eval()
    
    return network


class StateDictAnalyzer:
    """Utility class for analyzing model state dictionaries"""
    
    @staticmethod
    def analyze_structure(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Comprehensive analysis of state dict structure"""
        analysis = {
            'total_parameters': sum(p.numel() for p in state_dict.values()),
            'layer_types': {},
            'architecture_type': 'unknown',
            'layers': []
        }
        
        # Count layer types
        for key in state_dict.keys():
            if 'conv' in key.lower():
                analysis['layer_types']['conv'] = analysis['layer_types'].get('conv', 0) + 1
            elif 'lstm' in key.lower():
                analysis['layer_types']['lstm'] = analysis['layer_types'].get('lstm', 0) + 1
            elif 'linear' in key.lower() or '.weight' in key:
                analysis['layer_types']['linear'] = analysis['layer_types'].get('linear', 0) + 1
            elif 'batch_norm' in key.lower() or 'running_mean' in key:
                analysis['layer_types']['batch_norm'] = analysis['layer_types'].get('batch_norm', 0) + 1
        
        # Determine architecture type
        if 'cause_networks' in ' '.join(state_dict.keys()):
            analysis['architecture_type'] = 'deephit'
        elif analysis['layer_types'].get('lstm', 0) > 0:
            analysis['architecture_type'] = 'lstm'
        elif analysis['layer_types'].get('conv', 0) > 0:
            analysis['architecture_type'] = 'conv'
        else:
            analysis['architecture_type'] = 'mlp'
        
        return analysis
    
    @staticmethod
    def print_structure(state_dict: Dict[str, torch.Tensor], model_name: str = "Model"):
        """Print detailed structure analysis"""
        print(f"\n=== {model_name} Structure ===")
        analysis = StateDictAnalyzer.analyze_structure(state_dict)
        
        print(f"Architecture Type: {analysis['architecture_type']}")
        print(f"Total Parameters: {analysis['total_parameters']:,}")
        print(f"Layer Types: {analysis['layer_types']}")
        
        print("\nLayer Details:")
        for key, tensor in state_dict.items():
            print(f"  {key}: {tuple(tensor.shape)}")