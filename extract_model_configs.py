import json
import os
from pathlib import Path

def extract_model_info(model_num):
    """Extract model information from both details and optimization metrics files."""
    base_path = Path("results/final_deploy/model_config")
    
    # Find the files for this model
    details_file = None
    metrics_file = None
    
    for file in base_path.glob(f"model{model_num}_*.json"):
        if "details" in file.name:
            details_file = file
        elif "optimization_metrics" in file.name:
            metrics_file = file
    
    if not details_file or not metrics_file:
        return None
    
    # Read the files
    with open(details_file, 'r') as f:
        details = json.load(f)
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Extract information
    best_params = metrics.get('best_params', {})
    
    info = {
        'model_num': model_num,
        'model_type': details.get('model_type', 'N/A'),
        'network_type': details.get('network_type', 'ann'),  # Default to ANN if not specified
        'learning_rate': best_params.get('learning_rate', 'N/A'),
        'optimizer': best_params.get('optimizer', 'N/A'),
        'dropout': best_params.get('dropout', details.get('dropout', 'N/A'))
    }
    
    # Handle hidden dimensions based on network type
    if info['network_type'] == 'lstm':
        # For LSTM models
        info['hidden_dims'] = []
        info['lstm_num_layers'] = best_params.get('lstm_num_layers', details.get('lstm_num_layers', 'N/A'))
        info['sequence'] = best_params.get('sequence', details.get('sequence_length', 'N/A'))
        
        # Get LSTM hidden dimensions
        lstm_hidden_dims = []
        for i in range(1, 5):  # Check up to 4 layers
            dim = best_params.get(f'lstm_hidden_dim_layer{i}')
            if dim is not None:
                lstm_hidden_dims.append(dim)
        
        if not lstm_hidden_dims and 'lstm_hidden_dims' in details:
            lstm_hidden_dims = details['lstm_hidden_dims']
        
        info['lstm_hidden_dims'] = lstm_hidden_dims
    else:
        # For ANN models
        hidden_dims = []
        for i in range(1, 5):  # Check up to 4 layers
            dim = best_params.get(f'hidden_units_layer{i}')
            if dim is not None:
                hidden_dims.append(dim)
        
        if not hidden_dims and 'hidden_dims' in details:
            hidden_dims = details['hidden_dims']
        
        info['hidden_dims'] = hidden_dims
        info['lstm_num_layers'] = 'N/A'
        info['lstm_hidden_dims'] = []
        info['sequence'] = 'N/A'
    
    return info

def format_hidden_dims(dims):
    """Format hidden dimensions as a string."""
    if not dims:
        return 'N/A'
    return ' → '.join(map(str, dims))

def create_markdown_table():
    """Create a markdown table with all model configurations."""
    # Collect data for all 36 models
    models_data = []
    
    for i in range(1, 37):
        info = extract_model_info(i)
        if info:
            models_data.append(info)
    
    # Sort by model number
    models_data.sort(key=lambda x: x['model_num'])
    
    # Create markdown content
    markdown = "# Model Configuration Summary\n\n"
    markdown += "This table summarizes the configuration parameters for all 36 models.\n\n"
    markdown += "| Model | Type | Network | Learning Rate | Optimizer | Dropout | Hidden Dimensions (ANN) | LSTM Layers | LSTM Hidden Dimensions | Sequence |\n"
    markdown += "|-------|------|---------|---------------|-----------|---------|------------------------|-------------|----------------------|----------|\n"
    
    for model in models_data:
        model_name = f"Model {model['model_num']}"
        model_type = model['model_type'].upper()
        network = model['network_type'].upper()
        lr = f"{model['learning_rate']:.6f}" if isinstance(model['learning_rate'], float) else model['learning_rate']
        optimizer = model['optimizer']
        dropout = f"{model['dropout']:.4f}" if isinstance(model['dropout'], float) else model['dropout']
        sequence = str(model.get('sequence', 'N/A'))
        
        if network == 'LSTM':
            hidden_dims_ann = 'N/A'
            lstm_layers = str(model['lstm_num_layers'])
            lstm_dims = format_hidden_dims(model['lstm_hidden_dims'])
        else:
            hidden_dims_ann = format_hidden_dims(model['hidden_dims'])
            lstm_layers = 'N/A'
            lstm_dims = 'N/A'
        
        markdown += f"| {model_name} | {model_type} | {network} | {lr} | {optimizer} | {dropout} | {hidden_dims_ann} | {lstm_layers} | {lstm_dims} | {sequence} |\n"
    
    # Add summary statistics
    markdown += "\n## Summary Statistics\n\n"
    
    # Count model types
    deepsurv_count = sum(1 for m in models_data if m['model_type'].lower() == 'deepsurv')
    deephit_count = sum(1 for m in models_data if m['model_type'].lower() == 'deephit')
    
    # Count network types
    ann_count = sum(1 for m in models_data if m['network_type'].lower() == 'ann')
    lstm_count = sum(1 for m in models_data if m['network_type'].lower() == 'lstm')
    
    # Count optimizers
    adam_count = sum(1 for m in models_data if m['optimizer'] == 'Adam')
    adamw_count = sum(1 for m in models_data if m['optimizer'] == 'AdamW')
    
    markdown += f"- **Model Types**: DeepSurv ({deepsurv_count}), DeepHit ({deephit_count})\n"
    markdown += f"- **Network Types**: ANN ({ann_count}), LSTM ({lstm_count})\n"
    markdown += f"- **Optimizers**: Adam ({adam_count}), AdamW ({adamw_count})\n"
    
    return markdown

if __name__ == "__main__":
    # Generate the markdown table
    markdown_content = create_markdown_table()
    
    # Save to file
    output_file = "model_configurations_summary.md"
    with open(output_file, 'w') as f:
        f.write(markdown_content)
    
    print(f"Model configuration summary saved to {output_file}")
    
    # Also print to console
    print("\n" + "="*80)
    print(markdown_content)