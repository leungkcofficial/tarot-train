# LSTM Implementation Guide for Survival Analysis

This guide explains how to use the LSTM (Long Short-Term Memory) neural network implementation for survival analysis in the CKD risk prediction pipeline.

## Overview

The LSTM implementation extends the existing survival analysis pipeline to support temporal sequence modeling. Instead of treating each patient observation as an independent row, LSTM models can learn from sequences of patient data over time.

## Key Features

- **Pure LSTM Architecture**: Direct LSTM-to-output networks without intermediate MLP layers
- **Sequence Generation**: Automatic conversion of tabular data to temporal sequences
- **Hyperparameter Optimization**: LSTM-specific parameters included in search space
- **Backward Compatibility**: Existing MLP functionality remains unchanged
- **Extensible Design**: Framework ready for future network types (e.g., Transformers)

## Network Types

### 1. ANN (Artificial Neural Network) - Default
- **Type**: `ann` (default)
- **Architecture**: Multi-layer Perceptron (MLP)
- **Data Format**: Row-based (each row is one observation)
- **Use Case**: Traditional survival analysis with independent observations

### 2. LSTM (Long Short-Term Memory)
- **Type**: `lstm`
- **Architecture**: Pure LSTM with direct output layer
- **Data Format**: Sequence-based (sequences of observations per patient)
- **Use Case**: Temporal survival analysis with sequential patient data

## Configuration

### Basic LSTM Configuration

```yaml
# src/hyperparameter_config.yml

# Network type selection
network:
  type: "lstm"  # Options: "ann" (default), "lstm"
  
  # LSTM-specific parameters
  lstm:
    hidden_dim: 64
    num_layers: 2
    bidirectional: true
    sequence_length: 5  # Default sequence length

# LSTM hyperparameter search space
search_space:
  lstm:
    sequence:
      type: "int"
      min: 3
      max: 10
    lstm_hidden_dim:
      type: "int"
      min: 32
      max: 128
    lstm_num_layers:
      type: "int"
      min: 1
      max: 3
    bidirectional:
      type: "categorical"
      values: [true, false]
```

### Example Configurations

#### Configuration 1: Pure LSTM for DeepSurv
```yaml
model_type: "deepsurv"
target_endpoint: 2  # Focus on mortality

network:
  type: "lstm"
  lstm:
    hidden_dim: 64
    num_layers: 2
    bidirectional: true
    sequence_length: 5

search_space:
  lstm:
    sequence:
      type: "int"
      min: 3
      max: 8
    lstm_hidden_dim:
      type: "int"
      min: 32
      max: 96
```

#### Configuration 2: Pure LSTM for DeepHit
```yaml
model_type: "deephit"

network:
  type: "lstm"
  lstm:
    hidden_dim: 128
    num_layers: 3
    bidirectional: true
    sequence_length: 7

search_space:
  lstm:
    sequence:
      type: "int"
      min: 5
      max: 10
    lstm_hidden_dim:
      type: "int"
      min: 64
      max: 128
```

## Data Requirements

### Input Data Format
The LSTM implementation requires the following columns in your dataframe:

- **Cluster Column**: Patient identifier (default: `'key'`)
- **Date Column**: Temporal ordering (default: `'date'`)
- **Duration Column**: Survival time (default: `'duration'`)
- **Event Column**: Event indicator (default: `'endpoint'`)
- **Feature Columns**: Clinical features as specified in `default_master_df_mapping.yml`

### Sequence Generation Process

1. **Grouping**: Data is grouped by patient ID
2. **Sorting**: Within each patient, observations are sorted chronologically
3. **Sequence Creation**: 
   - If patient has ≥ sequence_length observations: Use last N observations
   - If patient has < sequence_length observations: Zero-pad at the beginning
4. **Output**: 3D tensor of shape `(n_patients, sequence_length, n_features)`

### Example Data Transformation

**Input (Tabular)**:
```
patient_id | date       | feature_1 | feature_2 | duration | endpoint
4094      | 2023-11-14 | 1.2      | 0.8      | 365     | 0
4094      | 2023-11-28 | 1.3      | 0.9      | 365     | 0  
4094      | 2023-12-08 | 1.4      | 1.0      | 365     | 1
```

**Output (Sequence for sequence_length=3)**:
```
Shape: (1, 3, 2)
[[[1.2, 0.8],   # 2023-11-14
  [1.3, 0.9],   # 2023-11-28  
  [1.4, 1.0]]]  # 2023-12-08
```

## Usage Examples

### Running LSTM Training

1. **Update Configuration**:
   ```bash
   # Edit src/hyperparameter_config.yml
   # Set network.type: "lstm"
   ```

2. **Run Training Pipeline**:
   ```bash
   python -m pipelines.training_pipeline
   ```

3. **Monitor Training**:
   - LSTM-specific hyperparameters will be optimized
   - Sequence generation happens automatically
   - Training logs show LSTM parameters

### Testing LSTM Implementation

```bash
# Run comprehensive tests
python tests/test_lstm_implementation.py
```

## Architecture Details

### LSTM Network Classes

#### LSTMSurvival (for DeepSurv)
```python
class LSTMSurvival(nn.Module):
    """Pure LSTM architecture for survival analysis."""
    
    def __init__(
        self,
        input_dim: int,           # Features per timestep
        sequence_length: int,     # Sequence length
        lstm_hidden_dim: int,     # LSTM hidden dimension
        lstm_num_layers: int,     # Number of LSTM layers
        output_dim: int = 1,      # Output dimension
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
```

#### LSTMDeepHit (for DeepHit)
```python
class LSTMDeepHit(nn.Module):
    """LSTM architecture for competing risks analysis."""
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        output_dim: int,          # Number of time intervals
        num_causes: int = 2,      # Number of competing risks
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
```

### Key Design Decisions

1. **Pure LSTM**: No MLP layers after LSTM for simplicity and efficiency
2. **Bidirectional**: Default to bidirectional LSTM for better temporal understanding
3. **Zero Padding**: Pad short sequences at the beginning (older timestamps)
4. **Chronological Order**: Sequences ordered from oldest to newest
5. **Last Hidden State**: Use final hidden state for prediction

## Performance Considerations

### Memory Usage
- LSTM models use more memory than MLP models
- Memory scales with: `batch_size × sequence_length × hidden_dim`
- Consider reducing batch size for longer sequences

### Training Time
- LSTM training is slower than MLP training
- Time scales with sequence length and number of LSTM layers
- Bidirectional LSTM takes ~2x longer than unidirectional

### Hyperparameter Optimization
- LSTM has more hyperparameters to optimize
- Sequence length optimization requires data regeneration
- Consider reducing `n_trials` for initial experiments

## Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Reduce batch size in search space
   - Reduce sequence length range
   - Use smaller LSTM hidden dimensions

2. **Slow Training**:
   - Reduce number of LSTM layers
   - Use unidirectional LSTM
   - Reduce sequence length

3. **Poor Performance**:
   - Check sequence generation (patients need sufficient observations)
   - Verify temporal ordering of data
   - Consider feature scaling

### Debug Mode

Enable detailed logging by running:
```python
# In your script
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Comparison: ANN vs LSTM

| Aspect | ANN (MLP) | LSTM |
|--------|-----------|------|
| **Data Format** | Row-based | Sequence-based |
| **Temporal Modeling** | No | Yes |
| **Memory Usage** | Lower | Higher |
| **Training Speed** | Faster | Slower |
| **Interpretability** | Higher | Lower |
| **Use Case** | Independent observations | Sequential observations |

## Future Extensions

The current implementation provides a foundation for additional network types:

1. **Transformer Networks**: Can be added following the same pattern
2. **Hybrid Networks**: LSTM + MLP combinations
3. **Attention Mechanisms**: For better interpretability
4. **Multi-scale Temporal Modeling**: Different sequence lengths

## References

- [LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [DeepSurv Paper](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1)
- [DeepHit Paper](https://ojs.aaai.org/index.php/AAAI/article/view/11842)