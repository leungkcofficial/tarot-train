# LSTM Support for Survival Analysis Pipeline

This document provides a quick start guide for using LSTM networks in the CKD risk prediction pipeline.

## Quick Start

### 1. Switch to LSTM Network

Edit `src/hyperparameter_config.yml`:

```yaml
# Change network type from 'ann' to 'lstm'
network:
  type: "lstm"  # Was: "ann"
  
  # Add LSTM configuration
  lstm:
    hidden_dim: 64
    num_layers: 2
    bidirectional: true
    sequence_length: 5
```

### 2. Run Training

```bash
# Run the training pipeline (same command as before)
python -m pipelines.training_pipeline
```

The pipeline will automatically:
- Generate sequences from your tabular data
- Optimize LSTM-specific hyperparameters
- Train LSTM models instead of MLP models

### 3. Monitor Training

Look for LSTM-specific logs:
```
Network type: lstm
Preparing sequence data for LSTM...
Using default sequence length: 5
Training LSTM with: lr=0.001, sequence_length=5, lstm_hidden=64, lstm_layers=2, bidirectional=True
```

## Configuration Examples

### Use Pre-configured Examples

Copy one of the example configurations:

```bash
# For LSTM DeepSurv (mortality prediction)
cp examples/lstm_deepsurv_config.yml src/hyperparameter_config.yml

# For LSTM DeepHit (competing risks)
cp examples/lstm_deephit_config.yml src/hyperparameter_config.yml
```

### Custom Configuration

Modify these key parameters in `src/hyperparameter_config.yml`:

```yaml
network:
  type: "lstm"
  lstm:
    sequence_length: 5    # How many previous observations to use
    hidden_dim: 64        # LSTM hidden dimension
    num_layers: 2         # Number of LSTM layers
    bidirectional: true   # Use bidirectional LSTM

search_space:
  lstm:
    sequence:
      min: 3              # Minimum sequence length to try
      max: 10             # Maximum sequence length to try
    lstm_hidden_dim:
      min: 32             # Minimum hidden dimension
      max: 128            # Maximum hidden dimension
```

## Data Requirements

Your data must have:
- **Patient ID column** (default: `'key'`): To group observations by patient
- **Date column** (default: `'date'`): To order observations chronologically
- **Multiple observations per patient**: LSTM works best with patients having several time points

Example data structure:
```
key  | date       | creatinine | hemoglobin | duration | endpoint
4094 | 2023-10-15 | 1.2       | 12.5      | 365     | 0
4094 | 2023-11-15 | 1.3       | 12.0      | 365     | 0
4094 | 2023-12-15 | 1.5       | 11.8      | 365     | 1
```

## Key Differences: ANN vs LSTM

| Aspect | ANN (MLP) | LSTM |
|--------|-----------|------|
| **Configuration** | `network.type: "ann"` | `network.type: "lstm"` |
| **Data Processing** | Row-based | Sequence-based |
| **Memory Usage** | Lower | Higher |
| **Training Time** | Faster | Slower |
| **Temporal Modeling** | No | Yes |

## Performance Tips

### For Better Performance
- Start with shorter sequences (3-5)
- Use smaller batch sizes (32-64)
- Reduce number of trials for initial experiments
- Consider unidirectional LSTM for speed

### For Better Accuracy
- Use longer sequences (7-10) if patients have sufficient data
- Use bidirectional LSTM
- Increase LSTM hidden dimensions
- Use multiple LSTM layers

## Troubleshooting

### Common Issues

**"Not enough observations for sequence"**
- Reduce `sequence_length` in configuration
- Check that patients have multiple time points
- Verify date column is properly formatted

**"Out of memory"**
- Reduce `batch_size` in search space
- Reduce `lstm_hidden_dim` range
- Use shorter sequences

**"Training very slow"**
- Reduce `n_trials` in optimization settings
- Use unidirectional LSTM (`bidirectional: false`)
- Reduce sequence length range

### Debug Mode

Run tests to verify setup:
```bash
python tests/test_lstm_implementation.py
```

## Switching Back to ANN

To return to traditional MLP networks:

```yaml
network:
  type: "ann"  # Change back from "lstm"
```

All existing functionality remains unchanged.

## Next Steps

1. **Start Simple**: Begin with default LSTM configuration
2. **Monitor Performance**: Compare LSTM vs ANN results
3. **Optimize**: Adjust sequence length and architecture based on your data
4. **Scale Up**: Increase complexity once basic setup works

For detailed information, see `docs/LSTM_Implementation_Guide.md`.