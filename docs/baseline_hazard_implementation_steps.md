# Baseline Hazard Implementation Steps

## Step 1: Create the ZenML Pipeline Structure

### Pipeline Components:
1. **Data Processing Steps** (reuse from ensemble_deploy_pipeline.py):
   - ingest_data
   - clean_data
   - merge_data
   - split_data
   - impute_data
   - preprocess_data

2. **New Step**: compute_all_baseline_hazards
   - Input: preprocessed training data
   - Process: compute baseline hazards for all DeepSurv models
   - Output: summary of results

## Step 2: Model Processing Logic

### For Each Model (1-24):

1. **Load Model Configuration**
   ```python
   config_path = f"results/final_deploy/model_config/model{model_no}_details_*.json"
   with open(config_path, 'r') as f:
       model_config = json.load(f)
   ```

2. **Extract Key Parameters**
   - model_type (should be "deepsurv")
   - network_type ("ann" or "lstm")
   - input_dim (11)
   - output_dim (1)
   - dropout
   - For ANN: hidden_dims
   - For LSTM: lstm_hidden_dims, lstm_num_layers, lstm_bidirectional, sequence_length

3. **Load Model Weights**
   ```python
   model_path = f"results/final_deploy/models/Ensemble_model{model_no}_*.pt"
   ```

4. **Create Network Architecture**
   - For ANN models:
     ```python
     net = create_network(
         model_type='deepsurv',
         network_type='ann',
         input_dim=11,
         hidden_dims=model_config.get('hidden_dims', [256, 128, 64]),
         output_dim=1,
         dropout=model_config['dropout']
     )
     ```
   
   - For LSTM models:
     ```python
     net = create_network(
         model_type='deepsurv',
         network_type='lstm',
         input_dim=11,
         lstm_hidden_dims=model_config['lstm_hidden_dims'],
         lstm_num_layers=model_config['lstm_num_layers'],
         bidirectional=model_config['lstm_bidirectional'],
         sequence_length=model_config['sequence_length'],  # Model-specific!
         output_dim=1,
         dropout=model_config['dropout']
     )
     ```

5. **Prepare Data**
   - Extract event number from model name (Event_1 or Event_2)
   - For ANN: Use standard features
   - For LSTM: Create sequences with model-specific sequence_length
     ```python
     if network_type == 'lstm':
         X_seq, y_seq, seq_keys = create_sequences_from_dataframe(
             train_df,
             feature_cols,
             sequence_length=model_config['sequence_length']  # Use model-specific length
         )
     ```

6. **Compute Baseline Hazards**
   ```python
   model = CoxPH(net, optimizer=optimizer)
   model.net.load_state_dict(torch.load(model_path))
   model.compute_baseline_hazards(
       input=X_train_tensor,
       target=(durations, events),
       batch_size=256
   )
   ```

7. **Save Baseline Hazards**
   ```python
   baseline_hazards = {
       'baseline_hazards_': model.baseline_hazards_,
       'baseline_cumulative_hazards_': model.baseline_cumulative_hazards_,
       'model_config': model_config  # Include config for reference
   }
   
   output_path = f"results/final_deploy/models/baseline_hazards_model{model_no}_{timestamp}.pkl"
   with open(output_path, 'wb') as f:
       pickle.dump(baseline_hazards, f)
   ```

## Step 3: Error Handling

```python
for model_no in range(1, 25):
    try:
        # Process model
        process_single_model(model_no, train_df_preprocessed)
        successful_models.append(model_no)
    except Exception as e:
        failed_models.append({
            'model_no': model_no,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        print(f"Failed to process model {model_no}: {e}")
        continue
```

## Step 4: Summary Report

Generate a summary including:
- Total models processed: 24
- Successful: X
- Failed: Y
- Details of any failures
- Paths to saved baseline hazards

## Key Differences from Original compute_baseline_hazards.py

1. **Model Source**: Use models from `/results/final_deploy/models/` instead of `/results/model_details/`
2. **Configuration**: Read from model-specific JSON files in `/results/final_deploy/model_config/`
3. **Sequence Length**: Use model-specific sequence lengths for LSTM models
4. **Pipeline Integration**: Implement as a ZenML pipeline step with caching
5. **Output Location**: Save baseline hazards alongside model weights in the same directory