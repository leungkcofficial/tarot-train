# Model Grouping Summary

## Overview
- **Total Models**: 36
- **DeepSurv Models**: 24 (grouped into 12 groups)
- **DeepHit Models**: 12 (no grouping needed - each provides CIF for both events)

## DeepSurv Model Groups

DeepSurv models are grouped based on:
- Same Algorithm (DeepSurv)
- Same Structure (ANN/LSTM)
- Same Balancing Method
- Same Optimization Target

Each group contains 2 models: one for Event 1 and one for Event 2.

### Group 1: ANN + None + Concordance Index
- **Event 1**: Model 1 (Ensemble_model1_DeepSurv_ANN_Event_1)
- **Event 2**: Model 3 (Ensemble_model3_DeepSurv_ANN_Event_2)

### Group 2: ANN + None + Log-likelihood
- **Event 1**: Model 2 (Ensemble_model2_DeepSurv_ANN_Event_1)
- **Event 2**: Model 4 (Ensemble_model4_DeepSurv_ANN_Event_2)

### Group 3: ANN + NearMiss version 1 + Concordance Index
- **Event 1**: Model 5 (Ensemble_model5_DeepSurv_ANN_Event_1)
- **Event 2**: Model 7 (Ensemble_model7_DeepSurv_ANN_Event_2)

### Group 4: ANN + NearMiss version 1 + Log-likelihood
- **Event 1**: Model 6 (Ensemble_model6_DeepSurv_ANN_Event_1)
- **Event 2**: Model 8 (Ensemble_model8_DeepSurv_ANN_Event_2)

### Group 5: ANN + KNN + Concordance Index
- **Event 1**: Model 9 (Ensemble_model9_DeepSurv_ANN_Event_1)
- **Event 2**: Model 11 (Ensemble_model11_DeepSurv_ANN_Event_2)

### Group 6: ANN + KNN + Log-likelihood
- **Event 1**: Model 10 (Ensemble_model10_DeepSurv_ANN_Event_1)
- **Event 2**: Model 12 (Ensemble_model12_DeepSurv_ANN_Event_2)

### Group 7: LSTM + None + Concordance Index
- **Event 1**: Model 13 (Ensemble_model13_DeepSurv_LSTM_Event_1)
- **Event 2**: Model 15 (Ensemble_model15_DeepSurv_LSTM_Event_2)

### Group 8: LSTM + None + Log-likelihood
- **Event 1**: Model 14 (Ensemble_model14_DeepSurv_LSTM_Event_1)
- **Event 2**: Model 16 (Ensemble_model16_DeepSurv_LSTM_Event_2)

### Group 9: LSTM + NearMiss version 3 + Concordance Index
- **Event 1**: Model 17 (Ensemble_model17_DeepSurv_LSTM_Event_1)
- **Event 2**: Model 19 (Ensemble_model19_DeepSurv_LSTM_Event_2)

### Group 10: LSTM + NearMiss version 3 + Log-likelihood
- **Event 1**: Model 18 (Ensemble_model18_DeepSurv_LSTM_Event_1)
- **Event 2**: Model 20 (Ensemble_model20_DeepSurv_LSTM_Event_2)

### Group 11: LSTM + KNN + Concordance Index
- **Event 1**: Model 21 (Ensemble_model21_DeepSurv_LSTM_Event_1)
- **Event 2**: Model 23 (Ensemble_model23_DeepSurv_LSTM_Event_2)

### Group 12: LSTM + KNN + Log-likelihood
- **Event 1**: Model 22 (Ensemble_model22_DeepSurv_LSTM_Event_1)
- **Event 2**: Model 24 (Ensemble_model24_DeepSurv_LSTM_Event_2)

## DeepHit Models (No Grouping Required)

DeepHit models predict both events simultaneously, so no grouping is needed:

### ANN Models:
- Model 25: Ensemble_model25_DeepHit_ANN_Both
- Model 26: Ensemble_model26_DeepHit_ANN_Both
- Model 27: Ensemble_model27_DeepHit_ANN_Both
- Model 28: Ensemble_model28_DeepHit_ANN_Both
- Model 29: Ensemble_model29_DeepHit_ANN_Both
- Model 30: Ensemble_model30_DeepHit_ANN_Both

### LSTM Models:
- Model 31: Ensemble_model31_DeepHit_LSTM_Both
- Model 32: Ensemble_model32_DeepHit_LSTM_Both
- Model 33: Ensemble_model33_DeepHit_LSTM_Both
- Model 34: Ensemble_model34_DeepHit_LSTM_Both
- Model 35: Ensemble_model35_DeepHit_LSTM_Both
- Model 36: Ensemble_model36_DeepHit_LSTM_Both

## Stacking Strategy

For the ensemble pipeline:
1. **DeepSurv Models**: Stack Event 1 and Event 2 predictions from each group
2. **DeepHit Models**: Use predictions directly (already contain both events)
3. **Final Ensemble**: Combine all stacked DeepSurv predictions and DeepHit predictions