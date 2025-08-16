# Model Configuration Summary

This table summarizes the configuration parameters for all 36 models.

| Model | Type | Network | Learning Rate | Optimizer | Dropout | Hidden Dimensions (ANN) | LSTM Layers | LSTM Hidden Dimensions | Sequence |
|-------|------|---------|---------------|-----------|---------|------------------------|-------------|----------------------|----------|
| Model 1 | DEEPSURV | ANN | 0.000422 | Adam | 0.0302 | 119 → 124 → 14 → 5 | N/A | N/A | N/A |
| Model 2 | DEEPSURV | ANN | 0.000778 | AdamW | 0.1793 | 112 → 96 → 37 → 30 | N/A | N/A | N/A |
| Model 3 | DEEPSURV | ANN | 0.000424 | AdamW | 0.0068 | 63 → 103 → 59 → 20 | N/A | N/A | N/A |
| Model 4 | DEEPSURV | ANN | 0.000423 | AdamW | 0.0055 | 31 → 113 → 49 → 20 | N/A | N/A | N/A |
| Model 5 | DEEPSURV | ANN | 0.000426 | Adam | 0.0397 | 117 → 90 → 41 → 13 | N/A | N/A | N/A |
| Model 6 | DEEPSURV | ANN | 0.000867 | Adam | 0.0275 | 61 → 61 → 57 → 16 | N/A | N/A | N/A |
| Model 7 | DEEPSURV | ANN | 0.000423 | Adam | 0.0797 | 101 → 107 → 64 → 17 | N/A | N/A | N/A |
| Model 8 | DEEPSURV | ANN | 0.000256 | Adam | 0.0365 | 89 → 72 → 34 → 13 | N/A | N/A | N/A |
| Model 9 | DEEPSURV | ANN | 0.000286 | AdamW | 0.0427 | 106 → 127 → 48 → 30 | N/A | N/A | N/A |
| Model 10 | DEEPSURV | ANN | 0.000441 | Adam | 0.0011 | 119 → 99 → 47 → 9 | N/A | N/A | N/A |
| Model 11 | DEEPSURV | ANN | 0.000899 | AdamW | 0.0586 | 115 → 109 → 47 → 11 | N/A | N/A | N/A |
| Model 12 | DEEPSURV | ANN | 0.001459 | Adam | 0.0543 | 86 → 41 → 64 → 15 | N/A | N/A | N/A |
| Model 13 | DEEPSURV | LSTM | 0.000903 | Adam | 0.0047 | N/A | 2 | 64 → 34 | 10 |
| Model 14 | DEEPSURV | LSTM | 0.002292 | AdamW | 0.0242 | N/A | 2 | 66 → 32 | 8 |
| Model 15 | DEEPSURV | LSTM | 0.001712 | Adam | 0.0612 | N/A | 3 | 83 → 87 → 42 | 9 |
| Model 16 | DEEPSURV | LSTM | 0.004036 | Adam | 0.2051 | N/A | 2 | 119 → 80 | 9 |
| Model 17 | DEEPSURV | LSTM | 0.009604 | Adam | 0.0196 | N/A | 2 | 83 → 50 | 9 |
| Model 18 | DEEPSURV | LSTM | 0.000541 | Adam | 0.0939 | N/A | 2 | 84 → 86 | 10 |
| Model 19 | DEEPSURV | LSTM | 0.002770 | Adam | 0.2967 | N/A | 3 | 111 → 56 → 34 | 10 |
| Model 20 | DEEPSURV | LSTM | 0.000178 | AdamW | 0.0094 | N/A | 2 | 64 → 60 | 10 |
| Model 21 | DEEPSURV | LSTM | 0.000730 | Adam | 0.6374 | N/A | 2 | 81 → 80 | 8 |
| Model 22 | DEEPSURV | LSTM | 0.003045 | Adam | 0.1283 | N/A | 2 | 73 → 34 | 10 |
| Model 23 | DEEPSURV | LSTM | 0.001358 | AdamW | 0.1246 | N/A | 2 | 100 → 58 | 9 |
| Model 24 | DEEPSURV | LSTM | 0.000416 | Adam | 0.5755 | N/A | 2 | 74 → 54 | 7 |
| Model 25 | DEEPHIT | ANN | 0.000137 | AdamW | 0.0374 | 124 → 103 → 42 → 23 | N/A | N/A | N/A |
| Model 26 | DEEPHIT | ANN | 0.000576 | Adam | 0.0635 | 100 → 118 → 15 → 30 | N/A | N/A | N/A |
| Model 27 | DEEPHIT | ANN | 0.000364 | AdamW | 0.0094 | 103 → 99 → 60 → 16 | N/A | N/A | N/A |
| Model 28 | DEEPHIT | ANN | 0.000135 | Adam | 0.3711 | 46 → 25 → 14 → 12 | N/A | N/A | N/A |
| Model 29 | DEEPHIT | ANN | 0.000334 | Adam | 0.0526 | 77 → 127 → 11 → 19 | N/A | N/A | N/A |
| Model 30 | DEEPHIT | ANN | 0.000339 | Adam | 0.0538 | 113 → 71 → 58 → 20 | N/A | N/A | N/A |
| Model 31 | DEEPHIT | LSTM | 0.000515 | AdamW | 0.2147 | N/A | 1 | 119 | 9 |
| Model 32 | DEEPHIT | LSTM | 0.001370 | AdamW | 0.4745 | N/A | 3 | 121 → 61 → 28 | 7 |
| Model 33 | DEEPHIT | LSTM | 0.003372 | AdamW | 0.0506 | N/A | 3 | 99 → 54 → 47 | 10 |
| Model 34 | DEEPHIT | LSTM | 0.000663 | Adam | 0.4741 | N/A | 2 | 91 → 83 | 9 |
| Model 35 | DEEPHIT | LSTM | 0.000226 | AdamW | 0.0406 | N/A | 3 | 107 → 45 → 61 | 10 |
| Model 36 | DEEPHIT | LSTM | 0.001664 | Adam | 0.5339 | N/A | 2 | 75 → 85 | 8 |

## Summary Statistics

- **Model Types**: DeepSurv (24), DeepHit (12)
- **Network Types**: ANN (18), LSTM (18)
- **Optimizers**: Adam (22), AdamW (14)
