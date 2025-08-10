# Multi-Model Ensemble Flow Diagram

## Overall Process Flow

```mermaid
flowchart TB
    Start[Start: Load Model Configurations]
    Start --> LoadCSV[Load model_config.csv]
    LoadCSV --> LoadJSON[Load JSON files for each model]
    
    LoadJSON --> SplitModels{Split by Model Type}
    
    SplitModels -->|DeepSurv| DS[24 DeepSurv Models]
    SplitModels -->|DeepHit| DH[12 DeepHit Models]
    
    DS --> DSPred[Generate Predictions<br/>Shape: 1825 x n_samples each]
    DH --> DHPred[Generate Predictions<br/>Shape: 2 x 5 x n_samples each]
    
    DSPred --> ConvertCIF[Convert to CIF]
    ConvertCIF --> ExtractTP[Extract Time Points<br/>365, 730, 1095, 1460, 1825]
    ExtractTP --> DSShape[Shape: 5 x n_samples each]
    
    DSShape --> GroupDS[Group DeepSurv Models<br/>by Algorithm, Structure,<br/>Balancing, Optimization]
    GroupDS --> StackDS[Stack Event 1 & Event 2<br/>12 groups x 2 x 5 x n_samples]
    
    StackDS --> Combine[Combine All Models]
    DHPred --> Combine
    
    Combine --> FinalStack[Final Stack<br/>24 x 2 x 5 x n_samples]
    FinalStack --> Ensemble[Ensemble Averaging]
    Ensemble --> FinalPred[Final Predictions<br/>2 x 5 x n_samples]
    
    FinalPred --> Save[Save Results]
```

## DeepSurv Grouping Logic

```mermaid
flowchart LR
    subgraph Group1 [Group 1: ANN + None + CI]
        M1[Model 1: Event 1]
        M3[Model 3: Event 2]
    end
    
    subgraph Group2 [Group 2: ANN + None + LL]
        M2[Model 2: Event 1]
        M4[Model 4: Event 2]
    end
    
    subgraph Group3 [Group 3: ANN + NearMiss1 + CI]
        M5[Model 5: Event 1]
        M7[Model 7: Event 2]
    end
    
    M1 --> Stack1[Stack to 2x5xn]
    M3 --> Stack1
    
    M2 --> Stack2[Stack to 2x5xn]
    M4 --> Stack2
    
    M5 --> Stack3[Stack to 2x5xn]
    M7 --> Stack3
```

## Prediction Shape Transformations

```mermaid
flowchart TB
    subgraph DeepSurv Transformation
        DS1[DeepSurv Model<br/>1825 x n_samples]
        DS1 --> CIF1[Convert to CIF<br/>1825 x n_samples]
        CIF1 --> Extract1[Extract 5 time points<br/>5 x n_samples]
    end
    
    subgraph DeepHit Output
        DH1[DeepHit Model<br/>2 x 5 x n_samples]
        DH1 --> Ready[Already in correct format]
    end
    
    subgraph Stacking Process
        E1[Event 1: 5 x n_samples]
        E2[Event 2: 5 x n_samples]
        E1 --> Stacked[Stacked: 2 x 5 x n_samples]
        E2 --> Stacked
    end
```

## Ensemble Architecture

```mermaid
flowchart TB
    subgraph Input Layer
        M1_24[24 Model Predictions<br/>Each: 2 x 5 x n_samples]
    end
    
    subgraph Ensemble Methods
        Avg[Simple Averaging]
        Weighted[Weighted Average<br/>Future]
        Vote[Voting<br/>Future]
        Stack[Stacking<br/>Future]
    end
    
    subgraph Output
        Final[Final Ensemble<br/>2 x 5 x n_samples]
    end
    
    M1_24 --> Avg
    M1_24 -.-> Weighted
    M1_24 -.-> Vote
    M1_24 -.-> Stack
    
    Avg --> Final
    Weighted -.-> Final
    Vote -.-> Final
    Stack -.-> Final
```

## Data Flow Summary

1. **Input**: 36 models (24 DeepSurv + 12 DeepHit)
2. **DeepSurv Processing**:
   - Convert survival probabilities to CIF
   - Extract 5 specific time points
   - Group by characteristics
   - Stack Event 1 & 2 predictions
3. **Combination**: Stack all 24 model groups
4. **Ensemble**: Average across all models
5. **Output**: Final predictions for 2 endpoints at 5 time points