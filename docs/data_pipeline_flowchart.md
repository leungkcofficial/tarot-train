# CKD Risk Prediction Data Pipeline Flowchart

```mermaid
flowchart TD
    %% Data Sources
    A1[Creatinine Data<br/>data/Cr/] --> A[ingest_data]
    A2[Hemoglobin Data<br/>data/Hb/] --> A
    A3[HbA1c Data<br/>data/a1c/] --> A
    A4[Albumin Data<br/>data/alb/] --> A
    A5[Phosphate Data<br/>data/po4/] --> A
    A6[Calcium Data<br/>data/ca/] --> A
    A7[HCO3 Data<br/>data/hco3/] --> A
    A8[UPCR Data<br/>data/upacr/] --> A
    A9[UACR Data<br/>data/uacr/] --> A
    A10[Demographics<br/>data/demo/] --> A
    A11[ICD10 Codes<br/>data/icd10/] --> A
    A12[Death Records<br/>data/death/] --> A
    A13[Operations<br/>data/ot/] --> A

    %% Main Pipeline Steps
    A[ingest_data<br/>📥 Load & Validate] --> B[clean_data<br/>🧹 Clean & Process]
    B --> C[merge_data<br/>🔗 Merge All Sources]
    C --> D[split_data<br/>📊 Train/Temporal/Spatial Split]
    
    %% Parallel Processing Branches
    D --> E[perform_eda<br/>📈 Exploratory Analysis]
    D --> F[kfre_eval<br/>⚖️ KFRE Baseline]
    D --> G[impute_data<br/>🔧 MICE Imputation]
    
    %% Data Preprocessing
    G --> H[preprocess_data<br/>⚙️ Feature Engineering]
    H --> I[feature_selection<br/>🎯 Select Features]
    
    %% Model Training Branch
    H --> J[train_model<br/>🤖 Hyperparameter Optimization]
    J --> K[deploy_model<br/>🚀 Model Deployment]
    K --> L[eval_model<br/>📊 Model Evaluation]
    
    %% Configuration Files
    CONFIG1[src/hyperparameter_config.yml<br/>⚙️ Model Config] --> J
    CONFIG2[src/default_master_df_mapping.yml<br/>📋 Feature Mapping] --> I
    CONFIG2 --> J
    
    %% Outputs and Artifacts
    L --> OUT1[📈 Evaluation Metrics<br/>C-index, Brier Score, etc.]
    L --> OUT2[📊 Visualizations<br/>Calibration, DCA Plots]
    L --> OUT3[💾 Model Artifacts<br/>PyTorch, ONNX Models]
    L --> OUT4[📝 MLflow Registry<br/>Model Versioning]
    
    %% Model Comparison
    F --> M[Model Comparison<br/>🔍 KFRE vs Deep Learning]
    L --> M
    
    %% Styling
    classDef dataSource fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef pipeline fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef config fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef output fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef model fill:#fce4ec,stroke:#880e4f,stroke-width:3px
    
    class A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13 dataSource
    class A,B,C,D,E,F,G,H,I pipeline
    class J,K,L,M model
    class CONFIG1,CONFIG2 config
    class OUT1,OUT2,OUT3,OUT4 output
```

## Pipeline Description

### Data Ingestion Phase
- **Multiple Data Sources**: Ingests 13 different types of clinical data including lab results, demographics, diagnoses, and outcomes
- **Validation**: Automatic schema detection and data validation during ingestion
- **Format Conversion**: Converts CSV files to efficient Parquet format

### Data Processing Phase
- **Cleaning**: Standardizes data formats, handles missing values, calculates derived features
- **Merging**: Combines all data sources into a unified master dataframe
- **Splitting**: Creates temporal and spatial test sets for robust evaluation

### Analysis & Modeling Phase
- **EDA**: Comprehensive exploratory data analysis with statistical summaries
- **KFRE Baseline**: Calculates Kidney Failure Risk Equation as baseline comparison
- **Feature Engineering**: Advanced preprocessing including scaling, encoding, and derived features
- **Feature Selection**: Statistical feature selection with VIF analysis
- **Deep Learning**: Hyperparameter optimization for DeepSurv/DeepHit models using Optuna
- **Evaluation**: Comprehensive model evaluation with survival analysis metrics

### Configuration Management
- **Hyperparameter Config**: YAML-based configuration for model architectures and optimization
- **Feature Mapping**: Centralized feature definitions ensuring consistency across pipeline

### Outputs & Artifacts
- **Model Artifacts**: Trained models in PyTorch and ONNX formats
- **Evaluation Metrics**: C-index, Brier score, calibration metrics with confidence intervals
- **Visualizations**: Calibration plots, decision curve analysis, SHAP explanations
- **MLflow Integration**: Automatic model versioning and experiment tracking