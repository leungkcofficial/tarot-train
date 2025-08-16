# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is TAROT (The AI-driven Renal Outcome Tracking) - a comprehensive MLOps repository for CKD (Chronic Kidney Disease) risk prediction. The project implements deep learning survival models (DeepSurv, DeepHit) to predict renal replacement therapy (RRT) and all-cause mortality over 1-5 year horizons. The architecture uses ensemble learning combining multiple model types with clinical validation targeting c-index ≥ 0.80 for mortality and ≥ 0.95 for RRT predictions.

## Development Commands

### Environment Setup
```bash
# Create virtual environment and install dependencies
make setup

# Install package in development mode
make install

# Alternative setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Testing
```bash
# Run tests (requires TYPE specification)
make test TYPE=creatinine
pytest tests/  # Run all pytest tests
```

### Data Pipeline
```bash
# Run complete MLOps pipeline
python run_pipeline.py

# Data ingestion only
make run-all  # All data types
make run-single TYPE=creatinine  # Single data type
python run_pipeline.py ingest --data-path data --output-dir data_lake

# List available data types
make list-types
```

### Model Training & Deployment
```bash
# Train models with custom config
python run_pipeline.py --config custom_config.yml train

# Run without ZenML (for testing)
python run_pipeline.py --no-zenml

# Deploy ensemble models
python pipelines/final_deploy_v2_fixed.py
```

### Cleanup
```bash
make clean  # Remove generated files and directories
```

## Architecture Overview

### Core Components
- **Data Ingestion**: Processes longitudinal EHR data from CSV sources (lab results, diagnoses, procedures)
- **Preprocessing Pipeline**: Data cleaning, validation, and feature engineering with configurable mappings
- **Model Training**: Deep learning survival models with hyperparameter optimization using Optuna
- **Ensemble Learning**: Combines DeepSurv (single-event), DeepHit (competing risks), and LSTM models
- **MLOps Integration**: Full pipeline orchestration with ZenML and experiment tracking with MLflow

### Key Directory Structure
- `src/`: Core source code including neural architectures, survival utilities, data processing
- `steps/`: ZenML pipeline steps for modular workflow components
- `pipelines/`: Complete pipeline definitions (training, deployment, ensemble evaluation)
- `data/`: Raw input data organized by biomarker type (Cr/, Hb/, a1c/, etc.)
- `data_lake/`: Processed parquet files from ingestion
- `foundation_models/`: Pre-trained model weights and baseline hazards
- `results/`: Model evaluation outputs, metrics, and visualizations

### Configuration System

#### Environment Variables (.env)
Configure data paths, model settings, and column mappings:
```env
DATA_PATH=./data
OUTPUT_DIR=./data_lake
METADATA_DIR=./metadata
RANDOM_SEED=42
NETWORK_TYPE=mlp
BATCH_SIZE=64
```

#### Feature Configuration (src/default_master_df_mapping.yml)
Defines which features to use, categorical vs continuous distinction, and KFRE mappings. Current features include: gender, creatinine, hemoglobin, phosphate, age_at_obs, bicarbonate, albumin, uacr, cci_score_total, ht, observation_period.

#### Hyperparameter Configuration (src/hyperparameter_config.yml)
Controls model architecture and search space for optimization. Supports both ANN and LSTM networks for DeepSurv/DeepHit models with configurable hidden dimensions, dropout rates, and training parameters.

### Model Pipeline Flow
1. **Data Ingestion**: Raw CSV → validated parquet files in data_lake/
2. **Preprocessing**: Feature engineering, imputation, and sequence generation
3. **Model Training**: Hyperparameter optimization with cross-validation
4. **Ensemble Creation**: Combines multiple model predictions
5. **Evaluation**: Clinical metrics (c-index, Brier score, DCA) and visualization
6. **Deployment**: Model registration and serving setup

### Technology Stack
- Python 3.11.8, PyTorch 2.4.1, PyCox for survival modeling
- ZenML 0.82.1 for MLOps orchestration, MLflow 2.22.0 for experiment tracking
- Great Expectations and Pandera for data validation
- Supports both local development and production stacks with S3/MinIO

### Important Notes
- Models are cached in foundation_models/ directory with Git LFS
- Pipeline supports both spatial (cross-sectional) and temporal (longitudinal) data splits
- Ensemble evaluation can process millions of model combinations with checkpointing
- Target metrics: c-index ≥ 0.80 for mortality, ≥ 0.95 for RRT, Integrated Brier Score < 0.10