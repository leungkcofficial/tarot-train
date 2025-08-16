# 🏥 TAROT: AI-Driven CKD Risk Prediction Platform

<div align="center">

[![Status](https://img.shields.io/badge/Status-MVP%20Ready-brightgreen)](https://github.com/leungkcofficial/tarot-train)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-blue)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)

**The AI-driven Renal Outcome Tracking** system for Chronic Kidney Disease progression prediction

</div>

---

## 🎯 Overview

TAROT is a **production-ready MVP** that provides nephrologists and healthcare professionals with AI-powered risk predictions for CKD progression. The system combines **36 ensemble models** (DeepSurv + DeepHit) to predict dialysis and mortality risks over 1-5 year horizons with clinical-grade accuracy.

### ✨ Key Features

- 🧠 **36-Model Ensemble**: Advanced deep learning combining DeepSurv, DeepHit, and LSTM architectures
- 🌐 **Full-Stack Web Application**: React TypeScript frontend with FastAPI backend
- ⚡ **Real-Time Predictions**: <100ms inference time for clinical workflow integration
- 📊 **Interactive Visualizations**: Risk charts with confidence intervals and SHAP explanations
- 🏥 **Clinical Integration**: KDIGO guidelines, eGFR calculation, and care recommendations
- 🔒 **Privacy-First**: No data persistence, temporary session-only processing
- 🚀 **Production-Ready**: Docker deployment, comprehensive logging, health monitoring

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │   ML Models     │
│  (React TS)     │◄──►│   (FastAPI)      │◄──►│  36 Ensemble    │
│  - Assessment   │    │  - Validation    │    │  - DeepSurv     │
│  - Visualize    │    │  - Inference     │    │  - DeepHit      │
│  - Interpret    │    │  - SHAP          │    │  - LSTM         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Model Architecture

- **24 DeepSurv Models**: Single-event survival analysis (ANN + LSTM variants)
- **12 DeepHit Models**: Competing risks analysis (ANN + LSTM variants)
- **Flexible Loading**: Dynamic architecture reconstruction from state dictionaries
- **Ensemble Averaging**: Weighted predictions across all 36 models

---

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/leungkcofficial/tarot-train.git
cd tarot-train

# Start with Docker Compose
cd webapp
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
```

### Option 2: Local Development

```bash
# 1. Backend Setup
cd webapp/backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Frontend Setup (in another terminal)
cd webapp/frontend
npm install
npm start

# Access at http://localhost:3000
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` in the webapp directory:

```env
# API Configuration
REACT_APP_API_URL=http://localhost:8000/api/v1

# Model Configuration
MODEL_PATH=./foundation_models
METADATA_DIR=./metadata
RANDOM_SEED=42
BATCH_SIZE=64
```

### Model Files

The system requires pre-trained model files in `foundation_models/`:
- `Ensemble_model{1-36}_*.pt`: Model weights
- `ckd_preprocessor.pkl`: Data preprocessing pipeline
- Model configurations in `results/final_deploy/model_config/`

---

## 📱 User Interface

### Clinical Assessment Workflow

1. **📋 Demographics**: Patient age and gender
2. **🔬 Laboratory Values**: Creatinine, hemoglobin, phosphate, bicarbonate, albumin, UACR
3. **📋 Medical History**: Comorbidities and conditions
4. **📊 Risk Analysis**: AI-generated predictions with clinical interpretations

### Interactive Visualizations

- **Risk Timeline Charts**: Plotly-powered interactive graphs
- **Confidence Intervals**: Uncertainty quantification (95% CI)
- **SHAP Explanations**: Feature importance for model interpretability
- **Clinical Recommendations**: Actionable guidance based on KDIGO thresholds

---

## 🏥 Clinical Features

### Risk Stratification

| Risk Level | 2-Year Dialysis Risk | Clinical Action |
|------------|---------------------|-----------------|
| Low | <5% | Standard CKD care |
| Moderate | 5-10% | Nephrology referral consideration |
| High | 10-40% | Multidisciplinary care |
| Very High | >40% | Urgent KRT preparation |

### KDIGO Integration

- **eGFR Calculation**: CKD-EPI 2021 equation
- **CKD Staging**: Automatic classification (Stages 3-5)
- **Care Thresholds**: Evidence-based referral recommendations
- **Clinical Benchmarks**: Nephrology, multidisciplinary, KRT preparation

---

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Async web framework
- **PyTorch**: Deep learning inference
- **PyCox**: Survival analysis utilities
- **Pydantic**: Data validation
- **Structured Logging**: JSON-formatted logs

### Frontend
- **React 18**: UI framework
- **TypeScript**: Type safety
- **Material-UI**: Professional medical UI
- **Plotly**: Interactive visualizations
- **Axios**: API communication

### MLOps & Deployment
- **Docker**: Containerization
- **Git LFS**: Model artifact management
- **ZenML**: ML pipeline orchestration
- **MLflow**: Experiment tracking
- **Health Monitoring**: API status endpoints

---

## 📊 Model Performance

### Target Metrics
- **C-index**: ≥0.80 (mortality), ≥0.95 (RRT)
- **Integrated Brier Score**: <0.10
- **Inference Time**: <100ms (P95)

### Current Performance
- **Ensemble Size**: 36 models
- **Average Inference**: ~40-90ms
- **Model Loading**: ~3 seconds (startup)
- **Memory Usage**: ~1.5GB (all models loaded)

---

## 🔬 Research & Development

### MLOps Pipeline

The complete training pipeline includes:

```bash
# Data ingestion and preprocessing
python run_pipeline.py ingest --data-path data --output-dir data_lake

# Model training with hyperparameter optimization
python run_pipeline.py train --config config.yml

# Ensemble evaluation and deployment
python pipelines/final_deploy_v2_fixed.py
```

### Development Commands

```bash
# Setup development environment
make setup
make install

# Run tests
make test TYPE=creatinine
pytest tests/

# Data pipeline
make run-all  # All data types
make run-single TYPE=creatinine  # Single type

# Model deployment
python pipelines/final_deploy_v2_fixed.py

# Cleanup
make clean
```

---

## 📁 Project Structure

```
tarot-train/
├── 📁 webapp/                    # Web Application
│   ├── 📁 backend/               # FastAPI Backend
│   │   ├── 📁 app/
│   │   │   ├── 📁 api/v1/        # API Endpoints
│   │   │   ├── 📁 models/        # Model Management
│   │   │   ├── 📁 core/          # Configuration
│   │   │   └── 📁 schemas/       # Data Models
│   │   └── 📄 requirements.txt
│   ├── 📁 frontend/              # React Frontend
│   │   ├── 📁 src/
│   │   │   ├── 📁 components/    # UI Components
│   │   │   ├── 📁 pages/         # App Pages
│   │   │   ├── 📁 contexts/      # State Management
│   │   │   └── 📁 services/      # API Client
│   │   └── 📄 package.json
│   └── 📄 docker-compose.yml
├── 📁 foundation_models/         # Pre-trained Models
├── 📁 src/                      # Training Pipeline
├── 📁 pipelines/                # ZenML Pipelines
├── 📁 steps/                    # Pipeline Steps
├── 📄 CLAUDE.md                 # Development Documentation
└── 📄 README.md                 # This file
```

---

## 🚦 API Endpoints

### Health & Status
- `GET /health` - System health check
- `GET /health/detailed` - Detailed system status
- `GET /health/models` - Model loading status

### Prediction
- `POST /api/v1/predict` - Generate risk predictions
- `POST /api/v1/predict/validate` - Validate input data

### Information
- `GET /info/performance` - Model performance metrics
- `GET /info/disclaimer` - Clinical disclaimer
- `GET /info/clinical-benchmarks` - KDIGO thresholds

---

## 🔐 Privacy & Security

- **No Data Persistence**: Patient data processed in memory only
- **Session-Based**: Automatic cleanup after 4 hours inactivity
- **Privacy Notices**: Clear user communication
- **Clinical Disclaimers**: Appropriate medical warnings
- **Validation**: Input sanitization and medical range checks

---

## 🎯 MVP Status

### ✅ Production Ready Features
- Complete end-to-end clinical workflow
- Professional medical UI/UX
- Real-time model inference with all 36 models
- Clinical validation and recommendations
- Docker deployment and orchestration
- Comprehensive logging and monitoring

### 🚀 Ready for Next Phase
- Clinical research pilots
- Healthcare institution trials
- EHR system integration
- Regulatory review preparation
- Multi-center deployment

---

## 🤝 Contributing

This is a research project associated with clinical studies. For collaboration inquiries, please contact the research team.

---

## 📄 License

This project contains proprietary AI models and clinical algorithms. Please refer to licensing terms for usage permissions.

---

## 📞 Contact

For technical questions or clinical collaboration:
- **Research Team**: [Contact Information]
- **Technical Issues**: GitHub Issues
- **Clinical Inquiries**: [Clinical Contact]

---

<div align="center">

**TAROT CKD Risk Prediction Platform**  
*Advancing precision medicine through AI-driven clinical decision support*

</div>