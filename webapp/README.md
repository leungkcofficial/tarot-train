# TAROT CKD Risk Prediction Web Application

A comprehensive web application for predicting chronic kidney disease (CKD) progression risk using ensemble deep learning models. This application provides risk predictions for dialysis initiation and all-cause mortality over 1-5 year horizons.

## Features

- **Multi-step Form Wizard**: Intuitive data collection with real-time validation
- **Advanced Input Validation**: eGFR-based screening, age restrictions, and outlier detection
- **Unit Conversion**: Automatic conversion between common laboratory units
- **Risk Visualization**: Interactive plots with 95% confidence intervals and clinical benchmarks
- **SHAP Analysis**: Feature importance explanations for clinical interpretation
- **API Endpoints**: RESTful API for healthcare system integration
- **Privacy-First**: Zero data logging with session-based temporary storage

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional)
- 8GB+ RAM (for model loading)

### Development Setup

1. **Clone and setup backend:**
```bash
cd webapp/backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Download models:**
```bash
python scripts/download_models.py
```

3. **Start backend server:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. **Setup frontend:**
```bash
cd ../frontend
npm install
npm start
```

5. **Access application:**
- Web App: http://localhost:3000
- API Docs: http://localhost:8000/docs

### Docker Deployment

```bash
docker-compose up --build
```

## Architecture

```
webapp/
├── backend/           # FastAPI backend
│   ├── app/
│   │   ├── api/      # API endpoints
│   │   ├── core/     # Configuration and validation
│   │   ├── models/   # ML model management
│   │   └── utils/    # Utility functions
│   └── scripts/      # Model download and setup
├── frontend/         # React frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/      # Page components
│   │   ├── utils/      # Helper functions
│   │   └── styles/     # CSS and styling
├── docs/             # Documentation
└── docker/           # Docker configuration
```

## Clinical Usage

### Target Users
- Healthcare professionals (nephrologists, internists, family physicians)
- CKD patients and their families

### Input Requirements
- **Required**: Age, gender, creatinine, hemoglobin, phosphate, bicarbonate, UACR/UPCR
- **Optional**: Medical history (20 comorbidities for Charlson Comorbidity Index)
- **Validation**: Age ≥18, eGFR 10-60 mL/min/1.73m²

### Clinical Benchmarks
- **5-year kidney failure risk 3-5%**: Nephrology referral threshold
- **2-year kidney failure risk >10%**: Multidisciplinary care timing
- **2-year kidney failure risk >40%**: KRT preparation threshold

## API Documentation

### POST /api/predict
Predict CKD progression risk from patient data.

**Request:**
```json
{
  "demographics": {
    "age": 65,
    "gender": "male",
    "date_of_birth": "1958-01-15"
  },
  "laboratory_values": [
    {
      "parameter": "creatinine",
      "value": 180,
      "unit": "umol/L",
      "date": "2024-01-15"
    }
  ],
  "medical_history": [
    {
      "condition": "hypertension",
      "diagnosed": true,
      "date": "2020-01-01"
    }
  ]
}
```

**Response:**
```json
{
  "predictions": {
    "dialysis_risk": [0.05, 0.12, 0.23, 0.35, 0.48],
    "mortality_risk": [0.08, 0.15, 0.24, 0.35, 0.47],
    "confidence_intervals": {
      "dialysis_lower": [0.03, 0.08, 0.15, 0.25, 0.35],
      "dialysis_upper": [0.07, 0.16, 0.31, 0.45, 0.61]
    }
  },
  "shap_values": {
    "dialysis": {"creatinine": 0.15, "age": 0.08},
    "mortality": {"age": 0.22, "creatinine": 0.12}
  },
  "model_info": {
    "ensemble_size": 36,
    "inference_time_ms": 145
  }
}
```

## Model Performance

The ensemble model demonstrates superior performance compared to traditional KFRE models:

- **C-index**: 0.85 (temporal), 0.77 (spatial) for 2-year predictions
- **Brier Score**: 0.019 (temporal), 0.054 (spatial) for 2-year predictions
- **Statistical Significance**: p < 0.001 vs. KFRE models

See `/performance` endpoint for detailed metrics.

## Security & Privacy

- **Zero Data Persistence**: No patient data is stored or logged
- **Session-based Processing**: Temporary storage cleared after each session
- **Input Sanitization**: Comprehensive validation and sanitization
- **HIPAA Considerations**: Designed for healthcare compliance

## Support

For technical issues or deployment questions:
- Documentation: `/docs/`
- API Reference: `/api/docs`
- Model Performance: `/performance`
- Clinical Guidelines: `/disclaimer`

## License

Copyright © 2024 TAROT Study. All rights reserved.