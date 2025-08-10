# CKD Risk Prediction - Data Ingestion

This repository contains the data ingestion component of the CKD Risk Prediction MLOps pipeline. It processes raw CSV data from various sources, validates it, and stores it in a data lake for further processing.

## Project Structure

```
.
├── data/                  # Raw data directory
│   ├── Cr/                # Creatinine data
│   ├── a1c/               # Hemoglobin A1c data
│   ├── alb/               # Albumin data
│   ├── death/             # Death records
│   ├── icd10/             # ICD-10 diagnosis codes
│   └── ...                # Other data types
├── data_lake/             # Output directory for processed data (created by pipeline)
├── metadata/              # Metadata about ingested data (created by pipeline)
├── steps/                 # ZenML pipeline steps
│   ├── __init__.py        # Package initialization
│   └── ingest_data.py     # Data ingestion steps
├── __init__.py            # Package initialization
└── run_pipeline.py        # Pipeline runner script
```

## Requirements

- Python 3.11.8
- ZenML 0.82.1 (optional)
- pandas
- numpy
- pyarrow
- MLflow 2.22.0 (optional)

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize ZenML (optional):
   ```bash
   zenml init
   ```

## Usage

### Running the Data Ingestion Pipeline

The data ingestion pipeline can be run in three different modes:

1. **Full Ingestion**: Process all data types
   ```bash
   python run_pipeline.py --data-path data --output-dir data_lake --metadata-dir metadata ingest
   ```

2. **Single Data Type Ingestion**: Process a specific data type
   ```bash
   python run_pipeline.py --data-path data --output-dir data_lake --metadata-dir metadata ingest-single --data-type creatinine
   ```

3. **Detailed Ingestion**: Process a specific data type with detailed step-by-step execution
   ```bash
   python run_pipeline.py --data-path data --output-dir data_lake --metadata-dir metadata ingest-detailed --data-type creatinine
   ```

### Running Without ZenML

If you don't have ZenML installed or prefer to run without it, you can use the `--no-zenml` option:

```bash
python run_pipeline.py --data-path data --output-dir data_lake --metadata-dir metadata --no-zenml ingest
```

This will use the class-based implementation directly without ZenML orchestration.

### Using the Shell Script

For convenience, you can also use the provided shell script:

```bash
# Make the script executable
chmod +x run_ingestion.sh

# Run ingestion for all data types
./run_ingestion.sh

# Run ingestion for a specific data type
./run_ingestion.sh -s creatinine

# Run detailed ingestion for a specific data type
./run_ingestion.sh -s creatinine -d

# Run without ZenML
./run_ingestion.sh -n

# List available data types
./run_ingestion.sh -l
```

### Available Data Types

- `creatinine`: Creatinine lab results (Cr directory)
- `hemoglobin_a1c`: Hemoglobin A1c lab results (a1c directory)
- `albumin`: Albumin lab results (alb directory)
- `calcium`: Calcium lab results (ca directory)
- `hemoglobin`: Hemoglobin lab results (Hb directory)
- `bicarbonate`: Bicarbonate lab results (hco3 directory)
- `phosphate`: Phosphate lab results (po4 directory)
- `urine_protein_creatinine_ratio`: Urine protein-to-creatinine ratio (upacr directory)
- `death`: Death records (death directory)
- `icd10`: ICD-10 diagnosis codes (icd10 directory)
- `operation`: Operation/treatment records (ot directory)

## ZenML Integration

This project uses ZenML for pipeline orchestration. The pipeline is defined in `run_pipeline.py` and uses steps from `steps/ingest_data.py`.

### Pipeline Structure

The data ingestion pipeline consists of the following steps:

1. **Discover Files**: Find all CSV files for a specific data type
2. **Detect Schema**: Automatically detect and map columns to standardized names
3. **Load and Validate Data**: Load data from CSV files and validate
4. **Convert to Parquet**: Convert validated data to Parquet format for efficient storage
5. **Register Metadata**: Store metadata about the ingested data

### ZenML Stack

By default, a local ZenML stack is created if no active stack is found. For production use, you can configure a proper ZenML stack with appropriate components:

```bash
# Example: Set up a stack with S3 artifact store
zenml stack register ckd_risk_stack \
    -a artifact_store=s3_store \
    -o orchestrator=local \
    -c container_registry=default \
    -s secrets_manager=local

# Configure S3 artifact store
zenml artifact-store register s3_store \
    --flavor=s3 \
    --path=s3://ckd-risk-artifacts \
    --key=${MINIO_ACCESS_KEY} \
    --secret=${MINIO_SECRET_KEY} \
    --endpoint=${MINIO_ENDPOINT}

# Set as active stack
zenml stack set ckd_risk_stack
```

## Class-Based Architecture

The data ingestion code is organized using a class-based architecture:

- `FileDiscovery`: Discovers CSV files for a specific data type
- `DataProcessor`: Processes data from CSV files (schema detection, validation, Parquet conversion)
- `DataIngestor`: Orchestrates the ingestion process for one or more data types

This architecture makes the code more modular and easier to maintain. It also allows running the code without ZenML if needed.

## Longitudinal Data Handling

The data ingestion pipeline is designed to handle the longitudinal nature of the data, with support for up to 10 previous visits for each patient. This is achieved by:

1. Extracting year and quarter information from filenames
2. Preserving the temporal order of visits
3. Maintaining patient identifiers across different data types

## Next Steps

After running the data ingestion pipeline, the processed data will be available in the `data_lake` directory in Parquet format. This data can then be used by the data engineering pipeline for further processing, including:

1. Feature engineering
2. Sequence creation for longitudinal analysis
3. Competing risk analysis between death and RRT events
4. Data splitting for model training and validation

## Troubleshooting

### ZenML Issues

If you encounter issues with ZenML, you can:

1. Try running without ZenML using the `--no-zenml` option
2. Check your ZenML installation with `zenml status`
3. Reinitialize ZenML with `zenml init`
4. Check the ZenML documentation for your specific version

### Data Issues

If you encounter issues with the data:

1. Check that the data directories exist and contain CSV files
2. Verify the file naming conventions match what the code expects
3. Look at the logs for specific validation errors
4. Try running with a single data type to isolate the issue