# Makefile for CKD Risk Prediction project

.PHONY: setup install test clean run-all run-single list-types help

# Default Python interpreter
PYTHON = python
PIP = pip

# Default paths
DATA_PATH = data
OUTPUT_DIR = data_lake
METADATA_DIR = metadata

# Help message
help:
	@echo "CKD Risk Prediction - Data Ingestion"
	@echo ""
	@echo "Usage:"
	@echo "  make setup         Create virtual environment and install dependencies"
	@echo "  make install       Install the package in development mode"
	@echo "  make test          Run tests"
	@echo "  make clean         Remove generated files and directories"
	@echo "  make run-all       Run ingestion for all data types"
	@echo "  make run-single    Run ingestion for a single data type (specify with TYPE=...)"
	@echo "  make list-types    List available data types"
	@echo ""
	@echo "Examples:"
	@echo "  make run-all"
	@echo "  make run-single TYPE=creatinine"
	@echo "  make test TYPE=creatinine"

# Create virtual environment and install dependencies
setup:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv venv
	@echo "Installing dependencies..."
	./venv/bin/$(PIP) install -r requirements.txt
	@echo "Setup complete. Activate the virtual environment with:"
	@echo "  source venv/bin/activate  # On Windows: venv\\Scripts\\activate"

# Install the package in development mode
install:
	$(PIP) install -e .

# Run tests
test:
ifdef TYPE
	$(PYTHON) test_ingest.py --data-path $(DATA_PATH) --data-type $(TYPE)
else
	@echo "Error: Please specify a data type with TYPE=..."
	@echo "Example: make test TYPE=creatinine"
	@exit 1
endif

# Clean generated files and directories
clean:
	@echo "Cleaning generated files and directories..."
	rm -rf $(OUTPUT_DIR) $(METADATA_DIR) test_output test_metadata
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "Clean complete."

# Run ingestion for all data types
run-all:
	$(PYTHON) run_pipeline.py ingest --data-path $(DATA_PATH) --output-dir $(OUTPUT_DIR) --metadata-dir $(METADATA_DIR)

# Run ingestion for a single data type
run-single:
ifdef TYPE
	$(PYTHON) run_pipeline.py ingest-single --data-type $(TYPE) --data-path $(DATA_PATH) --output-dir $(OUTPUT_DIR) --metadata-dir $(METADATA_DIR)
else
	@echo "Error: Please specify a data type with TYPE=..."
	@echo "Example: make run-single TYPE=creatinine"
	@exit 1
endif

# List available data types
list-types:
	$(PYTHON) test_ingest.py --list-types