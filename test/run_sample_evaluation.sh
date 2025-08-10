#!/bin/bash
# Run the sample-based evaluation for 60,000 representative combinations

echo "Starting sample-based ensemble evaluation..."
echo "This will evaluate 60,000 representative combinations"
echo "Estimated time: ~20 hours"
echo ""

cd /mnt/dump/yard/projects/tarot2
python fill_metrics_sample_based.py