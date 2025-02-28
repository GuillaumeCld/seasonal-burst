#!/bin/bash

# Ensure we're in the correct conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" || "${CONDA_DEFAULT_ENV}" != "seasonal-burst" ]]; then
    echo "Please activate the seasonal-burst conda environment first:"
    echo "conda activate seasonal-burst"
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p experiments/results

# Run all experiments
echo "Running all ICDM paper experiments..."
python experiments/run_all.py --experiments all

# Check if experiments completed successfully
if [ $? -eq 0 ]; then
    echo "All experiments completed successfully!"
    echo "Results are available in experiments/results/"
else
    echo "Some experiments failed. Please check the logs for details."
    exit 1
fi 