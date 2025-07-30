#!/bin/bash
# Activate the uv virtual environment and run commands

# Source the activation script with full path
source /Users/amjain/source/hobby/cogniverse/.venv/bin/activate

# Now python will use the venv python
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Run the debug script
python debug_videoprism_encoder.py