#!/bin/bash
# Install Cogniverse modules in development mode

echo "Installing Cogniverse modular architecture..."

# Install core module first (no dependencies)
echo "Installing cogniverse-core..."
pip install -e src/common

# Install backend modules (depends on core)
echo "Installing cogniverse-vespa-backend..."
pip install -e src/backends/vespa

# Install app module (depends on core)
echo "Installing cogniverse-app..."
pip install -e src/app

# Install evaluation module (depends on core and app)
echo "Installing cogniverse-evaluation..."
pip install -e src/evaluation

echo "Modular installation complete!"
echo ""
echo "Verify installation:"
python -c "
import sys
modules = [
    'src.common.core',
    'src.app.ingestion',
    'src.app.search',
    'src.backends.vespa',
    'src.evaluation.evaluators'
]
for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError as e:
        print(f'❌ {module}: {e}')
"