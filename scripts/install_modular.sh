#!/bin/bash
# Install Cogniverse modules in development mode

set -e  # Exit on error

echo "Installing Cogniverse modular architecture..."
echo "========================================="

# Function to install module
install_module() {
    local path=$1
    local name=$2
    echo ""
    echo "Installing $name..."
    echo "-------------------"
    if [ -f "$path/setup.py" ]; then
        pip install -e "$path" --no-deps
        echo "✅ $name installed"
    else
        echo "❌ No setup.py found in $path"
        exit 1
    fi
}

# Install in dependency order
install_module "src/common" "cogniverse-core"
install_module "src/backends/vespa" "cogniverse-vespa-backend"
install_module "src/app" "cogniverse-app"
install_module "src/evaluation" "cogniverse-evaluation"

echo ""
echo "========================================="
echo "Installing dependencies..."
pip install -e src/common
pip install -e src/backends/vespa
pip install -e src/app
pip install -e src/evaluation

echo ""
echo "========================================="
echo "Modular installation complete!"
echo ""
echo "Verifying installation..."
echo "-------------------------"
python -c "
import sys
import importlib

# Test module imports
test_modules = [
    ('Core', 'src.common.core'),
    ('Core Interfaces', 'src.common.core.interfaces'),
    ('Core Registry', 'src.common.core.backend_registry'),
    ('App Ingestion', 'src.app.ingestion'),
    ('App Search', 'src.app.search'),
    ('App Instrumentation', 'src.app.instrumentation'),
    ('Vespa Backend', 'src.backends.vespa'),
    ('Evaluation', 'src.evaluation.evaluators')
]

all_pass = True
for name, module in test_modules:
    try:
        importlib.import_module(module)
        print(f'✅ {name:20} ({module})')
    except ImportError as e:
        print(f'❌ {name:20} ({module}): {e}')
        all_pass = False

print()
if all_pass:
    print('🎉 All modules imported successfully!')
else:
    print('⚠️  Some modules failed to import')
    sys.exit(1)
"

echo ""
echo "To use the modular installation:"
echo "  - Import from src.common.core for interfaces"
echo "  - Import from src.app for application logic"
echo "  - Import from src.backends.vespa for Vespa backend"
echo "  - Import from src.evaluation for evaluation tools"