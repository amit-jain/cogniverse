#!/bin/bash

# Script to run comprehensive local tests with full coverage
# These tests are skipped in CI/CD but provide thorough coverage locally

echo "========================================="
echo "Running Comprehensive Local Tests"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Ollama is running (for LLM tests)
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
    OLLAMA_AVAILABLE=1
else
    echo -e "${YELLOW}⚠ Ollama is not running - some tests will be skipped${NC}"
    OLLAMA_AVAILABLE=0
fi

# Check for GLiNER models
python -c "from gliner import GLiNER" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ GLiNER is available${NC}"
    GLINER_AVAILABLE=1
else
    echo -e "${YELLOW}⚠ GLiNER not available - some tests will be skipped${NC}"
    GLINER_AVAILABLE=0
fi

echo ""
echo "Running tests..."
echo ""

# Build marker expression based on available services
MARKERS="not benchmark"  # Exclude benchmark by default

if [ $OLLAMA_AVAILABLE -eq 0 ]; then
    MARKERS="$MARKERS and not requires_ollama"
fi

if [ $GLINER_AVAILABLE -eq 0 ]; then
    MARKERS="$MARKERS and not requires_gliner"
fi

# Run all tests including local_only
uv run pytest tests/ \
    -v \
    -m "$MARKERS" \
    --cov=src/app/routing \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=json \
    --tb=short \
    --durations=10

# Capture exit code
TEST_EXIT_CODE=$?

# Generate coverage summary
echo ""
echo "========================================="
echo "Coverage Summary"
echo "========================================="

# Extract coverage percentages for routing modules
python -c "
import json
with open('coverage.json', 'r') as f:
    data = json.load(f)
    
print('Module Coverage:')
print('-' * 50)

routing_files = [
    'src/app/routing/router.py',
    'src/app/routing/strategies.py', 
    'src/app/routing/optimizer.py',
    'src/app/routing/config.py',
    'src/app/routing/base.py'
]

total_stmts = 0
total_miss = 0

for file in routing_files:
    if file in data['files']:
        stats = data['files'][file]['summary']
        coverage = stats['percent_covered']
        name = file.split('/')[-1]
        total_stmts += stats['num_statements']
        total_miss += stats['missing_lines']
        
        # Color code based on coverage
        if coverage >= 80:
            color = '\033[0;32m'  # Green
        elif coverage >= 60:
            color = '\033[1;33m'  # Yellow
        else:
            color = '\033[0;31m'  # Red
        
        print(f'{name:20s}: {color}{coverage:5.1f}%\033[0m ({stats[\"num_statements\"]} stmts, {stats[\"missing_lines\"]} missing)')

if total_stmts > 0:
    overall = (total_stmts - total_miss) / total_stmts * 100
    print('-' * 50)
    
    if overall >= 70:
        color = '\033[0;32m'  # Green
    elif overall >= 50:
        color = '\033[1;33m'  # Yellow
    else:
        color = '\033[0;31m'  # Red
    
    print(f'Overall Routing:      {color}{overall:5.1f}%\033[0m ({total_stmts} stmts, {total_miss} missing)')
" 2>/dev/null || echo "Coverage report not available"

echo ""

# Run benchmark tests if requested
if [ "$1" == "--benchmark" ]; then
    echo "========================================="
    echo "Running Benchmark Tests"
    echo "========================================="
    
    uv run pytest tests/ \
        -v \
        -m "benchmark" \
        --tb=short \
        --durations=10
fi

# Open coverage report in browser if requested
if [ "$1" == "--open" ] || [ "$2" == "--open" ]; then
    echo ""
    echo "Opening coverage report in browser..."
    open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null
fi

# Exit with test status
exit $TEST_EXIT_CODE