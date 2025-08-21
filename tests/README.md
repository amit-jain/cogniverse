# Routing System Test Suite

## Overview
Comprehensive test suite for the multi-tiered routing system with proper unit tests, integration tests, and demonstrations.

## Quick Start
```bash
# Install test dependencies
uv pip install pytest pytest-asyncio pytest-cov

# Run all tests
uv run python run_tests.py

# Run specific suite
uv run python run_tests.py --suite unit
uv run python run_tests.py --suite integration
uv run python run_tests.py --suite demo
```

## Test Organization

### Unit Tests (`tests/unit/`)
- **test_routing_strategies.py**: Tests each routing strategy in isolation
  - GLiNER strategy with mocked models
  - LLM strategy with mocked API calls
  - Keyword strategy logic
  - LangExtract strategy with mocked Ollama

### Integration Tests (`tests/integration/`)
- **test_tiered_routing.py**: Tests complete routing flow
  - Tier escalation logic
  - Generation type classification
  - Search modality detection
  - Caching behavior
  - Error handling

### Demonstration (`demo_routing_tiers.py`)
- Interactive demonstration of all 4 tiers
- Shows confidence scores and escalation decisions
- Provides performance statistics

## Key Test Features
- ✅ Mocked dependencies for unit tests
- ✅ Real model testing for integration (when available)
- ✅ Async test support
- ✅ Parametrized test cases
- ✅ Coverage reporting
- ✅ CI/CD ready

## Running Tests with Coverage
```bash
uv run pytest --cov=src/routing --cov-report=html
# Open htmlcov/index.html to view coverage report
```