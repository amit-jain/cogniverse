# End-to-End Integration Tests

**Last Updated:** 2025-11-13

This directory contains **real** end-to-end integration tests for **cogniverse-agents** (Implementation Layer) that use actual LLMs via Ollama, real DSPy 3.0 optimization with GEPA, and real backend services (Vespa, Phoenix) without mocks. Tests validate the complete 10-package architecture stack and multi-modal processing (video, audio, images, documents, text, dataframes).

## Test Categories

### 1. **TestOllamaAvailability**
- Checks if Ollama server is running and has required models
- Acts as a gate for all other real integration tests

### 2. **TestRealQueryAnalysisIntegration**
- Tests QueryAnalysisToolV3 with actual local LLMs
- Validates query analysis accuracy and response structure
- Tests different query types and complexity levels

### 3. **TestRealAgentRoutingIntegration**  
- Tests RoutingAgent routing decisions with real LLMs
- Validates agent selection and workflow recommendations
- Tests confidence scoring and reasoning

### 4. **TestRealAgentSpecializationIntegration**
- Tests SummarizerAgent and DetailedReportAgent with real LLMs
- Validates content summarization and report generation
- Tests response quality and structure

### 5. **TestRealDSPyOptimizationIntegration**
- Tests actual DSPy prompt optimization pipeline
- Validates DSPy integration with agents
- Tests optimization performance and metadata

### 6. **TestRealEndToEndWorkflow**
- Tests complete multi-agent workflows
- Validates agent communication and handoffs
- Tests realistic user scenarios

### 7. **TestRealPerformanceComparison**
- Compares optimized vs default agent performance
- Measures response times and quality metrics
- Validates DSPy optimization benefits

## Prerequisites

### Required Services

1. **Ollama Server**
   ```bash
   # Install and start Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama serve
   
   # Pull required models
   ollama pull smollm3:8b
   ```

2. **Python Dependencies**
   ```bash
   # Ensure all agent dependencies are installed
   uv sync
   ```

### Optional Services

1. **Vespa Backend** (for video search tests)
   ```bash
   ./scripts/start_vespa.sh
   ```

2. **Phoenix Telemetry** (for advanced metrics)
   ```bash
   ./scripts/start_phoenix.sh
   ```

## Running Tests

### Run All E2E Tests

**Critical**: Always use `JAX_PLATFORM_NAME=cpu` for DSPy tests and UV workspace:

```bash
# Run all end-to-end tests with UV workspace (will skip if services unavailable)
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/e2e/ -v

# Run with specific timeout
JAX_PLATFORM_NAME=cpu uv run python -m pytest tests/agents/e2e/ -v --timeout=600
```

### Run Specific Test Categories
```bash
# Test only Ollama availability
uv run python -m pytest tests/agents/e2e/::TestOllamaAvailability -v

# Test only query analysis
uv run python -m pytest tests/agents/e2e/::TestRealQueryAnalysisIntegration -v

# Test only DSPy optimization
uv run python -m pytest tests/agents/e2e/::TestRealDSPyOptimizationIntegration -v

# Test full workflow
uv run python -m pytest tests/agents/e2e/::TestRealEndToEndWorkflow -v
```

### Run with Different Models
```bash
# Use different model (must be available in Ollama)
OLLAMA_MODEL=llama3.2:3b uv run python -m pytest tests/agents/e2e/ -v

# Use different Ollama server
OLLAMA_BASE_URL=http://remote-server:11434/v1 uv run python -m pytest tests/agents/e2e/ -v
```

## Configuration

### Environment Variables

- `OLLAMA_BASE_URL`: Ollama API endpoint (default: http://localhost:11434/v1)
- `OLLAMA_MODEL`: Model to use for tests (default: smollm3:8b)
- `E2E_TEST_TIMEOUT`: Test timeout in seconds (default: 300)
- `DSPY_OPTIMIZATION_ROUNDS`: DSPy optimization rounds (default: 1)
- `MAX_TRAINING_EXAMPLES`: Max examples for training (default: 3)
- `ENABLE_VESPA_TESTS`: Enable Vespa-dependent tests (default: false)
- `ENABLE_PHOENIX_TESTS`: Enable Phoenix-dependent tests (default: false)
- `ENABLE_LONG_RUNNING_TESTS`: Enable long-running tests (default: false)

### Example Configuration
```bash
export OLLAMA_MODEL="smollm3:8b"
export E2E_TEST_TIMEOUT="600"
export ENABLE_VESPA_TESTS="true"
export ENABLE_LONG_RUNNING_TESTS="true"

uv run python -m pytest tests/agents/e2e/ -v
```

## Test Output Examples

### Successful Test Run
```
tests/agents/e2e/test_real_multi_agent_integration.py::TestOllamaAvailability::test_ollama_model_available PASSED
tests/agents/e2e/test_real_multi_agent_integration.py::TestRealQueryAnalysisIntegration::test_real_query_analysis_with_local_llm PASSED
tests/agents/e2e/test_real_multi_agent_integration.py::TestRealEndToEndWorkflow::test_real_multi_agent_workflow PASSED
```

### Skipped Due to Missing Services
```
tests/agents/e2e/test_real_multi_agent_integration.py::TestOllamaAvailability::test_ollama_model_available SKIPPED [100%]
SKIPPED [1] Model smollm3:8b not available in Ollama - skipping real integration tests
```

## Comparison with Existing Tests

### Mocked Integration Tests (`tests/agents/integration/`)
- **Purpose**: Fast, reliable, isolated testing
- **Use mocks**: Heavy mocking of LLMs and services  
- **Runtime**: Fast (~10-30 seconds)
- **Dependencies**: None (self-contained)
- **CI/CD**: Always run

### Real E2E Tests (`tests/agents/e2e/`)
- **Purpose**: Real-world validation and performance testing
- **Use real services**: Actual LLMs, DSPy, backends
- **Runtime**: Slower (~5-15 minutes)  
- **Dependencies**: Ollama, models, optional services
- **CI/CD**: Optional, environment-dependent

## Troubleshooting

### Common Issues

1. **"Model not available"**
   ```bash
   # Check available models
   ollama list
   
   # Pull missing model
   ollama pull smollm3:8b
   ```

2. **"Could not connect to Ollama"**
   ```bash
   # Check Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if needed
   ollama serve
   ```

3. **"Tests timeout"**
   ```bash
   # Increase timeout
   E2E_TEST_TIMEOUT=900 uv run python -m pytest tests/agents/e2e/ -v
   ```

4. **"DSPy optimization failed"**
   - This is expected in test environments
   - Tests will validate structure even if optimization fails
   - Use `ENABLE_LONG_RUNNING_TESTS=true` for full optimization

## Development

### Adding New E2E Tests

1. Add new test class to `test_real_multi_agent_integration.py`
2. Use `@pytest.mark.timeout(TEST_CONFIG["test_timeout"])` for timeouts
3. Check service availability in test setup
4. Use `pytest.skip()` for missing dependencies
5. Add meaningful assertions for real behavior validation

### Best Practices

1. **Graceful Degradation**: Always skip if services unavailable
2. **Realistic Scenarios**: Test actual user workflows  
3. **Performance Awareness**: Monitor test runtime and quality
4. **Environment Independence**: Don't assume specific model capabilities
5. **Comprehensive Validation**: Test both success and edge cases

## Integration with CI/CD

These tests are designed to be **optional** in CI/CD:

```yaml
# Example GitHub Actions
- name: Run E2E Tests (if Ollama available)
  run: |
    if curl -f http://localhost:11434/api/tags; then
      uv run python -m pytest tests/agents/e2e/ -v
    else
      echo "Ollama not available, skipping E2E tests"
    fi
```