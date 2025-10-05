# Multi-Agent System Testing Guide

This directory contains comprehensive tests for the multi-agent routing system with three distinct testing layers: unit tests, integration tests, and end-to-end (e2e) tests.

## Testing Architecture

### Test Structure
```
tests/agents/
├── unit/                    # Unit tests - fast, isolated, mocked
├── integration/             # Integration tests - component interactions
├── e2e/                     # End-to-end tests - full system, real services
└── README.md               # This documentation
```

### Testing Philosophy

1. **Unit Tests**: Fast, isolated tests with comprehensive mocking
2. **Integration Tests**: Test component interactions with selective mocking
3. **E2E Tests**: Real system tests using actual services (Ollama, Vespa)

## Unit Tests (`tests/agents/unit/`)

### Purpose
Test individual agent components in isolation with full mocking of external dependencies.

### Key Features
- **Fast execution** (< 1 second per test)
- **Complete mocking** of external services
- **High coverage** of core logic
- **Deterministic results**

### Test Files

#### `test_routing_agent.py`
Tests the central RoutingAgent functionality:
```python
class TestRoutingAgent:
    def test_analyze_and_route_video_query(self)
    def test_route_to_video_search_agent(self)
    def test_route_to_summarizer_agent(self)
    def test_route_to_detailed_report_agent(self)
    def test_unsupported_query_type(self)
```

#### `test_a2a_routing_agent.py`
Tests A2A (Agent-to-Agent) communication protocol:
```python
class TestA2ARoutingAgent:
    def test_send_to_agent_success(self)
    def test_send_to_agent_http_error(self)
    def test_send_to_agent_connection_error(self)

class TestA2ARoutingAgentIntegration:
    def test_video_search_routing_integration(self)
    def test_summarizer_routing_integration(self)
```

#### `test_video_search_agent.py`
Tests video search capabilities with VLM integration:
```python
class TestVideoProcessor:
    def test_extract_frames_and_encode(self)
    def test_extract_frames_error_handling(self)

class TestVideoSearchAgent:
    def test_analyze_visual_content(self)
    def test_search_videos(self)
    def test_process_search_request(self)
```

#### `test_query_analysis_tool_v3.py`
Tests query analysis and routing decisions:
```python
class TestQueryAnalysisToolV3:
    def test_analyzer_initialization_default(self)
    def test_analyze_simple_query(self)
    def test_analyze_complex_query(self)
    def test_routing_decision_logic(self)
```

#### `test_dspy_integration.py`
Tests DSPy optimization integration:
```python
class TestDSPyIntegrationMixin:
    def test_mixin_initialization(self)
    def test_prompt_optimization(self)

class TestDSPyAgentIntegration:
    def test_routing_agent_dspy_integration(self)
    def test_summarizer_agent_dspy_integration(self)
```

### Running Unit Tests
```bash
# Run all unit tests
JAX_PLATFORM_NAME=cpu uv run python -m pytest tests/agents/unit/ -v

# Run specific test file
JAX_PLATFORM_NAME=cpu uv run python -m pytest tests/agents/unit/test_routing_agent.py -v

# Run with coverage
JAX_PLATFORM_NAME=cpu uv run python -m pytest tests/agents/unit/ --cov=src/app/agents --cov-report=term-missing
```

## Integration Tests (`tests/agents/integration/`)

### Purpose
Test interactions between components with selective mocking of external services.

### Key Features
- **Component interaction testing**
- **Selective mocking** (mock external APIs, keep internal logic)
- **Moderate execution time** (1-10 seconds per test)
- **Real agent communication patterns**

### Test Files

#### `test_specialized_agents_integration.py`
Tests specialized agents (SummarizerAgent, DetailedReportAgent) with OpenAI-compatible APIs:
```python
class TestSummarizerAgentIntegration:
    def test_generate_summary_with_openai_api(self)
    def test_analyze_visual_content_enabled(self)
    def test_process_request_full_pipeline(self)

class TestDetailedReportAgentIntegration:
    def test_generate_report_with_openai_api(self)
    def test_process_video_results(self)
    def test_generate_report_edge_cases(self)
```

#### `test_routing_agent_integration.py`
Tests routing agent with FastAPI integration:
```python
class TestRoutingAgentFastAPIIntegration:
    def test_agent_card_endpoint(self)
    def test_send_task_endpoint(self)
    def test_health_check_endpoint(self)
```

#### `test_dspy_optimization_integration.py`
Tests DSPy optimization pipeline with mocked teleprompters:
```python
class TestDSPyOptimizerIntegration:
    def test_pipeline_optimization_with_mocked_teleprompter(self)
    def test_optimizer_with_local_llm(self)
    def test_module_optimization_with_training_data(self)
```

### Running Integration Tests
```bash
# Run all integration tests
JAX_PLATFORM_NAME=cpu timeout 300 uv run python -m pytest tests/agents/integration/ -v

# Run specific integration test
JAX_PLATFORM_NAME=cpu timeout 120 uv run python -m pytest tests/agents/integration/test_routing_agent_integration.py -v

# Run with markers
JAX_PLATFORM_NAME=cpu uv run python -m pytest tests/agents/integration/ -m integration -v
```

## End-to-End Tests (`tests/agents/e2e/`)

### Purpose
Test complete system functionality using real external services.

### Key Features
- **Real service integration** (Ollama, Vespa, Phoenix)
- **Complete workflow testing**
- **Service availability checks**
- **Graceful degradation** when services unavailable

### Prerequisites

#### Required Services
1. **Ollama** (for local LLM inference)
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull required models
   ollama pull smollm3:8b
   ollama pull llama3.2:3b
   ollama pull qwen2.5:3b
   ```

2. **Vespa** (for video search backend)
   ```bash
   # Start Vespa container
   ./scripts/start_vespa.sh
   ```

3. **Phoenix** (optional - for telemetry)
   ```bash
   # Start Phoenix server
   ./scripts/start_phoenix.sh
   ```

#### Environment Variables
```bash
export OLLAMA_BASE_URL="http://localhost:11434/v1"
export OLLAMA_MODEL="smollm3:8b"
export VESPA_URL="http://localhost:8080"
export PHOENIX_URL="http://localhost:6006"
export ENABLE_VESPA_TESTS="true"
export ENABLE_PHOENIX_TESTS="true"
export E2E_TEST_TIMEOUT="300"
```

### Test Configuration (`tests/agents/e2e/test_config.py`)

The E2E tests use a centralized configuration system:
```python
E2E_CONFIG = {
    "ollama_base_url": "http://localhost:11434/v1",
    "ollama_model": "smollm3:8b",
    "vespa_url": "http://localhost:8080",
    "test_timeout": 300,
    "enable_vespa_tests": False,  # Set to True when Vespa available
}
```

### Test Files

#### `test_real_multi_agent_integration.py`
Comprehensive real-world testing of the multi-agent system:

```python
class TestOllamaAvailability:
    def test_ollama_service_available(self)
    def test_ollama_model_available(self)

class TestRealQueryAnalysis:
    def test_real_query_analysis_with_ollama(self)
    def test_query_analysis_different_types(self)

class TestRealAgentRouting:
    def test_real_routing_decisions(self)
    def test_agent_communication_flow(self)

class TestRealSpecializedAgents:
    def test_real_summarizer_agent(self)
    def test_real_detailed_report_agent(self)

class TestRealDSPyIntegration:
    def test_real_dspy_optimization_pipeline(self)
    def test_dspy_with_training_data(self)

class TestRealEndToEndWorkflow:
    def test_complete_video_search_workflow(self)
    def test_multi_agent_collaboration(self)

class TestRealPerformanceAndReliability:
    def test_agent_response_times(self)
    def test_error_handling_with_real_services(self)
```

### Running E2E Tests

#### Prerequisites Check
```bash
# Check service availability
JAX_PLATFORM_NAME=cpu uv run python -c "
from tests.agents.e2e.test_config import is_service_available
print('Ollama:', is_service_available('ollama'))
print('Vespa:', is_service_available('vespa'))
print('Phoenix:', is_service_available('phoenix'))
"
```

#### Basic E2E Tests
```bash
# Run all E2E tests (skips unavailable services)
JAX_PLATFORM_NAME=cpu timeout 600 uv run python -m pytest tests/agents/e2e/ -v

# Run only Ollama tests
JAX_PLATFORM_NAME=cpu timeout 300 uv run python -m pytest tests/agents/e2e/test_real_multi_agent_integration.py::TestOllamaAvailability -v

# Run with specific timeout
JAX_PLATFORM_NAME=cpu timeout 900 uv run python -m pytest tests/agents/e2e/ -v --tb=short
```

#### Full System Tests (requires all services)
```bash
# Enable all services and run complete tests
ENABLE_VESPA_TESTS=true ENABLE_PHOENIX_TESTS=true \
JAX_PLATFORM_NAME=cpu timeout 1800 uv run python -m pytest tests/agents/e2e/ -v
```

## Test Markers and Configuration

### Pytest Markers
```ini
# pytest.ini
[tool:pytest]
markers =
    unit: Unit tests (fast, mocked)
    integration: Integration tests (moderate, selective mocking)
    e2e: End-to-end tests (slow, real services)
    timeout: Tests with custom timeout requirements
```

### Running by Markers
```bash
# Run only unit tests
uv run python -m pytest -m unit tests/agents/

# Run integration and e2e tests
uv run python -m pytest -m "integration or e2e" tests/agents/

# Run all tests except e2e
uv run python -m pytest -m "not e2e" tests/agents/
```

## CI/CD Integration

### GitHub Actions Configuration
The tests are integrated into CI with different strategies:

#### Unit Tests (Always Run)
```yaml
- name: Run Unit Tests
  run: |
    JAX_PLATFORM_NAME=cpu uv run python -m pytest tests/agents/unit/ \
      --cov=src/app/agents --cov-report=xml -v
```

#### Integration Tests (Run with Docker services)
```yaml
- name: Run Integration Tests
  run: |
    JAX_PLATFORM_NAME=cpu timeout 600 uv run python -m pytest tests/agents/integration/ -v
```

#### E2E Tests (Optional/Manual)
```yaml
- name: Run E2E Tests
  if: env.RUN_E2E_TESTS == 'true'
  run: |
    JAX_PLATFORM_NAME=cpu timeout 1800 uv run python -m pytest tests/agents/e2e/ -v
```

## Performance Benchmarks

### Test Execution Times
- **Unit Tests**: ~30 seconds (38 tests)
- **Integration Tests**: ~2-5 minutes (depends on API responses)
- **E2E Tests**: ~5-15 minutes (depends on model loading)

### Coverage Goals
- **Unit Tests**: >95% line coverage
- **Integration Tests**: Critical path coverage
- **E2E Tests**: User workflow coverage

## Troubleshooting

### Common Issues

#### 1. Ollama Service Not Available
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

#### 2. Model Not Found
```bash
# Pull required models
ollama pull smollm3:8b
ollama pull llama3.2:3b
```

#### 3. Test Timeouts
```bash
# Increase timeout for slow models
E2E_TEST_TIMEOUT=600 uv run python -m pytest tests/agents/e2e/ -v
```

#### 4. Memory Issues with Large Models
```bash
# Use smaller models for testing
OLLAMA_MODEL=smollm3:8b uv run python -m pytest tests/agents/e2e/ -v
```

#### 5. DSPy Configuration Issues
```bash
# Check DSPy async context
import dspy
with dspy.context(lm=your_lm):
    # Your DSPy operations
```

### Debug Mode
```bash
# Enable verbose logging
PYTHONPATH=. uv run python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# Run specific test
"

# Run single test with full output
JAX_PLATFORM_NAME=cpu uv run python -m pytest tests/agents/e2e/test_real_multi_agent_integration.py::TestOllamaAvailability::test_ollama_model_available -v -s --tb=long
```

## Best Practices

### Writing Unit Tests
1. **Mock external dependencies** completely
2. **Test edge cases** and error conditions
3. **Keep tests fast** (< 1 second each)
4. **Use descriptive test names**
5. **Follow AAA pattern** (Arrange, Act, Assert)

### Writing Integration Tests
1. **Test component boundaries**
2. **Mock external APIs** but keep internal logic
3. **Test realistic scenarios**
4. **Handle async operations properly**
5. **Use appropriate timeouts**

### Writing E2E Tests
1. **Check service availability** before testing
2. **Use graceful degradation** (skip if service unavailable)
3. **Test complete user workflows**
4. **Include performance assertions**
5. **Clean up resources** after tests

### Test Data Management
1. **Use fixtures** for common test data
2. **Isolate test environments**
3. **Clean up after tests**
4. **Use deterministic test data**
5. **Document test scenarios**

## Contributing

### Adding New Tests
1. **Choose appropriate test level** (unit/integration/e2e)
2. **Follow existing patterns** and naming conventions
3. **Add proper documentation** and comments
4. **Include error cases** and edge conditions
5. **Update this README** if adding new test categories

### Test Review Checklist
- [ ] Tests are in correct directory (unit/integration/e2e)
- [ ] Appropriate mocking strategy used
- [ ] Error cases covered
- [ ] Performance considerations addressed
- [ ] Documentation updated
- [ ] CI integration considered

---

For more information about the multi-agent system itself, see [src/app/agents/README.md](../../src/app/agents/README.md).