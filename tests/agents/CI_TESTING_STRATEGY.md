# Agent Tests CI Strategy

## Overview
To reduce CI execution time while maintaining test coverage, we use a two-tier testing approach:

## CI Fast Tests (`@pytest.mark.ci_fast`)
- Essential tests that cover core functionality
- Run in CI for every commit (78 tests, ~1:48 minutes)
- **37% coverage** of src/app/agents with key components:
  - detailed_report_agent.py: 82% coverage (comprehensive core functionality tests)
  - summarizer_agent.py: 76% coverage (thinking phase, theme extraction, content categorization)
  - routing_agent.py: Core routing and workflow determination
  - enhanced_video_search_agent.py: Search functionality and video processing
  - DSPy integration: Relationship extraction, query enhancement, optimization
  - Multi-agent orchestration: Workflow planning, result aggregation

## Full Test Suite
- Complete test coverage (381 tests, ~6+ minutes for unit tests)
- Run locally during development
- Run on main branch pushes for comprehensive validation

## Usage

### CI (Automatic)
```bash
# Runs automatically in CI
pytest tests/agents/unit -m "unit and ci_fast"
```

### Local Development
```bash
# Run all tests (comprehensive)
uv run python -m pytest tests/agents/unit -m unit

# Run only fast tests (quick validation)
uv run python -m pytest tests/agents/unit -m "unit and ci_fast"
```

## Marking Tests as CI Fast
Mark essential tests with both markers:
```python
@pytest.mark.unit
@pytest.mark.ci_fast
def test_essential_functionality(self):
    # Core functionality test
    pass
```

## Current CI Fast Tests Summary
78 tests across 8 test modules covering:

### Core Agent Tests (39 tests)
- **Detailed Report Agent** (16 tests): Thinking phase, visual analysis, report generation, enhanced reporting
- **Summarizer Agent** (12 tests): Summary generation, theme extraction, content categorization, visual analysis
- **Enhanced Video Search Agent** (3 tests): Initialization, text search, video search
- **Routing Agent** (6 tests): Initialization, routing decisions, workflow determination
- **Agent Registry** (5 tests): Registry operations and agent management

### DSPy Integration Tests (9 tests)
- Relationship extraction, query enhancement, basic routing modules
- Enhanced routing agent with DSPy optimization
- Signature validation and module structure

### Multi-Agent Orchestration (22 tests)
- **Workflow Intelligence** (12 tests): Template generation, workflow optimization, execution tracking
- **Result Aggregator** (7 tests): Result enhancement, aggregation logic, edge cases  
- **Multi-Agent Orchestrator** (4 tests): Workflow planning, agent utilization tracking

### System Integration (8 tests)
- Agent endpoint creation, registry initialization
- DSPy optimizer initialization and configuration
- Video processor initialization and encoding