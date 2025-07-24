# Cogniverse Test Suite

This directory contains all tests for the cogniverse multi-modal retrieval system.

## Directory Structure

### `/routing/` 
Query routing and temporal extraction tests and optimizations:
- **`test_combined_routing.py`** - Main unified test suite for LLM and GLiNER routing models
- **`combined_comparison_report.json`** - Consolidated test results (14 models: 10 LLM + 4 GLiNER)
- **`combined_test_results.csv`** - Detailed per-query results
- **`test_queries.txt`** - Test query dataset for routing evaluation
- **`gliner_optimizer.py`** - GLiNER model optimization and hyperparameter tuning
- **`teacher_student_optimizer.py`** - Teacher-student model optimization
- **`dspy_optimizer.py`** - DSPy framework optimization
- **`analyze_failures.py`** - Routing failure analysis tools

### Main Directory
- **`test_system.py`** - Comprehensive multi-agent system tests with optional test selection
- **`test_colpali_search.py`** - Text-to-video semantic search using ColPali embeddings
- **`test_document_similarity.py`** - Document similarity search (find similar frames)
- **`__init__.py`** - Test package initialization

## Running Tests

### Comprehensive System Tests
```bash
# Run all tests (from project root)
python tests/test_system.py

# Run specific tests
python tests/test_system.py --tests colpali_search end_to_end_system

# List available tests
python tests/test_system.py --list

# Available individual tests:
# - configuration: Config loading and validation
# - model_imports: Required library imports
# - data_directories: Data directory structure
# - local_llm_connectivity: Ollama server connection
# - agent_connectivity: A2A protocol communication
# - agent_discovery: Agent service discovery
# - video_search_agent: Video search functionality
# - colpali_search: Text-to-video semantic search
# - document_similarity: Frame similarity search
# - end_to_end_system: Complete multi-agent workflow
```

### Individual Component Tests
```bash
# Test ColPali semantic search
python tests/test_colpali_search.py

# Test document similarity (useful for UI "find similar" features)
python tests/test_document_similarity.py
```

### Routing Tests
```bash
# Run all routing tests (LLM + GLiNER)
python tests/routing/test_combined_routing.py

# Quick test (6 queries, all models)
python tests/routing/test_combined_routing.py quick

# Test only LLM models
python tests/routing/test_combined_routing.py llm-only

# Test only GLiNER models  
python tests/routing/test_combined_routing.py gliner-only
```

## Test Results

Current best performers (from consolidated results):
- **🥇 Overall Accuracy**: GLiNER Large (75.0%)
- **🎯 Best Routing**: GLiNER Large (83.3%) 
- **⏰ Best Temporal**: GLiNER Small/Medium/Multi (66.7%)
- **⚡ Fastest**: GLiNER Small (0.06s)

## Future Additions

This structure allows for easy addition of:
- `/video/` - Video processing and analysis tests
- `/text/` - Text processing and retrieval tests  
- `/multimodal/` - Cross-modal interaction tests
- `/performance/` - Performance and benchmarking tests 