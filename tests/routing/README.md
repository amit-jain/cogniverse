# Cogniverse Routing Testing Suite

**Last Updated:** 2025-11-13

This directory contains comprehensive tests for query routing and analysis in the **cogniverse-agents** package (Implementation Layer). Tests validate DSPy 3.0 routing with GEPA optimization, GLiNER entity extraction, multi-modal query classification (video, audio, images, documents, text, dataframes), and relationship-based relevance boosting.

## 🚀 Quick Start

### Run All Tests (Recommended)

**Critical**: Always use `JAX_PLATFORM_NAME=cpu` for DSPy routing tests and UV workspace:

```bash
# Run comprehensive test with all models and modes (UV workspace)
JAX_PLATFORM_NAME=cpu uv run python tests/test_comprehensive_routing.py

# Quick test with limited models (faster)
python tests/test_comprehensive_routing.py quick
```

### Test Specific Components
```bash
# Test only LLM models
python tests/test_comprehensive_routing.py llm-only

# Test only GLiNER models  
python tests/test_comprehensive_routing.py gliner-only

# Test only hybrid modes
python tests/test_comprehensive_routing.py hybrid-only
```

### Help
```bash
python tests/test_comprehensive_routing.py help
```

## 📊 Test Files

### Main Test Suite
- **`test_comprehensive_routing.py`** - **PRIMARY TEST FILE**
  - Unified test suite combining LLM, GLiNER, and hybrid testing
  - Tests 10 LLM models across multiple model families
  - Tests all configured GLiNER models
  - Tests hybrid approaches with best-performing combinations
  - Comprehensive metrics and reporting

### Legacy Tests (Deprecated)
- ~~`test_llm_routing.py`~~ - Replaced by comprehensive test
- ~~`test_gliner_models.py`~~ - Replaced by comprehensive test  
- ~~`test_quick_routing.py`~~ - Functionality merged into comprehensive test

### Supporting Tests
- `test_system.py` - System integration tests
- `analyze_failures.py` - Test failure analysis
- `test_queries.txt` - Test query definitions

## 🧪 Test Coverage

### LLM Models Tested
- **DeepSeek R1**: 1.5b, 7b, 8b variants
- **Gemma 3**: 1b, 4b, 12b variants  
- **Qwen 3**: 0.6b, 1.7b, 4b, 8b variants

### GLiNER Models
- All models configured in `config.json` under `query_inference_engine.available_gliner_models`

### Test Categories
1. **Routing Accuracy** - Video vs Text vs Both classification
2. **Temporal Extraction** - Date/time pattern recognition
3. **Response Time** - Performance metrics
4. **Success Rate** - Inference reliability
5. **Hybrid Performance** - Combined LLM + GLiNER effectiveness

## 📈 Output Files

### Generated Reports
- `comprehensive_test_results.csv` - Detailed per-query results
- `comprehensive_summary_report.json` - Aggregated metrics and rankings

### Report Contents
- **Overall Rankings** - Best performing models across all metrics
- **Category Breakdowns** - LLM vs GLiNER vs Hybrid comparisons
- **Performance Metrics** - Accuracy, speed, reliability analysis
- **Best Performers** - Top models for specific use cases

## 🔧 Configuration

### Test Queries
Edit `test_queries.txt` to modify test cases:
```
query_text, expected_routing, expected_temporal
Show me videos from yesterday, video, yesterday
Find documents about AI, text, none
Search content from 2024-01-15, both, 2024-01-15
```

### Model Configuration
Configure available models in `config.json`:
```json
{
  "local_llm_model": "qwen3:4b",
  "query_inference_engine": {
    "available_gliner_models": [
      "urchade/gliner_multi-v2.1",
      "urchade/gliner_large-v2.1"
    ]
  }
}
```

## 🎯 Understanding Results

### Accuracy Metrics
- **Overall Accuracy**: Combined routing + temporal correctness
- **Routing Accuracy**: Video/text/both classification accuracy  
- **Temporal Accuracy**: Date/time extraction accuracy
- **Success Rate**: Percentage of successful inferences

### Model Types
- **LLM**: Large Language Model only
- **GLiNER**: Generic Named Entity Recognition model only
- **Hybrid**: Combined LLM + GLiNER approach

### Performance Indicators
- **🌟 90%+**: Excellent performance
- **✅ 80-89%**: Very good performance  
- **✅ 70-79%**: Good performance
- **⚠️ <70%**: Needs improvement

## 🚦 Example Output

```
🏆 COMPREHENSIVE ROUTING TEST RESULTS
================================================================================

🎯 OVERALL RANKINGS (Total Queries: 25)
--------------------------------------------------------------------------------
 1. LLM: qwen3:8b                    92.5% (R:95.0% T:90.0% 1.23s)
 2. HYBRID: gliner_multi-v2.1        90.0% (R:92.0% T:88.0% 0.87s)
 3. LLM: gemma3:12b                  88.5% (R:90.0% T:87.0% 2.14s)

🥇 BEST PERFORMERS
--------------------------------------------------
Overall:  llm:qwen3:8b (92.5%)
Routing:  llm:qwen3:8b (95.0%)
Temporal: hybrid:gliner_multi-v2.1 (90.0%)
Fastest:  gliner:gliner_multi-v2.1 (0.65s)
```

## 🔄 Migration Guide

If you were using the old test files:

### From `test_llm_routing.py`
```bash
# Old way
python tests/test_llm_routing.py

# New way  
python tests/test_comprehensive_routing.py llm-only
```

### From `test_gliner_models.py`
```bash
# Old way
python tests/test_gliner_models.py

# New way
python tests/test_comprehensive_routing.py gliner-only
```

### From `test_quick_routing.py`
```bash
# Old way
python tests/test_quick_routing.py

# New way
python tests/test_comprehensive_routing.py quick
```

## 📝 Contributing

### Adding New Test Queries
1. Edit `test_queries.txt`
2. Follow format: `query, routing_type, temporal_pattern`
3. Run tests to validate

### Adding New Models
1. Update `config.json` with new model names
2. Ensure models are available in your environment
3. Run comprehensive test to benchmark

### Custom Analysis
Use the generated CSV files for custom analysis:
```python
import pandas as pd
results = pd.read_csv('tests/comprehensive_test_results.csv')
# Your analysis here
```

## 🐛 Troubleshooting

### Common Issues
- **No GLiNER models**: Configure models in `config.json`
- **Model not found**: Ensure Ollama models are installed
- **Permission denied**: Run `chmod +x tests/test_comprehensive_routing.py`
- **Import errors**: Check Python path and dependencies

### Getting Help
1. Run with `help` argument for usage information
2. Check generated error messages in CSV output
3. Verify model availability with Ollama

---

**🎉 Happy Testing!** The comprehensive test suite provides everything you need to evaluate and optimize your query routing system. 