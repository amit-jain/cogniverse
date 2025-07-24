# Testing Documentation

This directory contains testing documentation for the Cogniverse multi-agent system.

## Test Categories

### 1. System Tests
- **Location**: `tests/test_system.py`
- **Purpose**: End-to-end system validation
- **Coverage**: All major components and workflows

### 2. Search Client Tests
- **Location**: `tests/test_search_client.py`
- **Purpose**: Comprehensive validation of all 13 Vespa ranking strategies
- **Coverage**: Text-only, visual, and hybrid search capabilities
- **Documentation**: [search_client_testing.md](search_client_testing.md)

### 3. Component Tests
- **Location**: `tests/test_colpali_search.py`
- **Purpose**: ColPali text-to-video search validation
- **Coverage**: Core semantic search functionality
- **Features**: Table output, CSV export, configurable test selection

### 4. Comprehensive Video Query Test
- **Location**: `comprehensive_video_query_test.py`
- **Purpose**: Compare binary vs float search accuracy
- **Coverage**: Multiple test queries per video with expected results
- **Features**: Summary statistics table, success rate analysis, CSV/JSON export

### 5. Routing Tests
- **Location**: `tests/routing/test_combined_routing.py`
- **Purpose**: Query routing optimization and model performance
- **Coverage**: LLM and GLiNER model comparison

## Quick Testing Commands

```bash
# Run all system tests
python tests/test_system.py

# Test comprehensive search capabilities
python tests/test_search_client.py

# Test specific ColPali visual search with table output
python tests/test_colpali_search.py

# Test with CSV export
python tests/test_colpali_search.py --save

# Test only binary search
python tests/test_colpali_search.py --test binary

# Test with text format instead of table
python tests/test_colpali_search.py --format text

# Comprehensive video query test with summary table
python comprehensive_video_query_test.py --format table --save

# Test query routing optimization
python tests/routing/test_combined_routing.py
```

## Documentation Index

- **[search_client_testing.md](search_client_testing.md)** - How to test the Vespa search client
- **[vespa_search_strategies.md](vespa_search_strategies.md)** - Complete guide to all 13 ranking strategies

## Search Client Test Details

The comprehensive search client test (`test_search_client.py`) validates:

### Ranking Strategies Tested
- **Text-Only**: `bm25_only` with fieldsets
- **Visual**: `float_float`, `binary_binary`, `float_binary`, `phased`
- **Hybrid**: `hybrid_float_bm25`, `hybrid_binary_bm25`, `hybrid_bm25_binary`, `hybrid_bm25_float`
- **No-Description**: All hybrid variants excluding frame descriptions

### Validation Features
- **Input Validation**: Ensures required inputs (text/embeddings) are provided
- **Strategy Recommendation**: Automatic strategy selection based on query type
- **Error Handling**: Proper error messages for missing requirements
- **Performance Testing**: Response time and accuracy measurements

### Technical Features Tested
- **BM25 Fieldsets**: Multi-field text search functionality
- **ColPali Embeddings**: Both float and binary embedding support
- **Multi-Vector Search**: Patch-based visual similarity
- **Hybrid Ranking**: Combined text and visual scoring

## Enhanced Test Output Features

### Table Formatting
All test scripts now support human-readable table output using the `tabulate` library:
- **Default Format**: Grid tables with clear headers and alignment
- **Command**: `--format table` (default) or `--format text` for legacy output
- **Benefits**: Easy visual comparison of results, scores, and metadata

### CSV/JSON Export
Test results can be exported for further analysis:
- **Command**: Add `--save` flag to any test
- **Output Directory**: `test_results/` (created automatically)
- **Filename Format**: `{test_name}_{timestamp}.csv` or `.json`
- **Contents**: All test results with headers matching table columns

### Test Utilities
The `tests/test_utils.py` module provides:
- **TestResultsFormatter**: Consistent formatting across all tests
- **Comparison Tables**: Side-by-side result comparison
- **Summary Statistics**: Automatic calculation of success rates
- **Multiple Export Formats**: CSV, JSON, and HTML tables

### Configuration
Test output behavior can be configured in `config.json`:
```json
{
  "test_output_format": "table",  // or "text"
  "test_results_dir": "test_results"
}
```

## Test Data Requirements

### For `test_search_client.py`:
- **Vespa Instance**: Running on localhost:8080
- **Test Data**: Any indexed video content (will work with empty index)
- **Models**: ColPali model (`vidore/colsmol-500m`) auto-loaded

### For `test_colpali_search.py`:
- **Vespa Instance**: Running on localhost:8080
- **Test Data**: Indexed video frames
- **Models**: ColPali model for embedding generation

## Troubleshooting Test Failures

### Common Issues:

**Vespa Connection Failed:**
- Ensure Vespa is running: `docker ps | grep vespa`
- Check port accessibility: `curl localhost:8080/ApplicationStatus`
- Restart if needed: `./scripts/start_vespa.sh`

**Model Loading Failures:**
- Verify sufficient RAM (16GB+ recommended)
- Check internet connection for model downloads
- Clear model cache if corrupted

**Search Returns 0 Results:**
- Verify data is indexed: Check Vespa coverage in results
- Ensure proper schema deployment
- Re-run ingestion if needed

### Debug Commands:

```bash
# Check Vespa status
curl localhost:8080/ApplicationStatus

# Test basic search
curl "localhost:8080/search/?yql=select * from video_frame where true&hits=1"

# Validate schema
python -c "from src.processing.vespa.vespa_search_client import VespaVideoSearchClient; print('OK' if VespaVideoSearchClient().health_check() else 'FAILED')"
```