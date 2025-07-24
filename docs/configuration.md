# Configuration Management

## Overview

The Cogniverse system uses a centralized configuration management system through the `Config` class in `src/tools/config.py`. This provides a singleton pattern to ensure consistent configuration across all components.

## Configuration Loading

### Basic Usage

```python
from src.tools.config import get_config

# Get the config instance
config = get_config()

# Access configuration values
vespa_url = config.get("vespa_url")
vespa_schema = config.vespa_schema  # Using property accessor
model_name = config.get("colpali_model", "default-model")
```

### Dot Notation Support

The Config class supports dot notation for nested configuration values:

```python
# Access nested values
gliner_model = config.get("query_inference_engine.current_gliner_model")
optimization_enabled = config.get("optimization.enabled", False)

# Set nested values
config.set("optimization.settings.num_trials", 50)
```

## Configuration Properties

Common configuration values are available as properties for convenience:

```python
config.vespa_schema      # Vespa schema name (default: "video_frame")
config.vespa_url        # Vespa URL (default: "http://localhost")
config.vespa_port       # Vespa port (default: 8080)
config.colpali_model    # ColPali model name
config.search_backend   # Search backend ("vespa" or "byaldi")
config.log_level        # Logging level
```

## Configuration File Locations

The system searches for `config.json` in the following order:
1. `configs/config.json` (standard location)
2. Project root `configs/` directory
3. Current working directory `configs/`
4. `~/.cogniverse/config.json` (user home)

If no config file is found, the system will use default values.

## Environment Variables

Environment variables override config file values:

```bash
export VESPA_URL="http://production-vespa"
export VESPA_SCHEMA="custom_schema"
export COLPALI_MODEL="vidore/colpali-v1.2"
```

## Reloading Configuration

To reload configuration after changes:

```python
config = get_config()
config.reload()  # Reloads from file
```

## Saving Configuration

To save current configuration state:

```python
config.save()  # Saves to original location
config.save("custom_config.json")  # Save to specific file
```

## Schema-Specific Configuration

The Vespa schema name is now centrally configured and used by all components:

```json
{
  "vespa_schema": "video_frame"
}
```

This is automatically used by:
- `EmbeddingGenerator` for document ingestion
- `VespaVideoSearchClient` for search queries
- All test scripts for query construction

## Test Configuration

Test-specific settings can be configured:

```json
{
  "test_output_format": "table",  // "table" or "text"
  "test_results_dir": "test_results"
}
```

These are used by test scripts when the `--format` flag is not provided.

## Example Configuration

```json
{
  "vespa_url": "http://localhost",
  "vespa_port": 8080,
  "vespa_schema": "video_frame",
  "search_backend": "vespa",
  "colpali_model": "vidore/colsmol-500m",
  "local_llm_model": "deepseek-r1:7b",
  "base_url": "http://localhost:11434",
  "log_level": "INFO",
  "pipeline_config": {
    "extract_keyframes": true,
    "transcribe_audio": true,
    "generate_descriptions": true,
    "generate_embeddings": true
  },
  "test_output_format": "table"
}
```

## Best Practices

1. **Always use the Config class** instead of hardcoding values
2. **Use properties** for commonly accessed values
3. **Handle defaults** when getting configuration values
4. **Validate required config** using `config.validate_required_config()`
5. **Avoid direct config_data access** - use get/set methods instead