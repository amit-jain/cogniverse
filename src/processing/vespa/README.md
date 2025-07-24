# Vespa Schema Management

This directory contains utilities for managing Vespa schemas, converting JSON schema definitions to PyVespa objects, and deploying them to Vespa instances.

## 📁 Files Overview

- **`json_schema_parser.py`** - Converts JSON schema definitions to PyVespa objects
- **`vespa_schema_manager.py`** - Manages schema deployment and validation
- **`README.md`** - This documentation

## 🚀 Quick Start

### 1. Deploy Schema from JSON

Deploy a JSON schema definition to Vespa:

```python
from src.processing.vespa.vespa_schema_manager import VespaSchemaManager

# Deploy comprehensive video schema with 9 ranking profiles
manager = VespaSchemaManager()
result = manager.upload_schema_from_json_file('schemas/video_frame_schema.json', 'videosearch')
print("Schema deployed successfully!")
```

### 2. Command Line Deployment

```bash
# Deploy schema using Python
python -c "
from src.processing.vespa.vespa_schema_manager import VespaSchemaManager
manager = VespaSchemaManager()
manager.upload_schema_from_json_file('schemas/video_frame_schema.json', 'videosearch')
print('✅ Schema deployed!')
"
```

## 🔍 Schema Validation Commands

### Check Vespa Application Status

```bash
# Check if Vespa is running and ready
curl -s http://localhost:8080/ApplicationStatus | head -10
```

### Verify Schema Deployment

```bash
# Basic search to verify schema is deployed
curl -s "http://localhost:8080/search/?yql=select%20*%20from%20sources%20*%20where%20true&hits=0"

# Expected response for successful deployment:
# {"root":{"id":"toplevel","relevance":1.0,"fields":{"totalCount":0},"coverage":{"coverage":100,"documents":0,"full":true,"nodes":1,"results":1,"resultsFull":1}}}
```

### Test All Ranking Profiles

```bash
# Test each ranking profile to ensure they're deployed correctly
python -c "
import requests

strategies = ['bm25_only', 'float_float', 'binary_binary', 'float_binary', 'phased', 'hybrid_float_bm25', 'binary_bm25', 'bm25_binary_rerank', 'bm25_float_rerank']

print('Testing all ranking profiles...')
for strategy in strategies:
    try:
        params = {'yql': 'select * from sources * where true', 'ranking': strategy, 'hits': 0}
        response = requests.get('http://localhost:8080/search/', params=params)
        status = '✅' if response.status_code == 200 else '❌'
        print(f'{status} {strategy}: {response.status_code}')
    except Exception as e:
        print(f'❌ {strategy}: Error - {e}')
"
```

### Verify Document Count

```bash
# Check number of documents in the index
curl -s "http://localhost:8080/search/?yql=select%20*%20from%20sources%20*%20where%20true&hits=0" | python -c "
import json, sys
data = json.load(sys.stdin)
count = data.get('root', {}).get('fields', {}).get('totalCount', 0)
print(f'Total documents: {count}')
"
```

## 📊 Current Schema: Video Frame Search

The current schema (`schemas/video_frame_schema.json`) includes:

### Document Fields
- `video_id` (string) - Video identifier
- `video_title` (string) - Video title with BM25 indexing
- `creation_timestamp` (long) - Video creation time
- `frame_id` (int) - Frame identifier within video
- `start_time`, `end_time` (double) - Frame time boundaries
- `frame_description` (string) - VLM-generated frame description
- `audio_transcript` (string) - Audio transcript for this frame
- `colpali_embedding` (tensor<float>) - Float embeddings (128-dim per patch)
- `colpali_binary` (tensor<int8>) - Binary embeddings (16-dim per patch)

### Ranking Profiles (9 Total)

1. **`bm25_only`** - Pure text search baseline
2. **`float_float`** - Direct ColPali with float embeddings
3. **`binary_binary`** - Hamming distance on binary embeddings
4. **`float_binary`** - Float query × unpack_bits(binary storage)
5. **`phased`** - Hamming first → Float reranking (20 candidates)
6. **`hybrid_float_bm25`** - Float ColPali + BM25 reranking
7. **`binary_bm25`** - Binary ColPali + BM25 combined
8. **`bm25_binary_rerank`** - BM25 first → Binary reranking
9. **`bm25_float_rerank`** - BM25 first → Float reranking

## 🛠️ Troubleshooting

### Schema Deployment Issues

```bash
# If deployment fails, check Vespa logs
docker logs vespa | tail -20

# Restart Vespa if needed
docker restart vespa

# Wait for Vespa to be ready (30 seconds)
sleep 30 && curl -s http://localhost:8080/ApplicationStatus
```

### Clean Deployment

To start completely fresh:

```bash
# Stop and remove Vespa container
docker stop vespa && docker rm vespa

# Start fresh Vespa instance
docker run -d --name vespa -p 8080:8080 -p 19071:19071 vespaengine/vespa

# Wait for startup (60 seconds)
sleep 60

# Deploy schema
python -c "
from src.processing.vespa.vespa_schema_manager import VespaSchemaManager
manager = VespaSchemaManager()
manager.upload_schema_from_json_file('schemas/video_frame_schema.json', 'videosearch')
"
```

### Validate Tensor Fields

```bash
# Test if tensor fields are properly configured
python -c "
import requests

# Test with dummy tensor inputs
params = {
    'yql': 'select * from sources * where true',
    'ranking': 'float_float',
    'hits': 0,
    'input.query(qt).querytoken0': '[0.1] * 128'  # Dummy float tensor
}

response = requests.get('http://localhost:8080/search/', params=params)
if response.status_code == 200:
    print('✅ Float tensor inputs working')
else:
    print(f'❌ Float tensor error: {response.text[:100]}')
"
```

## 📚 API Reference

### VespaSchemaManager Methods

- `upload_schema_from_json_file(json_path, app_name)` - Deploy JSON schema to Vespa
- `parse_sd_schema(sd_content)` - Parse .sd files to PyVespa objects
- `upload_frame_schema()` - Programmatic schema creation

### JsonSchemaParser Methods

- `load_schema_from_json_file(json_path)` - Load and parse JSON schema
- `parse_schema(schema_config)` - Convert JSON to PyVespa Schema
- `parse_rank_profile(rp_config)` - Convert JSON rank profile
- `validate_schema_config(schema_config)` - Validate JSON structure

## 🎯 Next Steps

After successful schema deployment:

1. **Ingest video data** using `scripts/run_ingestion.py`
2. **Benchmark ranking profiles** using `src/utils/comprehensive_query_utils.py`
3. **Test with Video-ChatGPT Q&A** dataset for evaluation
4. **Optimize performance** based on benchmarking results