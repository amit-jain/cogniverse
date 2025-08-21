# Vespa Feed Configuration

## Overview

The Vespa ingestion client now includes production-ready configuration for the `feed_iterable` method from pyvespa. This configuration helps prevent connection errors, memory issues, and improves reliability during bulk document feeding.

## Configuration Parameters

### Available Parameters

The following parameters can be configured for optimal feeding performance:

| Parameter | Default | Description | Environment Variable |
|-----------|---------|-------------|---------------------|
| `max_queue_size` | 500 | Maximum number of documents in the feeding queue | `VESPA_FEED_MAX_QUEUE_SIZE` |
| `max_workers` | 4 | Number of worker threads for parallel feeding | `VESPA_FEED_MAX_WORKERS` |
| `max_connections` | 8 | Maximum number of HTTP connections to Vespa | `VESPA_FEED_MAX_CONNECTIONS` |
| `compress` | "auto" | Compression setting for requests ("auto", true, false) | `VESPA_FEED_COMPRESS` |

### Parameter Details

#### max_queue_size
- Controls the maximum number of documents that can be queued for feeding
- Lower values (500) prevent memory issues with large documents
- Higher values may improve throughput but increase memory usage
- Default pyvespa value: 1000

#### max_workers
- Number of concurrent threads processing the feed queue
- Should be balanced with max_connections
- Lower values (4) provide better stability
- Higher values may improve throughput on powerful machines
- Default pyvespa value: 8

#### max_connections
- Maximum number of persistent HTTP connections to Vespa
- Should be approximately 2x max_workers for optimal performance
- Too many connections may overwhelm Vespa
- Default pyvespa value: 16

#### compress
- "auto": Automatically compress requests larger than 1024 bytes
- true: Always compress
- false: Never compress
- Compression reduces network traffic but increases CPU usage

## Configuration Methods

### 1. Environment Variables

Set environment variables before running the ingestion:

```bash
export VESPA_FEED_MAX_QUEUE_SIZE=500
export VESPA_FEED_MAX_WORKERS=4
export VESPA_FEED_MAX_CONNECTIONS=8
export VESPA_FEED_COMPRESS=auto

uv run python scripts/run_ingestion.py --video_dir /path/to/videos --backend vespa
```

### 2. Configuration File

Add to your backend configuration:

```python
config = {
    "vespa_url": "http://localhost",
    "vespa_port": 8080,
    "schema_name": "video_schema",
    "feed_max_queue_size": 500,
    "feed_max_workers": 4,
    "feed_max_connections": 8,
    "feed_compress": "auto"
}
```

### 3. Priority Order

Configuration is applied in the following priority order (highest to lowest):
1. Environment variables
2. Configuration file settings
3. Default values

## Retry and Error Handling

The updated implementation includes:

1. **Connection Retry**: The `connect()` method has a retry decorator with exponential backoff
2. **Error Tracking**: Detailed logging of failed documents with HTTP status codes
3. **Retry Counting**: Tracks retry attempts per document for debugging
4. **Batch Recovery**: Continues processing even if individual documents fail

## Performance Tuning

### For Small Documents (< 1KB)
```bash
export VESPA_FEED_MAX_QUEUE_SIZE=1000
export VESPA_FEED_MAX_WORKERS=8
export VESPA_FEED_MAX_CONNECTIONS=16
export VESPA_FEED_COMPRESS=false
```

### For Large Documents (> 10KB)
```bash
export VESPA_FEED_MAX_QUEUE_SIZE=200
export VESPA_FEED_MAX_WORKERS=2
export VESPA_FEED_MAX_CONNECTIONS=4
export VESPA_FEED_COMPRESS=true
```

### For Mixed Workloads (Default)
```bash
export VESPA_FEED_MAX_QUEUE_SIZE=500
export VESPA_FEED_MAX_WORKERS=4
export VESPA_FEED_MAX_CONNECTIONS=8
export VESPA_FEED_COMPRESS=auto
```

## Monitoring

The client logs important metrics:

- Feed configuration on initialization
- Batch progress (X/Y documents)
- Success/failure counts per batch
- Retry attempts for failed documents
- HTTP status codes for failures

Example log output:
```
INFO: Feed configuration: {'max_queue_size': 500, 'max_workers': 4, 'max_connections': 8, 'compress': 'auto'}
INFO: Processing batch 1/5 for schema 'video_schema' (100 documents)
INFO: Batch 1 to schema 'video_schema': 98/100 documents fed successfully
WARNING: Batch 1 had 2 failed documents (some may have been retried)
```

## Troubleshooting

### Connection Reset Errors
- Reduce `max_workers` and `max_connections`
- Increase delay between batches
- Check Vespa memory configuration

### Out of Memory Errors
- Reduce `max_queue_size`
- Reduce batch size
- Enable compression

### Slow Feeding
- Increase `max_workers` (carefully)
- Increase `max_connections` to 2x workers
- Check network latency to Vespa

## References

- [pyvespa feed_iterable documentation](https://vespa-engine.github.io/pyvespa/)
- [Vespa feeding best practices](https://docs.vespa.ai/en/feed-using-client-api.html)
- [Vespa performance tuning](https://docs.vespa.ai/en/performance/)