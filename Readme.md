# Cogniverse - Multi-Modal Video Search System

A high-performance video content analysis and search system with configurable processing pipelines, built on state-of-the-art models like ColPali, VideoPrism, and ColQwen.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- 16GB+ RAM  
- CUDA-capable GPU (recommended)
- uv package manager: `pip install uv`

### Installation & Setup

```bash
# Clone and setup
git clone <repo>
cd cogniverse

# Install dependencies
uv sync

# Setup Vespa backend (required)
./scripts/start_vespa.sh

# Wait for Vespa to be ready
curl -s --head http://localhost:8080/ApplicationStatus
```

### Core Operations

#### Video Ingestion
```bash
# Process videos with specific profiles
uv run python scripts/run_ingestion.py \
    --video_dir data/testset/evaluation/sample_videos \
    --backend vespa \
    --profile video_colpali_smol500_mv_frame
```

Available profiles:
- `video_colpali_smol500_mv_frame` - ColPali frame extraction
- `video_colqwen_omni_mv_chunk_30s` - ColQwen multi-vector chunks
- `video_videoprism_base_mv_chunk_30s` - VideoPrism base model chunks
- `video_videoprism_lvt_base_sv_chunk_6s` - VideoPrism LVT single-vector

#### Search & Query
```bash
# Test search capabilities
JAX_PLATFORM_NAME=cpu uv run python tests/comprehensive_video_query_test_v2.py \
    --profiles video_colpali_smol500_mv_frame video_colqwen_omni_mv_chunk_30s \
    --test-multiple-strategies
```

#### Evaluation & Analytics
```bash
# Run experiments with Phoenix visualization
uv run python scripts/run_experiments_with_visualization.py \
    --csv-path data/testset/evaluation/video_search_queries.csv \
    --dataset-name golden_eval_v1 \
    --profiles video_colpali_smol500_mv_frame

# Launch Phoenix dashboard
uv run streamlit run scripts/phoenix_dashboard_standalone.py --server.port 8501
```

## 📁 Project Structure

```
cogniverse/
├── src/
│   ├── app/
│   │   ├── ingestion/     # Video processing pipeline
│   │   ├── routing/        # Query routing system
│   │   └── agents/         # Multi-agent orchestration
│   ├── backends/
│   │   └── vespa/          # Vector database backend
│   ├── evaluation/         # Evaluation framework
│   └── common/             # Shared utilities
├── scripts/                # Operational scripts
├── tests/                  # Test suite
└── configs/                # Configuration files
```

## 🔧 Configuration

### Environment Variables
```bash
export JAX_PLATFORM_NAME=cpu  # For VideoPrism on CPU
export VESPA_HOST=localhost
export VESPA_PORT=8080
```

### Profile Configuration
Profiles are defined in `configs/config.json` under `video_processing_profiles`. Each profile specifies:
- Frame extraction strategy
- Embedding models
- Chunk sizes
- Processing parameters

## 📊 Testing

```bash
# Unit tests
uv run pytest tests/

# Integration tests  
uv run pytest tests/ingestion/ -m integration

# Comprehensive routing test
uv run python tests/routing/integration/test_router_comprehensive.py
```

## 🛠️ Development

### Running in Background
```bash
nohup uv run python scripts/run_ingestion.py \
    --video_dir /path/to/videos \
    --backend vespa \
    > outputs/logs/ingestion_$(date +%s).log 2>&1 &
```

### Monitoring Logs
```bash
tail -f outputs/logs/*.log
```

### Common Issues

**Dimension Mismatch Errors**
- Check embedding dimensions: 768 (base), 1024 (large), 1152 (patch-based)
- Verify format: hex strings for binary, floats for dense

**Connection Errors**
- Reduce batch size if "Connection aborted"
- Check Vespa status: `curl http://localhost:8080/ApplicationStatus`

**HTTP 400 Errors**
- Schema/data format mismatch
- Use pyvespa `feed_iterable` not raw HTTP

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [Evaluation Framework](docs/evaluation.md)
- [Deployment Guide](docs/deployment.md)
- Module-specific docs in each `src/*/README.md`

## 🤝 Contributing

1. Follow existing code patterns and conventions
2. Run tests before committing
3. Update module README if changing functionality
4. Use `uv run` for all Python commands

## 📄 License

[License information here]