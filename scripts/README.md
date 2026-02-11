# Operational Scripts Documentation

**Last Updated**: 2025-11-13
**Architecture**: 10-Package Cogniverse System
**Audience**: Operators, DevOps Engineers, ML Engineers

Comprehensive automation scripts for the Cogniverse multi-modal agentic system. All scripts support the modular 10-package architecture with proper imports and multi-tenancy.

---

## Table of Contents

1. [Package Architecture](#package-architecture)
2. [Ingestion Scripts](#ingestion-scripts)
3. [Search & Query Scripts](#search--query-scripts)
4. [Optimization Scripts](#optimization-scripts)
5. [Evaluation Scripts](#evaluation-scripts)
6. [Dataset Management Scripts](#dataset-management-scripts)
7. [Deployment Scripts](#deployment-scripts)
8. [Monitoring & Analysis Scripts](#monitoring--analysis-scripts)
9. [System Setup Scripts](#system-setup-scripts)
10. [Common Workflows](#common-workflows)

---

## Package Architecture

Cogniverse uses a 10-package modular architecture:

### Foundation Layer (Blue)
- **cogniverse-sdk** - Core SDK and shared utilities
- **cogniverse-foundation** - Configuration management, telemetry foundation

### Core Layer (Pink)
- **cogniverse-core** - Multi-agent orchestration, context management
- **cogniverse-evaluation** - Evaluation framework and metrics
- **cogniverse-telemetry-phoenix** - Phoenix/Arize integration for observability

### Implementation Layer (Yellow/Green)
- **cogniverse-agents** - Agent implementations, routing, optimization
- **cogniverse-vespa** - Vespa backend integration
- **cogniverse-synthetic** - Synthetic data generation for DSPy training

### Application Layer (Light Blue/Purple)
- **cogniverse-runtime** - FastAPI server and API endpoints
- **cogniverse-dashboard** - Streamlit dashboards and UI components

**All scripts use proper package imports**:
```python
from cogniverse_agents.routing import ModalityOptimizer
from cogniverse_synthetic import SyntheticDataService
from cogniverse_vespa import VespaBackend
from cogniverse_foundation.config.utils import get_config
```

---

## Ingestion Scripts

### Multi-Modal Content Ingestion

#### `run_ingestion.py` - Video Ingestion Pipeline
**Async video processing with ColPali embeddings and multi-profile support**

```bash
# Basic video ingestion (uses active profile)
uv run python scripts/run_ingestion.py \
  --tenant-id default_tenant \
  --video_dir data/videos \
  --backend vespa

# Multiple profiles (parallel processing)
uv run python scripts/run_ingestion.py \
  --tenant-id acme_corp \
  --profile video_colpali_smol500_mv_frame video_colpali_base_sv_segment \
  --max-concurrent 5

# Test mode (limited frames)
uv run python scripts/run_ingestion.py \
  --video_dir data/testset/evaluation/sample_videos \
  --test-mode \
  --max-frames 10 \
  --debug

# Advanced: custom output directory
uv run python scripts/run_ingestion.py \
  --video_dir data/videos \
  --output_dir outputs/processed \
  --max-frames 500 \
  --max-concurrent 3
```

**Features:**
- Async concurrent video processing
- ColPali multi-vector embeddings (1024 patches × 128 dims)
- Frame/segment extraction strategies
- Whisper transcription integration
- Multi-profile support (frame-based, segment-based)
- Vespa/Byaldi backend support
- Builder pattern for clean initialization

**Imports:**
- `cogniverse_foundation.config.utils` - Configuration management
- `src.app.ingestion.pipeline_builder` - Pipeline builders
- `src.common.config_utils` - Utilities

---

#### `ingest_documents.py` - Document Ingestion (Dual Strategy)
**PDF ingestion with visual (ColPali) and text (semantic) strategies**

```bash
# Dual-strategy document ingestion
uv run python scripts/ingest_documents.py \
  --document_dir data/documents \
  --vespa_endpoint http://localhost:8080 \
  --colpali_model vidore/colsmol-500m \
  --app_name documentsearch

# Custom model
uv run python scripts/ingest_documents.py \
  --document_dir data/research_papers \
  --colpali_model vidore/colpali-v1.2 \
  --vespa_endpoint http://vespa.cogniverse.svc.cluster.local:8080
```

**Features:**
- **Visual Strategy**: ColPali page-as-image embeddings [1024, 128]
- **Text Strategy**: PyPDF2 extraction + sentence-transformers/all-mpnet-base-v2
- Creates two Vespa entries per document for comparison
- Page-level indexing with metadata
- Automatic PDF to image conversion (pdf2image)

**Imports:**
- `src.common.models.model_loaders` - ColPali model loading
- `PyPDF2` - Text extraction
- `pdf2image` - Page rendering

---

#### `ingest_audio.py` - Audio Ingestion
**Audio ingestion with Whisper transcription and dual embeddings**

```bash
# Audio ingestion with transcription
uv run python scripts/ingest_audio.py \
  --audio_dir data/audio \
  --vespa_endpoint http://localhost:8080 \
  --whisper_model base \
  --app_name audiosearch

# Large files: use larger Whisper model
uv run python scripts/ingest_audio.py \
  --audio_dir data/podcasts \
  --whisper_model large-v3 \
  --vespa_endpoint http://localhost:8080
```

**Features:**
- Whisper transcription (base/small/medium/large/large-v3)
- Acoustic embeddings (audio features)
- Semantic embeddings (text from transcript)
- Language detection
- Speaker diarization support
- Formats: MP3, WAV, M4A, OGG, FLAC

**Imports:**
- `src.app.ingestion.processors.audio_transcriber` - Whisper transcription
- `src.app.ingestion.processors.audio_embedding_generator` - Embeddings

---

#### `ingest_images.py` - Image Ingestion
**Image ingestion with ColPali visual embeddings**

```bash
# Image ingestion
uv run python scripts/ingest_images.py \
  --image_dir data/images \
  --vespa_endpoint http://localhost:8080 \
  --colpali_model vidore/colsmol-500m \
  --app_name imagesearch

# High-quality model
uv run python scripts/ingest_images.py \
  --image_dir data/product_images \
  --colpali_model vidore/colpali-v1.2
```

**Features:**
- ColPali multi-vector embeddings [1024, 128]
- Supports JPG, PNG, BMP, GIF
- Object/scene detection metadata
- Batch processing with error handling

**Imports:**
- `src.common.models.model_loaders` - ColPali model loading
- `PIL.Image` - Image processing

---

## Search & Query Scripts

### `demo_routing_unified.py` - Interactive Query Testing
**Test unified routing with real-time agent execution**

```bash
# Interactive routing demo
uv run python scripts/demo_routing_unified.py \
  --tenant-id default

# With custom config
uv run python scripts/demo_routing_unified.py \
  --tenant-id acme_corp \
  --config config/custom_routing.json
```

**Features:**
- Real-time query routing
- Modality detection
- Agent selection and execution
- Phoenix tracing integration
- Multi-modal result aggregation

---

## Optimization Scripts

### Core Optimization Workflows

#### `run_module_optimization.py` - DSPy Module Optimization
**Optimize routing/workflow modules with automatic DSPy optimizer selection**

```bash
# Optimize modality routing (all modalities: video, image, audio, document, text)
uv run python scripts/run_module_optimization.py \
  --module modality \
  --tenant-id default \
  --lookback-hours 24 \
  --min-confidence 0.7 \
  --output results/modality_optimization.json

# Optimize cross-modal fusion
uv run python scripts/run_module_optimization.py \
  --module cross_modal \
  --tenant-id default \
  --use-synthetic-data \
  --output results/cross_modal.json

# Optimize entity-based routing
uv run python scripts/run_module_optimization.py \
  --module routing \
  --tenant-id default \
  --lookback-hours 48 \
  --max-iterations 100

# Optimize ALL modules
uv run python scripts/run_module_optimization.py \
  --module all \
  --tenant-id default \
  --use-synthetic-data \
  --lookback-hours 24 \
  --force-training \
  --output results/full_optimization.json

# With Phoenix dataset
uv run python scripts/run_module_optimization.py \
  --module modality \
  --tenant-id acme_corp \
  --dataset-name golden_queries_v2 \
  --max-iterations 200
```

**Modules Supported:**
- **modality**: Per-modality routing (VIDEO, IMAGE, AUDIO, DOCUMENT, TEXT)
- **cross_modal**: Cross-modal fusion decisions
- **routing**: Advanced entity-based routing
- **workflow**: Workflow orchestration (pending implementation)
- **unified**: Unified routing + workflow (pending implementation)

**DSPy Optimizers (Automatic Selection):**
- **BootstrapFewShot**: Fast, few-shot learning (cold start)
- **MIPROv2**: Multi-stage instruction/prompt optimization
- **SIMBA**: Query enhancement and reformulation
- **GEPA**: Experience-Guided Policy Adaptation (advanced)

**Features:**
- XGBoost meta-learning for automatic training decisions
- Synthetic data generation via `cogniverse-synthetic`
- Phoenix trace collection with confidence filtering
- Progressive training strategies (synthetic → hybrid → pure real)
- Multi-tenancy support
- JSON output with detailed metrics

**Imports:**
- `cogniverse_agents.routing.modality_optimizer` - Per-modality optimization
- `cogniverse_agents.routing.cross_modal_optimizer` - Cross-modal optimization
- `cogniverse_agents.routing.advanced_optimizer` - Entity-based routing
- `cogniverse_synthetic` - Synthetic data generation

---

#### `auto_optimization_trigger.py` - Auto-Optimization Trigger
**Conditional optimization trigger for Argo CronWorkflow**

```bash
# Check conditions and trigger optimization
uv run python scripts/auto_optimization_trigger.py \
  --tenant-id default \
  --module routing \
  --phoenix-endpoint http://localhost:6006

# Manual trigger (bypass conditions)
uv run python scripts/auto_optimization_trigger.py \
  --tenant-id default \
  --module modality \
  --force \
  --lookback-hours 48
```

**Conditions Checked:**
1. `enable_auto_optimization = True` in routing config
2. Time interval met (since last optimization)
3. Sufficient Phoenix traces (>= `min_samples_for_optimization`)

**Features:**
- Config-driven optimization scheduling
- Phoenix trace count validation
- Marker file tracking (prevents duplicate runs)
- Exit codes: 0=triggered, 1=skipped, 2=error

**Imports:**
- `cogniverse_foundation.config.utils` - Config management
- `cogniverse_foundation.telemetry.manager` - Telemetry provider

---

#### `optimize_system.py` - System-Wide Optimization
**Comprehensive system optimization (all components)**

```bash
# Full system optimization
uv run python scripts/optimize_system.py \
  --tenant-id default \
  --components all

# Specific components
uv run python scripts/optimize_system.py \
  --tenant-id acme_corp \
  --components routing modality embeddings
```

---

## Evaluation Scripts

### Dataset Creation & Management

#### `bootstrap_dataset_from_traces.py` - Extract Datasets from Traces
**Create evaluation datasets from high-confidence Phoenix traces**

```bash
# Bootstrap dataset from last 24 hours (high-confidence results)
uv run python scripts/bootstrap_dataset_from_traces.py \
  --hours 24 \
  --min-score 0.8 \
  --dataset-name bootstrap_v1

# Export to CSV for manual review
uv run python scripts/bootstrap_dataset_from_traces.py \
  --hours 48 \
  --min-score 0.7 \
  --output-csv datasets/review.csv \
  --dedupe

# Filter by profile
uv run python scripts/bootstrap_dataset_from_traces.py \
  --filter-profile video_colpali_smol500_mv_frame \
  --output-json datasets/frame_based_queries.json \
  --min-results 3 \
  --max-results 10
```

**Features:**
- Extracts successful searches from Phoenix
- High-confidence results as ground truth
- Deduplication of queries
- Profile/strategy filtering
- Phoenix dataset creation
- CSV/JSON export

**Imports:**
- `src.evaluation.data.datasets` - DatasetManager
- `src.evaluation.data.traces` - TraceManager

---

#### `create_golden_dataset_from_traces.py` - Create Golden Datasets
**Curated golden datasets from traces with quality thresholds**

```bash
# Create golden dataset
uv run python scripts/create_golden_dataset_from_traces.py \
  --hours 72 \
  --min-score 0.9 \
  --min-results 5 \
  --dataset-name golden_queries_v2 \
  --dedupe \
  --manual-review

# Export for approval
uv run python scripts/create_golden_dataset_from_traces.py \
  --output-csv datasets/golden_candidates.csv \
  --min-score 0.85
```

---

#### `generate_dataset_from_videos.py` - Video Metadata Datasets
**Auto-generate evaluation queries from video content and metadata**

```bash
# Generate queries from video metadata
uv run python scripts/generate_dataset_from_videos.py \
  --video_dir data/videos \
  --output datasets/video_queries.json \
  --count 100
```

---

#### `interactive_dataset_builder.py` - Manual Dataset Building
**Interactive CLI for manual dataset curation**

```bash
# Interactive dataset builder
uv run python scripts/interactive_dataset_builder.py \
  --dataset-name curated_v1
```

---

### Evaluation Execution

#### `run_experiments_with_visualization.py` - Run Evaluation Experiments
**Run evaluation experiments with Phoenix tracking and visualization**

```bash
# Run experiments with dataset
uv run python scripts/run_experiments_with_visualization.py \
  --dataset golden_queries_v2 \
  --tenant-id default \
  --profiles video_colpali_smol500_mv_frame video_colpali_base_sv_segment

# Compare strategies
uv run python scripts/run_experiments_with_visualization.py \
  --dataset bootstrap_v1 \
  --strategies maxsim_fusion hybrid_fusion \
  --output results/strategy_comparison.html
```

**Note**: Use `cogniverse_evaluation` package for comprehensive evaluation framework.

---

#### `evaluate_comprehensive_test_spans.py` - Comprehensive Span Evaluation
**Evaluate Phoenix spans across multiple dimensions**

```bash
# Evaluate recent spans
uv run python scripts/evaluate_comprehensive_test_spans.py \
  --tenant-id default \
  --hours 24 \
  --output results/span_evaluation.json

# With detailed metrics
uv run python scripts/evaluate_comprehensive_test_spans.py \
  --tenant-id acme_corp \
  --hours 48 \
  --metrics accuracy latency cost \
  --breakdown-by modality profile
```

---

## Dataset Management Scripts

#### `manage_datasets.py` - Dataset CRUD Operations
**Create, list, update, delete Phoenix datasets**

```bash
# List datasets
uv run python scripts/manage_datasets.py list --tenant-id default

# Create dataset
uv run python scripts/manage_datasets.py create \
  --name test_queries_v1 \
  --description "Test queries for video search"

# Delete dataset
uv run python scripts/manage_datasets.py delete --name old_dataset
```

---

#### `manage_golden_datasets.py` - Golden Dataset Management
**Manage curated golden evaluation datasets**

```bash
# Create golden dataset
uv run python scripts/manage_golden_datasets.py create \
  --name golden_v3 \
  --source-csv datasets/curated.csv

# Export golden dataset
uv run python scripts/manage_golden_datasets.py export \
  --name golden_v3 \
  --output datasets/golden_v3.json
```

---

## Deployment Scripts

#### `deploy_production.py` - Production API Deployment
**Deploy production API to Modal (after optimization)**

```bash
# Deploy production API (uses existing artifacts)
uv run python scripts/deploy_production.py

# With custom Modal config
uv run python scripts/deploy_production.py \
  --modal-workspace acme_corp \
  --environment production
```

**Features:**
- Deploys optimized models to Modal
- Health check validation
- API usage examples
- Artifact verification

---

#### `run_optimization.py` - Full Optimization + Deployment Workflow
**Complete end-to-end optimization and deployment pipeline**

```bash
# Full workflow (optimization + deployment + testing)
uv run python scripts/run_optimization.py

# With custom config
uv run python scripts/run_optimization.py \
  --config config/optimization.json

# Skip specific steps
uv run python scripts/run_optimization.py \
  --skip-upload \
  --skip-test
```

**Workflow Steps:**
1. Run orchestrator optimization
2. Upload artifacts to Modal volume
3. Deploy production API
4. Test deployed system

---

#### `deploy_all_schemas.py` - Deploy Vespa Schemas
**Deploy all Vespa schemas for multi-modal search**

```bash
# Deploy all schemas
uv run python scripts/deploy_all_schemas.py \
  --vespa-endpoint http://localhost:8080

# Deploy specific schemas
uv run python scripts/deploy_all_schemas.py \
  --schemas video_colpali_mv_frame document_visual audio_content \
  --vespa-endpoint http://vespa.svc.cluster.local:8080
```

**Schemas:**
- `video_colpali_mv_frame` - Frame-based video search
- `video_colpali_sv_segment` - Segment-based video search
- `document_visual` - Visual document strategy (ColPali)
- `document_text` - Text document strategy (semantic)
- `image_content` - Image search
- `audio_content` - Audio search with transcripts
- `dataframe_content` - Structured data search

**Imports:**
- `cogniverse_vespa.schema_deployer` - Schema deployment

---

#### `deploy_kubernetes.sh` - Kubernetes Deployment
**Deploy full Cogniverse stack to Kubernetes**

```bash
# Deploy to Kubernetes
bash scripts/deploy_kubernetes.sh --namespace cogniverse --environment production

# With custom values
bash scripts/deploy_kubernetes.sh \
  --namespace acme-corp \
  --environment staging \
  --values helm/custom-values.yaml
```

---

#### `deploy_k3s.sh` - K3s Lightweight Deployment
**Deploy to K3s (lightweight Kubernetes)**

```bash
# Deploy to K3s
bash scripts/deploy_k3s.sh --namespace cogniverse
```

---

## Monitoring & Analysis Scripts

#### `analyze_traces.py` - Trace Analysis
**Analyze Phoenix traces for patterns and insights**

```bash
# Analyze recent traces
uv run python scripts/analyze_traces.py \
  --tenant-id default \
  --hours 24 \
  --output results/trace_analysis.html

# Deep analysis with breakdown
uv run python scripts/analyze_traces.py \
  --tenant-id acme_corp \
  --hours 72 \
  --breakdown modality profile strategy \
  --min-confidence 0.7
```

---

#### `phoenix_dashboard.py` - Phoenix Monitoring Dashboard
**Streamlit dashboard for Phoenix metrics and traces**

```bash
# Launch Phoenix dashboard
uv run streamlit run scripts/phoenix_dashboard.py -- \
  --tenant-id default \
  --phoenix-endpoint http://localhost:6006

# Multi-tenant dashboard
uv run streamlit run scripts/phoenix_dashboard.py -- \
  --tenants default acme_corp enterprise_client
```

**Imports:**
- `cogniverse_dashboard.components` - Dashboard UI components
- `cogniverse_telemetry_phoenix` - Phoenix integration

---

#### `export_vespa_embeddings.py` - Export Embeddings for Analysis
**Export embeddings from Vespa for dimensionality reduction/visualization**

```bash
# Export embeddings
uv run python scripts/export_vespa_embeddings.py \
  --schema video_colpali_mv_frame \
  --output embeddings/video_embeddings.npz \
  --limit 10000

# With metadata
uv run python scripts/export_vespa_embeddings.py \
  --schema document_visual \
  --output embeddings/doc_embeddings.npz \
  --include-metadata \
  --format parquet
```

---

#### `embedding_atlas_tab.py` - Embedding Visualization
**Interactive embedding atlas with UMAP/t-SNE visualization**

```bash
# Launch embedding atlas
uv run streamlit run scripts/embedding_atlas_tab.py -- \
  --embeddings embeddings/video_embeddings.npz \
  --reduction umap

# With clustering
uv run streamlit run scripts/embedding_atlas_tab.py -- \
  --embeddings embeddings/doc_embeddings.npz \
  --reduction tsne \
  --clustering kmeans \
  --n-clusters 20
```

---

## System Setup Scripts

#### `setup_system.py` - System Initialization
**Initialize Cogniverse system (directories, configs, dependencies)**

```bash
# Initialize system
uv run python scripts/setup_system.py

# With custom config
uv run python scripts/setup_system.py \
  --config config/system.yaml \
  --force-recreate
```

---

#### `setup_evaluation.sh` - Evaluation Framework Setup
**Set up evaluation framework and datasets**

```bash
# Setup evaluation
bash scripts/setup_evaluation.sh

# With sample datasets
bash scripts/setup_evaluation.sh --include-samples
```

---

#### `start_phoenix.py` - Start Phoenix Server
**Launch Phoenix observability server**

```bash
# Start Phoenix
uv run python scripts/start_phoenix.py

# With custom port
uv run python scripts/start_phoenix.py --port 6007 --host 0.0.0.0

# With persistence
uv run python scripts/start_phoenix.py --storage-dir data/phoenix
```

**Imports:**
- `cogniverse_telemetry_phoenix.server` - Phoenix server launcher

---

#### `run_servers.sh` - Start All Services
**Launch all required services (Vespa, Phoenix, API server)**

```bash
# Start all services
bash scripts/run_servers.sh

# Production mode
bash scripts/run_servers.sh --environment production --detach
```

---

## Common Workflows

### First-Time Setup

```bash
# 1. Initialize system
uv run python scripts/setup_system.py

# 2. Start services
bash scripts/run_servers.sh

# 3. Deploy Vespa schemas
uv run python scripts/deploy_all_schemas.py

# 4. Ingest sample data
uv run python scripts/run_ingestion.py \
  --video_dir data/testset/evaluation/sample_videos \
  --test-mode
```

### Production Ingestion Workflow

```bash
# 1. Ingest videos
uv run python scripts/run_ingestion.py \
  --tenant-id acme_corp \
  --video_dir /mnt/videos \
  --profile video_colpali_smol500_mv_frame \
  --max-concurrent 10

# 2. Ingest documents
uv run python scripts/ingest_documents.py \
  --document_dir /mnt/documents \
  --colpali_model vidore/colsmol-500m

# 3. Ingest audio
uv run python scripts/ingest_audio.py \
  --audio_dir /mnt/audio \
  --whisper_model large-v3

# 4. Ingest images
uv run python scripts/ingest_images.py \
  --image_dir /mnt/images
```

### Optimization & Deployment Workflow

```bash
# 1. Optimize modules with synthetic data
uv run python scripts/run_module_optimization.py \
  --module all \
  --tenant-id acme_corp \
  --use-synthetic-data \
  --lookback-hours 72 \
  --output results/optimization_$(date +%Y%m%d).json

# 2. Review optimization results
cat results/optimization_*.json | jq '.summary'

# 3. Deploy to production
uv run python scripts/deploy_production.py

# 4. Monitor performance
uv run streamlit run scripts/phoenix_dashboard.py
```

### Evaluation Workflow

```bash
# 1. Create dataset from traces
uv run python scripts/bootstrap_dataset_from_traces.py \
  --hours 48 \
  --min-score 0.8 \
  --dataset-name eval_$(date +%Y%m%d) \
  --dedupe

# 2. Run experiments
uv run python scripts/run_experiments_with_visualization.py \
  --dataset eval_$(date +%Y%m%d) \
  --profiles video_colpali_smol500_mv_frame video_colpali_base_sv_segment \
  --output results/eval_$(date +%Y%m%d).html

# 3. Analyze results
uv run python scripts/analyze_traces.py \
  --hours 24 \
  --breakdown modality profile
```

### Automated Background Optimization (Kubernetes/Argo)

```bash
# Deploy auto-optimization CronWorkflow
kubectl apply -f workflows/auto-optimization-multi-tenant.yaml

# Monitor optimization workflows
kubectl get workflows -n cogniverse -l optimization-type=auto

# Check specific tenant
kubectl get workflows -n cogniverse -l tenant-id=acme_corp

# View logs
kubectl logs -n cogniverse -l workflows.argoproj.io/workflow=auto-opt-default-abc123
```

---

## Environment Variables

```bash
# LLM API Keys (for DSPy teacher model and auto-annotator)
export ROUTER_OPTIMIZER_TEACHER_KEY="your-api-key"  # Works with any LiteLLM-supported provider
export ANNOTATION_API_KEY="your-api-key"            # For LLM auto-annotator (optional)

# Service Endpoints
export VESPA_URL="http://localhost:8080"
export PHOENIX_ENDPOINT="http://localhost:6006"

# Configuration
export TENANT_ID="default"
export CONFIG_STORE_URL="redis://localhost:6379"

# Modal Deployment
export MODAL_TOKEN_ID="..."
export MODAL_TOKEN_SECRET="..."
```

---

## Script Execution Format

All scripts support `uv run` for consistent dependency management:

```bash
# Standard format
uv run python scripts/<script>.py [OPTIONS]

# Examples
uv run python scripts/run_ingestion.py --help
uv run python scripts/run_module_optimization.py --module modality --tenant-id default
uv run streamlit run scripts/phoenix_dashboard.py
```

---

## Package Import Patterns

### Typical Script Imports

```python
# Foundation Layer
from cogniverse_sdk import BaseAgent, AgentContext
from cogniverse_foundation.config.utils import get_config, create_default_config_manager
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

# Core Layer
from cogniverse_core.orchestration import OrchestrationEngine
from cogniverse_evaluation.metrics import EvaluationMetrics
from cogniverse_telemetry_phoenix import PhoenixProvider

# Implementation Layer
from cogniverse_agents.routing import ModalityOptimizer, CrossModalOptimizer
from cogniverse_vespa import VespaBackend
from cogniverse_synthetic import SyntheticDataService, SyntheticDataRequest

# Application Layer
from cogniverse_runtime.api import create_app
from cogniverse_dashboard.components import MetricsViewer
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure all packages installed
uv sync

# Or install individually
uv pip install -e libs/agents -e libs/synthetic -e libs/vespa
```

**2. Vespa Connection Failed**
```bash
# Check Vespa status
curl http://localhost:8080/state/v1/health

# Start Vespa
bash scripts/start_vespa.sh
```

**3. Phoenix Not Running**
```bash
# Start Phoenix
uv run python scripts/start_phoenix.py --port 6006

# Or use shell script
bash scripts/start_phoenix.sh
```

**4. Optimization Failing**
- Check Phoenix traces: Ensure sufficient data (>100 traces)
- Verify LLM API keys: Set `ROUTER_OPTIMIZER_TEACHER_KEY`
- Review synthetic data generation: Use `--use-synthetic-data` for cold start

**5. Ingestion Slow**
- Increase `--max-concurrent` for parallel processing
- Use smaller ColPali models: `vidore/colsmol-500m` vs `vidore/colpali-v1.2`
- Reduce `--max-frames` for testing

---

## Script Categories Summary

| Category | Count | Key Scripts |
|----------|-------|-------------|
| **Ingestion** | 5 | `run_ingestion.py`, `ingest_documents.py`, `ingest_audio.py`, `ingest_images.py` |
| **Optimization** | 4 | `run_module_optimization.py`, `auto_optimization_trigger.py`, `optimize_system.py` |
| **Evaluation** | 6 | `bootstrap_dataset_from_traces.py`, `run_experiments_with_visualization.py`, `evaluate_comprehensive_test_spans.py` |
| **Dataset Management** | 5 | `manage_datasets.py`, `manage_golden_datasets.py`, `create_golden_dataset_from_traces.py` |
| **Deployment** | 6 | `deploy_production.py`, `deploy_all_schemas.py`, `deploy_kubernetes.sh`, `run_optimization.py` |
| **Monitoring** | 5 | `analyze_traces.py`, `phoenix_dashboard.py`, `export_vespa_embeddings.py`, `embedding_atlas_tab.py` |
| **Setup** | 5 | `setup_system.py`, `start_phoenix.py`, `run_servers.sh`, `setup_evaluation.sh` |
| **Total** | **36+** | Core operational scripts |

---

## Related Documentation

- **Architecture**: `docs/architecture/10-package-architecture.md`
- **Synthetic Data**: `docs/synthetic-data-generation.md`
- **Optimization**: `docs/modules/optimization.md`
- **Evaluation**: `docs/EVALUATION_FRAMEWORK.md`
- **Auto-Optimization**: `workflows/AUTO_OPTIMIZATION_README.md`
- **Package READMEs**: `libs/*/README.md`

---

## Development

### Adding New Scripts

1. Follow naming convention: `action_target.py`
2. Use proper package imports from 10-package architecture
3. Support `--tenant-id` for multi-tenancy
4. Include comprehensive `--help` documentation
5. Use `uv run python` for execution
6. Add error handling and logging
7. Output JSON results when appropriate
8. Update this README

### Script Template

```python
#!/usr/bin/env python3
"""
Script Description

What it does and when to use it.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Proper package imports
from cogniverse_foundation.config.utils import get_config
from cogniverse_agents.routing import ModalityOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function with error handling"""
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--tenant-id", default="default", help="Tenant identifier")
    # Add more arguments...

    args = parser.parse_args()

    try:
        # Implementation
        logger.info("Starting script...")
        # ... script logic ...
        logger.info("✅ Script completed successfully")
        return 0
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

**Last Updated**: 2025-11-13
**Maintainer**: Cogniverse DevOps Team
**Support**: See individual script `--help` for detailed usage
