# Files and Folders for First Git Commit

## Essential Configuration Files

### Root Configuration
- `.gitignore` - Git ignore file
- `pyproject.toml` - Python project configuration
- `requirements.txt` - Python dependencies
- `uv.lock` - UV package lock file
- `README.md` - Main project documentation
- `CLAUDE.md` - Claude AI instructions
- `QUICKSTART.md` - Quick start guide

### Configuration Directory
- `configs/config.json` - Main configuration
- `configs/examples/config.example.json` - Example configuration
- `configs/schemas/video_frame_schema.json` - Vespa schema

## Source Code

### Core Agents
- `src/agents/composing_agents_main.py` - Main orchestrator agent
- `src/agents/video_agent_server.py` - Video search agent
- `src/agents/text_agent_server.py` - Text search agent

### Tools and Utilities
- `src/tools/config.py` - Configuration management
- `src/tools/video_player_tool.py` - Video player tool
- `src/tools/a2a_utils.py` - Agent-to-agent communication
- `src/tools/temporal_extractor.py` - Temporal extraction

### Processing Pipeline
- `src/processing/video_processor.py` - Video processing
- `src/processing/byaldi_indexer.py` - Byaldi indexing
- `src/processing/pipeline_runner.py` - Pipeline orchestration
- `src/processing/pipeline_steps/` - All pipeline step modules
  - `audio_transcriber.py`
  - `description_generator.py`
  - `embedding_generator.py`
  - `keyframe_extractor.py`

### Vespa Integration
- `src/processing/vespa/vespa_indexer.py` - Vespa indexing
- `src/processing/vespa/vespa_search_client.py` - Search client
- `src/processing/vespa/vespa_schema_manager.py` - Schema management
- `src/processing/vespa/json_schema_parser.py` - Schema parsing

### Utilities
- `src/utils/output_manager.py` - Output directory management
- `src/utils/__init__.py`

### Optimization
- `src/optimizer/schemas.py` - Optimization schemas
- `src/optimizer/router_optimizer.py` - Router optimization
- `src/optimizer/orchestrator.py` - Optimization orchestration
- `src/optimizer/providers/` - Provider implementations
  - `base_provider.py`
  - `modal_provider.py`
  - `local_provider.py`

### Inference
- `src/inference/unified_router.py` - Unified routing

## Scripts

### Setup and Running
- `scripts/setup_ollama.py` - Ollama setup
- `scripts/setup_system.py` - System setup
- `scripts/run_servers.sh` - Server startup
- `scripts/stop_servers.sh` - Server shutdown
- `scripts/start_vespa.sh` - Vespa startup
- `scripts/stop_vespa.sh` - Vespa shutdown

### Processing and Ingestion
- `scripts/run_ingestion.py` - Video ingestion
- `scripts/deploy_all_schemas.py` - Deploy all Vespa schemas
- `scripts/setup_output_directories.py` - Output directory setup

### Optimization
- `scripts/run_optimization.py` - Run optimization
- `scripts/run_orchestrator.py` - Run orchestrator

## Tests

### Test Utilities
- `tests/__init__.py`
- `tests/test_utils.py` - Test formatting utilities

### Component Tests
- `tests/test_colpali_search.py` - ColPali search tests
- `tests/test_document_similarity.py` - Document similarity tests
- `tests/test_search_client.py` - Search client tests
- `tests/test_system.py` - System tests
- `tests/test_video_player.py` - Video player tests
- `tests/comprehensive_video_query_test.py` - Comprehensive query tests

### Routing Tests
- `tests/routing/test_routing_baseline.py`
- `tests/routing/test_combined_routing.py`

## Documentation

- `docs/` - All documentation files
  - `processing/frame_boundary_algorithm.md`
  - `git_first_commit_files.md` (this file)

## Data Directories (structure only, not content)

Create these directories but don't commit their contents:
- `data/` - Data root
- `data/videos/` - Video files
- `data/videos/processed/` - Processed data
- `data/vespa/` - Vespa data
- `outputs/` - All output files
- `logs/` - Log files

## Files to Exclude from First Commit

### Temporary and Generated Files
- `*.pyc`
- `__pycache__/`
- `.DS_Store`
- `*.log`
- `.env`

### Data Files
- Video files (`*.mp4`, `*.avi`, etc.)
- Processed data files
- Index files
- Model files

### IDE and Editor Files
- `.vscode/`
- `.idea/`
- `.cursorindexingignore`

### Test and Debug Files
- `debug_*.py`
- `test_*.py` (except those in tests/)
- `check_*.py`

### Large Directories
- `.byaldi/` - Byaldi index data
- `.specstory/` - Spec story data
- `archive/` - Archived files
- `optimization_results/` - Optimization outputs
- `test_results/` - Test result outputs
- `video_rag_agent/` - RAG agent data

## Recommended .gitignore Content

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.env
.venv
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
.cursorindexingignore

# OS
.DS_Store
Thumbs.db

# Project specific
data/videos/**/*.mp4
data/videos/**/*.avi
data/videos/**/*.mov
data/videos/processed/
data/vespa/
.byaldi/
.specstory/
archive/
*.log
logs/
test_results/
optimization_results/
outputs/
video_rag_agent/

# Debug files
debug_*.py
check_*.py
test_vespa_*.py
test_float_*.py

# Model files
*.bin
*.pth
*.onnx
*.safetensors

# Large data files
*.h5
*.hdf5
*.pkl
*.pickle
*.npy
*.npz

# Temporary files
*.tmp
*.temp
.cache/
temp/
```

## Git Commands for First Commit

```bash
# Initialize git repository
git init

# Create .gitignore with the content above
# (copy the .gitignore content to .gitignore file)

# Add all files according to this list
git add .gitignore
git add pyproject.toml requirements.txt uv.lock
git add README.md CLAUDE.md QUICKSTART.md
git add configs/
git add src/
git add scripts/
git add tests/
git add docs/

# Create empty directories
mkdir -p data/videos/processed data/vespa outputs logs

# Commit
git commit -m "Initial commit: Multi-Agent RAG System

- Core agent architecture with composing, video, and text agents
- Video processing pipeline with keyframe extraction and transcription
- Vespa integration for scalable vector search
- ColPali-based visual search capabilities
- Comprehensive test suite
- Configuration management system
- Documentation and setup scripts"
```