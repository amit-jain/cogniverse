# Getting Started with Cogniverse

**Architecture**: 10-Package Layered System
**Last Updated**: 2025-11-13

Welcome to Cogniverse! This guide will help you get your sophisticated multi-agent video search system up and running quickly.

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.12+** (Required for compatibility)
- **16GB+ RAM** (Required for local AI models)
- **5GB+ disk space** (For Llama 3.1 model)
- **CUDA-capable GPU** (Recommended for faster video processing)

### 2. Installation

The Python 3.12 compatibility issues have been resolved. Simply run:

```bash
# Install dependencies
pip install -r requirements.txt

# Or using uv (faster)
uv pip install -r requirements.txt
```

### 3. Setup Local AI

Setup the local AI coordinator (Llama 3.1):

```bash
# Install and configure Ollama + Llama 3.1
python scripts/setup_ollama.py
```

This automatically:
- Installs Ollama (local AI runtime)
- Downloads Llama 3.1 model 
- Starts the AI server
- Tests functionality

Optional: Copy and customize configuration:

```bash
cp configs/examples/config.example.json configs/config.json
# Edit config.json for custom settings
```

### 4. Set Up the Video Database

Run the setup script to create the video index:

```bash
python scripts/setup_system.py
```

This will:
- Create necessary directories
- Generate a sample test video (if imageio is available)
- Process videos and create the Byaldi search index
- Validate all components

### 5. Test the System (Optional)

Run the comprehensive test suite:

```bash
python scripts/test_system.py
```

This will validate all components and tell you if everything is working correctly.

### 6. Start the System

Use the automated startup script:

```bash
chmod +x scripts/run_servers.sh
./scripts/run_servers.sh
```

Or start components individually:

```bash
# Terminal 1 - Video Agent
uvicorn src.agents.video_agent_server:app --port 8001

# Terminal 2 - Text Agent  
uvicorn src.agents.text_agent_server:app --port 8002

# Terminal 3 - Composing Agent Web UI
python src/agents/composing_agents_main.py web
```

### 7. Access the Web Interface

Navigate to `http://localhost:8000` and select "CoordinatorAgent" from the dropdown.

## 📚 Detailed Setup

### Backend Options

#### Option A: Byaldi (Recommended for Development)

Byaldi is perfect for getting started quickly and is set up automatically by the setup script:

```bash
# Run the automated setup (recommended)
python scripts/setup_system.py

# Or manually create the index
python scripts/run_ingestion.py --video_dir data/videos --backend byaldi
```

The setup script automatically configures Byaldi as the default backend.

#### Option B: Vespa (Production)

For production deployments with large video collections:

```bash
# Start servers (auto-starts Vespa with persistent storage if not running)
./scripts/run_servers.sh vespa

# Deploy all Vespa schemas (first time only)
python scripts/deploy_all_schemas.py

# Process videos for Vespa
python scripts/run_ingestion.py --video_dir data/videos --backend vespa

# When done: Stop agent servers (keeps Vespa running)
./scripts/stop_servers.sh

# Stop Vespa container completely
./scripts/stop_vespa.sh
```

**Important**: 
- The `run_servers.sh` script automatically starts Vespa with persistent volumes if it's not already running
- The `start_vespa.sh` script ensures that Vespa data persists across container restarts by properly mounting volumes at `data/vespa/`
- Schema deployment is only needed for first-time setup or after cleaning Vespa data

### Elasticsearch Setup (Text Search)

If you have an Elasticsearch deployment:

```bash
export ELASTIC_CLOUD_ID="your_cloud_id"
export ELASTIC_API_KEY="your_api_key"
export ELASTIC_INDEX="your-text-index"
```

## 🎯 Usage Examples

### Basic Queries

Once the system is running, try these example queries:

**Video-focused queries:**
- "Show me videos of the product demonstration"
- "Find clips from yesterday's meeting"
- "Look for recordings of training sessions"

**Multi-modal queries (video-focused for now):**
- "Find videos about the new drone prototype from last week"
- "Search for video content about system installation"
- "Show me clips with red squares" (if using the sample test video)

> **Note:** Text search is currently disabled until Elasticsearch is configured. The system focuses on video search capabilities.

### Temporal Filtering

The system automatically detects and applies time filters:

- "Find videos from yesterday"
- "Show me documents from last week"
- "Search for content from 2024-01-15 to 2024-01-20"

### Programmatic Usage

You can also use the system programmatically:

```python
import asyncio
from cogniverse_agents.composing_agent import run_query_programmatically
from cogniverse_core.config import SystemConfig

async def search_example():
    # Initialize configuration
    config = SystemConfig(tenant_id="default")

    # Run query
    query = "Find information about system architecture"
    result = await run_query_programmatically(query, config)
    print(result["final_response"])

asyncio.run(search_example())
```

## 🔧 Customization

### Adding Your Own Data

#### Text Documents

1. Add documents to your Elasticsearch index
2. Ensure documents have `doc_id` and `content` fields
3. Update the index name in your configuration

#### Videos

1. Place video files in the `data/videos` directory
2. Choose your backend and run the ingestion pipeline:

```bash
# For Vespa backend (recommended - production ready)
python scripts/run_ingestion.py --video_dir data/videos --backend vespa

# For Byaldi backend (development/prototyping)
python scripts/run_ingestion.py --video_dir data/videos --backend byaldi

# Skip expensive steps if already processed
python scripts/run_ingestion.py --video_dir data/videos --backend vespa \
  --skip-keyframes --skip-audio --skip-descriptions  # Only generate embeddings
```

The system will automatically:
- Extract keyframes (threshold: 0.98 for scene changes)
- Generate visual descriptions using VLM
- Transcribe audio with word-level timestamps
- Create multi-vector embeddings (ColPali/ColQwen2)
- Index everything for search

### Modifying Agent Behavior

#### Composing Agent

Edit `src/agents/composing_agents_main.py` to modify:
- Query analysis logic
- Routing decisions
- Response synthesis

#### Search Agents

Modify the search logic in:
- `src/agents/text_agent_server.py` - Text search behavior
- `src/agents/video_agent_server.py` - Video search behavior

### Configuration Options

All configuration can be managed through:
- Environment variables
- `config.json` file
- Runtime parameters

See `src/tools/config.py` for all available options.

## 🐛 Troubleshooting

### Common Issues

#### "Ollama server not running" Error
```bash
# Start Ollama server
ollama serve

# Or run the setup script again
python scripts/setup_ollama.py
```

#### Agent Connection Failures
1. Check if all agent servers are running
2. Verify ports are not blocked
3. Run the test suite: `python scripts/test_system.py`

#### Model Loading Issues
1. Ensure you have enough RAM (16GB+ recommended)
2. Check CUDA installation for GPU usage
3. Some models may need manual download

#### Video Processing Failures
1. Check video file formats (MP4, MOV, AVI supported)
2. Ensure FFmpeg is installed
3. Verify sufficient disk space for processing

#### Vespa Data Persistence Issues
If your Vespa data disappears after container restarts:

1. **Always use the provided script**: `./scripts/start_vespa.sh`
2. **Check volume mounts**: Verify that `/opt/vespa/var` and `/opt/vespa/logs` are properly mounted
3. **Restart with proper volumes**: 
   ```bash
   docker stop vespa && docker rm vespa
   ./scripts/start_vespa.sh
   ```
4. **Re-index if needed**: If data is lost, re-run the ingestion:
   ```bash
   python scripts/run_ingestion.py --video_dir data/videos --backend vespa
   ```

### Logs and Debugging

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
```

Check the generated log files:
- `multi_agent_system.log` - Main system log
- `test_results.json` - Test results

### Performance Optimization

#### For Faster Video Processing:
- Use GPU acceleration
- Reduce video quality for testing
- Process videos in batches

#### For Better Search Quality:
- Use more descriptive file names
- Add metadata to videos
- Fine-tune embedding models

## 📈 Advanced Features

### Custom Models

Replace default models by updating configuration:

```json
{
  "vllm_model": "your/custom-vision-model",
  "colpali_model": "your/custom-retrieval-model",
  "embedding_model": "your/custom-embedding-model"
}
```

### Multiple Agent Instances

Scale horizontally by running multiple agent instances:

```bash
# Multiple video agents on different ports
uvicorn src.agents.video_agent_server:app --port 8001
uvicorn src.agents.video_agent_server:app --port 8003

# Load balance in configuration
```

### Integration with External Systems

The A2A protocol makes it easy to integrate with external systems:

```python
from cogniverse_core.tools.a2a_utils import A2AClient

client = A2AClient()
result = await client.send_task("http://your-agent-url", "search query")
```

## 🔒 Security Considerations

- Keep API keys secure and out of version control
- Use HTTPS in production deployments
- Implement authentication for agent endpoints
- Regularly update dependencies

## 📞 Support

If you encounter issues:

1. Run the test suite: `python scripts/test_system.py`
2. Check the troubleshooting section above
3. Review logs for error details
4. Ensure all prerequisites are met

## 🎉 What's Next?

Once you have the system running:

1. **Experiment** with different query types
2. **Add your own data** through the ingestion pipeline
3. **Customize agents** for your specific use case
4. **Scale up** with production backends (Vespa, Elasticsearch)
5. **Integrate** with your existing workflows

## Package Architecture Overview

Cogniverse uses a 10-package layered architecture:

### SDK Layer
- `cogniverse-sdk`: Interfaces and type contracts

### Foundation Layer
- `cogniverse-foundation`: Telemetry, logging, base utilities

### Core Layer
- `cogniverse-core`: System configuration, orchestration

### Agent Layer
- `cogniverse-agents`: Agent implementations (routing, search, etc.)

### Implementation Layer
- `cogniverse-retrieval`: Search backends (Vespa, Elasticsearch)
- `cogniverse-processing`: Video processing pipeline
- `cogniverse-synthetic`: Synthetic data generation
- `cogniverse-vlm`: VLM services

### Application Layer
- `cogniverse-services`: Web services, APIs

### Evaluation Layer
- `cogniverse-evaluation`: Metrics, experiments, Phoenix integration

For detailed architecture documentation, see [ARCHITECTURE.md](../ARCHITECTURE.md).

Your sophisticated multi-agent video search system is now ready to help you find information across video content with natural language queries!