# 🚀 Quick Start - Video Search System

> **Need more details?** See the [Detailed Setup Guide](docs/setup/detailed_setup.md) for comprehensive instructions.

Get your multi-agent video search system running in 3 simple steps!

## Prerequisites

- **Python 3.12+** ✅ (compatibility issues resolved)
- **16GB+ RAM recommended** (for model loading)
- **5GB disk space** (for Llama 3.1 model)

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Setup Local AI Models

```bash
# Setup Ollama and Llama 3.1 (local AI coordinator)
python scripts/setup_ollama.py

# Setup video search system
python scripts/setup_system.py
```

**Ollama setup** will:
- ✅ Install Ollama (local AI runtime)
- ✅ Download Llama 3.1 model (~4GB)
- ✅ Start local AI server
- ✅ Test model functionality

**System setup** will:
- ✅ Create directories
- ✅ Generate a sample test video  
- ✅ Process videos and create search index
- ✅ Validate all components

## Step 3: Start the System

```bash
# Start servers (auto-starts Vespa if using Vespa backend)
./scripts/run_servers.sh

# If using Vespa backend (first time only):
python scripts/deploy_all_schemas.py
```

Then open **http://localhost:8000** in your browser and select **"CoordinatorAgent"**.

## 🎯 Try These Queries

- "Show me videos with moving objects"
- "Find clips from the test video" 
- "Search for videos with red squares"
- "Find videos from yesterday"

## 🧪 Test Search Performance

Test all 13 ranking strategies with different query types:

```bash
# Test comprehensive search capabilities
python tests/test_search_client.py

# Test specific ColPali visual search
python tests/test_colpali_search.py
```

The test will show you:
- ✅ **Text-only strategies** (BM25 with fieldsets)
- ✅ **Visual strategies** (ColPali float/binary embeddings)  
- ✅ **Hybrid strategies** (combining text + visual)
- ✅ **Input validation** (automatic strategy recommendations)
- ✅ **Performance comparison** across all ranking profiles

## 🔧 Add Your Own Videos

1. Copy video files (MP4, MOV, AVI) to `data/videos/`
2. Process videos:
   ```bash
   # For Vespa backend (recommended - production ready)
   python scripts/run_ingestion.py --video_dir data/videos --backend vespa
   
   # For Byaldi backend (development/prototyping)
   python scripts/run_ingestion.py --video_dir data/videos --backend byaldi
   ```
3. Restart the servers (or they'll auto-reload if already running)

## 📚 More Info

See [Detailed Setup Guide](docs/setup/detailed_setup.md) for comprehensive documentation and advanced features.

---

**That's it!** You now have a sophisticated AI system that can search through video content using natural language queries. 🎉 