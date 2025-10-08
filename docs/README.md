# Cogniverse Documentation

Comprehensive documentation for the Cogniverse multi-agent RAG system with video content analysis and search capabilities.

---

## Quick Start

**New to Cogniverse?** Start here:
1. [Architecture Overview](architecture/overview.md) - Understand the system design
2. [Setup & Installation](operations/setup-installation.md) - Get up and running
3. [System Flows](architecture/system-flows.md) - See how everything works together

---

## 📚 Documentation Structure

### Architecture
High-level system design and integration patterns

- **[Overview](architecture/overview.md)** - System architecture, components, and design patterns
- **[System Flows](architecture/system-flows.md)** - End-to-end scenarios with Mermaid diagrams
- **[Integration](architecture/integration.md)** - System integration patterns and testing

### Modules
Technical documentation for each system module

- **[Agents](modules/agents.md)** - Multi-agent orchestration and routing
- **[Routing](modules/routing.md)** - Intelligent routing strategies and optimization
- **[Common](modules/common.md)** - Shared utilities and configuration
- **[Backends](modules/backends.md)** - Vespa search integration
- **[Telemetry](modules/telemetry.md)** - Phoenix observability and monitoring
- **[Evaluation](modules/evaluation.md)** - Experiment tracking and evaluation
- **[Ingestion](modules/ingestion.md)** - Video processing pipeline
- **[Optimization](modules/optimization.md)** - DSPy optimization framework
- **[Search & Reranking](modules/search-reranking.md)** - Result quality optimization
- **[Cache](modules/cache.md)** - Tiered caching architecture
- **[Utils](modules/utils.md)** - Utility functions and helpers
- **[Tools](modules/tools.md)** - A2A protocol and video tools

### Operations
Guides for deploying and operating the system

- **[Setup & Installation](operations/setup-installation.md)** - Complete installation guide
- **[Configuration](operations/configuration.md)** - Multi-tenant configuration management
- **[Deployment](operations/deployment.md)** - Production deployment patterns
- **[Performance & Monitoring](operations/performance-monitoring.md)** - Performance targets and monitoring

### Development
Development-specific documentation

- **[Scripts & Operations](development/scripts-operations.md)** - Deployment and ingestion scripts
- **[UI & Dashboard](development/ui-dashboard.md)** - Streamlit dashboards
- **[Instrumentation](development/instrumentation.md)** - Phoenix telemetry and observability

---

## 🎯 Common Tasks

### Getting Started
```bash
# Install dependencies
pip install uv
uv sync

# Start services (Vespa, Phoenix, Ollama)
# See operations/setup-installation.md for details

# Run a simple query
uv run python examples/simple_query.py
```

### Development
```bash
# Run tests
JAX_PLATFORM_NAME=cpu uv run pytest

# Run ingestion
uv run python scripts/run_ingestion.py --video_dir data/videos --backend vespa

# Start Phoenix dashboard
uv run streamlit run scripts/phoenix_dashboard_standalone.py
```

### Deployment
See [Deployment Guide](operations/deployment.md) for:
- Docker Compose deployment
- Kubernetes/Helm charts
- Cloud platforms (AWS, GCP, Modal)

---

## 📖 Reading Guide

### For Learning
1. **Start**: [Architecture Overview](architecture/overview.md)
2. **Understand**: [System Flows](architecture/system-flows.md)
3. **Deep Dive**: Pick a [module](modules/) based on your interest
4. **Practice**: [Setup & Installation](operations/setup-installation.md)

### For Implementation
1. **Setup**: [Setup & Installation](operations/setup-installation.md)
2. **Configure**: [Configuration](operations/configuration.md)
3. **Deploy**: [Deployment](operations/deployment.md)
4. **Monitor**: [Performance & Monitoring](operations/performance-monitoring.md)

### For Developers
1. **Architecture**: [Overview](architecture/overview.md)
2. **Modules**: Browse [modules/](modules/)
3. **Scripts**: [Scripts & Operations](development/scripts-operations.md)
4. **Testing**: [Integration](architecture/integration.md)

---

## 🔍 Key Concepts

- **Multi-Agent System**: Composing Agent orchestrates specialized agents (Video Search, Routing, etc.)
- **Multi-Tenant**: Schema-per-tenant isolation in Vespa, project isolation in Phoenix
- **DSPy 3.0**: Agent optimization framework with GRPO/GEPA
- **ColPali/VideoPrism**: Multi-modal embedding models for video content
- **Vespa**: Distributed vector database (port 8080, 19071)
- **Phoenix**: Telemetry and experiment tracking (port 6006, 4317)
- **Ollama**: Local LLM inference (port 11434)

---

## 📁 Additional Resources

### Specialized Guides
- **[Modal Deployment](modal/)** - Serverless GPU deployment
- **[Setup Details](setup/)** - Detailed setup instructions
- **[Testing](testing/)** - Search client and strategy testing

### Legacy Documentation
Some files remain for reference:
- `CONFIGURATION_SYSTEM.md` - Configuration system details
- `PERFORMANCE_TARGETS.md` - Performance benchmarks
- `deployment.md` - Legacy deployment notes

**Note**: These will be integrated or removed in future updates.

---

## 🤝 Contributing

When adding new documentation:
1. Follow the existing structure (architecture/, modules/, operations/, development/)
2. Use descriptive filenames (no numbers)
3. Include Mermaid diagrams where helpful
4. Cross-reference related documents
5. Keep the "why" not just the "what"

---

## 📞 Support

- **Issues**: Report at your issue tracker
- **Questions**: Check existing documentation first
- **Updates**: Documentation is maintained alongside code

---

**Last Updated**: 2025-10-08
**Version**: 2.0 (Professional structure, no study guides)
