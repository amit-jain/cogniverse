# Cogniverse Documentation

**Version:** 2.1.0 | **Last Updated:** 2025-01-13 | **Status:** Production Ready

Complete documentation for Cogniverse - the multi-agent system with 11-package layered architecture.

---

## 📖 Documentation Index

### For Users

**Getting Started:**
- **[Quick Start](QUICKSTART.md)** - 5-minute getting started guide
- **[Setup & Installation](operations/setup-installation.md)** - Complete installation guide (UV workspace, Docker, services)
- **[User Guide](USER_GUIDE.md)** - Complete user guide with features, operations, and API reference
- **[Glossary](GLOSSARY.md)** - Domain terms and concepts reference
- **[Configuration Guide](operations/configuration.md)** - Multi-tenant configuration and profiles
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

**Operations:**
- **[Deployment Guide](operations/deployment.md)** - Docker, Kubernetes, Modal deployment
- **[Docker Deployment](operations/docker-deployment.md)** - Docker Compose production setup
- **[Kubernetes Deployment](operations/kubernetes-deployment.md)** - K8s deployment with Helm
- **[Multi-Tenant Operations](operations/multi-tenant-ops.md)** - Tenant lifecycle management
- **[Performance Monitoring](operations/performance-monitoring.md)** - Monitoring and observability

### For Developers

**Core Guides:**
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Complete developer onboarding (setup, workflows, best practices)
- **[Creating Agents Tutorial](tutorials/creating-agents.md)** - Step-by-step guide to building custom agents
- **[Testing Guide](testing/TESTING_GUIDE.md)** - Comprehensive testing documentation
- **[Code Style](development/code-style.md)** - Code standards and conventions

**Architecture:**
- **[Architecture Overview](architecture/overview.md)** - 11-package layered architecture
- **[SDK Architecture](architecture/sdk-architecture.md)** - UV workspace deep-dive
- **[Multi-Tenant Architecture](architecture/multi-tenant.md)** - Complete tenant isolation
- **[Multi-Agent Interactions](architecture/multi-agent-interactions.md)** - Agent communication flows with 8 Mermaid diagrams (orchestrator, registry, workflows)
- **[Ensemble Composition](architecture/ensemble-composition.md)** - RRF fusion algorithm, parallel profile execution, performance analysis
- **[System Flows](architecture/system-flows.md)** - 20+ architectural diagrams
- **[Integration](architecture/integration.md)** - Component integration patterns

**Module Documentation (by Layer):**

*Foundation Layer:*
- **[SDK Module](modules/sdk.md)** - Pure backend interfaces, Document model
- **[Foundation Module](modules/foundation.md)** - ConfigManager, TelemetryManager, multi-tenant infrastructure

*Core Layer:*
- **[Core Module](modules/core.md)** - AgentBase, A2AAgent, mixins, registries, memory
- **[Evaluation Module](modules/evaluation.md)** - Experiments, metrics, datasets
- **[Telemetry Module](modules/telemetry.md)** - Phoenix provider, OpenTelemetry integration

*Implementation Layer:*
- **[Agents Module](modules/agents.md)** - Routing, search, orchestration agents
- **[Vespa Module](modules/backends.md)** - Vespa backend, tenant schemas
- **[Fine-Tuning Module](modules/finetuning.md)** - LLM fine-tuning pipeline (SFT, DPO, evaluation, dashboard)
- **[Synthetic Module](modules/synthetic.md)** - DSPy-driven synthetic data generation

*Application Layer:*
- **[Runtime Module](modules/runtime.md)** - FastAPI server, video ingestion pipeline
- **[Dashboard Module](modules/dashboard.md)** - Streamlit UI, Phoenix analytics

**Other Module Docs:**
- **[Routing Module](modules/routing.md)** - Query routing and optimization
- **[Ingestion Module](modules/ingestion.md)** - Video processing pipeline
- **[Search & Reranking](modules/search-reranking.md)** - Multi-modal search
- **[Cache Module](modules/cache.md)** - Multi-modal LRU cache
- **[Tools Module](modules/tools.md)** - Agent tools and A2A protocol
- **[Optimization](modules/optimization.md)** - DSPy optimization strategies

### For DevOps

**Deployment:**
- **[Deployment Overview](operations/deployment.md)** - All deployment options
- **[Docker Deployment](operations/docker-deployment.md)** - Docker Compose setup
- **[Kubernetes Deployment](operations/kubernetes-deployment.md)** - Helm charts and K8s
- **[Modal Deployment](modal/deployment_guide.md)** - Serverless deployment
- **[Argo Workflows](operations/argo-workflows.md)** - Workflow orchestration
- **[Istio Service Mesh](operations/istio-service-mesh.md)** - Service mesh setup

**Operations:**
- **[Multi-Tenant Operations](operations/multi-tenant-ops.md)** - Tenant management
- **[Configuration Management](operations/configuration.md)** - Config system
- **[Performance Monitoring](operations/performance-monitoring.md)** - Monitoring setup
- **[Troubleshooting](operations/troubleshooting.md)** - Common issues

### Cross-Cutting Documentation

**Configuration:**
- **[Configuration System](CONFIGURATION_SYSTEM.md)** - Complete config architecture
- **[Performance Targets](PERFORMANCE_TARGETS.md)** - Performance goals and benchmarks

**Testing:**
- **[Testing Guide](testing/TESTING_GUIDE.md)** - Comprehensive testing documentation
- **[Pytest Best Practices](testing/pytest-best-practices.md)** - Testing patterns
- **[Comprehensive Testing Plan](operations/comprehensive-testing-plan.md)** - Test strategy

**Data & Synthetic:**
- **[Synthetic Data Generation](synthetic-data-generation.md)** - Training data for optimizers

**Diagrams:**
- **[SDK Architecture Diagrams](diagrams/sdk-architecture-diagrams.md)** - Visual architecture
- **[Multi-Tenant Diagrams](diagrams/multi-tenant-diagrams.md)** - Tenant isolation

---

## 🚀 Quick Navigation

**I want to...**

| Goal | Document |
|------|----------|
| **Install Cogniverse** | [Setup & Installation](operations/setup-installation.md) |
| **Use Cogniverse (user)** | [User Guide](USER_GUIDE.md) |
| **Develop features** | [Developer Guide](DEVELOPER_GUIDE.md) |
| **Understand architecture** | [Architecture Overview](architecture/overview.md) |
| **Deploy to production** | [Deployment Guide](operations/deployment.md) |
| **Understand a package** | [Module Documentation](modules/) |
| **Run tests** | [Testing Guide](testing/TESTING_GUIDE.md) |
| **Configure tenants** | [Multi-Tenant Operations](operations/multi-tenant-ops.md) |
| **Monitor system** | [Performance Monitoring](operations/performance-monitoring.md) |
| **Troubleshoot issues** | [Troubleshooting](operations/troubleshooting.md) |

---

## 📦 Package Documentation

### 11-Package Architecture

Cogniverse uses a **UV workspace** with 11 packages in layered architecture:

**Foundation Layer:**
1. [cogniverse-sdk](modules/sdk.md) - Pure backend interfaces (zero dependencies)
2. [cogniverse-foundation](modules/foundation.md) - Config base, telemetry interfaces

**Core Layer:**
3. [cogniverse-core](modules/core.md) - Base classes, registries, memory
4. [cogniverse-evaluation](modules/evaluation.md) - Experiments, metrics, datasets
5. [cogniverse-telemetry-phoenix](modules/telemetry.md) - Phoenix provider (plugin)

**Implementation Layer:**
6. [cogniverse-agents](modules/agents.md) - Routing, search, orchestration
7. [cogniverse-vespa](modules/backends.md) - Vespa backend, schemas
8. [cogniverse-finetuning](modules/finetuning.md) - LLM fine-tuning pipeline (SFT, DPO, evaluation, dashboard)
9. [cogniverse-synthetic](modules/synthetic.md) - Synthetic data generation

**Application Layer:**
10. [cogniverse-runtime](modules/runtime.md) - FastAPI server, ingestion
11. [cogniverse-dashboard](modules/dashboard.md) - Streamlit UI, analytics

---

## 🎯 By Audience

### Users
1. [Setup & Installation](operations/setup-installation.md)
2. [User Guide](USER_GUIDE.md)
3. [Configuration](operations/configuration.md)
4. [Troubleshooting](operations/troubleshooting.md)

### Developers
1. [Developer Guide](DEVELOPER_GUIDE.md)
2. [Architecture Overview](architecture/overview.md)
3. [Module Documentation](modules/)
4. [Testing Guide](testing/TESTING_GUIDE.md)

### DevOps
1. [Deployment Guide](operations/deployment.md)
2. [Multi-Tenant Operations](operations/multi-tenant-ops.md)
3. [Performance Monitoring](operations/performance-monitoring.md)
4. [Docker/Kubernetes Deployment](operations/kubernetes-deployment.md)

---

## 🔗 External Links

- **[Project README](../Readme.md)** - Project overview and quick start
- **[GitHub Repository](https://github.com/org/cogniverse)** - Source code
- **[GitHub Issues](https://github.com/org/cogniverse/issues)** - Bug reports and features
- **[Phoenix Dashboard](http://localhost:8501)** - Telemetry and experiments (when running)

---

## 📊 Documentation Status

| Section | Status | Last Updated |
|---------|--------|--------------|
| Quick Start | ✅ Complete | 2025-01-13 |
| Setup & Installation | ✅ Complete | 2025-11-13 |
| User Guide | ✅ Complete | 2025-11-13 |
| Developer Guide | ✅ Complete | 2025-11-13 |
| Creating Agents Tutorial | ✅ Complete | 2025-01-13 |
| Architecture Docs | ✅ Complete | 2025-01-13 |
| Module Docs (all 11) | ✅ Complete | 2025-01-13 |
| Testing Guide | ✅ Complete | 2025-01-18 |
| Code Style Guide | ✅ Complete | 2025-01-13 |
| Troubleshooting | ✅ Complete | 2025-01-13 |
| Glossary | ✅ Complete | 2025-01-13 |

**Legend:**
- ✅ Complete and production-ready
- ⚠️ Partially complete, needs updates
- 🚧 In progress
- ❌ Not started

---

**Version**: 2.1.0
**Architecture**: UV Workspace (11 Packages - Layered Architecture)
**Last Updated**: 2025-01-13
**Status**: Production Ready