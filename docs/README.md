# Cogniverse Documentation

**Version:** 2.0.0 | **Last Updated:** 2025-11-13 | **Status:** Production Ready

Complete documentation for Cogniverse - the multi-agent system with 10-package layered architecture.

---

## 📖 Documentation Index

### For Users

**Getting Started:**
- **[Setup & Installation](operations/setup-installation.md)** - Complete installation guide (UV workspace, Docker, services)
- **[User Guide](USER_GUIDE.md)** - Complete user guide with features, operations, and API reference
- **[Configuration Guide](operations/configuration.md)** - Multi-tenant configuration and profiles
- **[Troubleshooting](operations/troubleshooting.md)** - Common issues and solutions

**Operations:**
- **[Deployment Guide](operations/deployment.md)** - Docker, Kubernetes, Modal deployment
- **[Docker Deployment](operations/docker-deployment.md)** - Docker Compose production setup
- **[Kubernetes Deployment](operations/kubernetes-deployment.md)** - K8s deployment with Helm
- **[Multi-Tenant Operations](operations/multi-tenant-ops.md)** - Tenant lifecycle management
- **[Performance Monitoring](operations/performance-monitoring.md)** - Monitoring and observability

### For Developers

**Core Guides:**
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Complete developer onboarding (setup, workflows, best practices)
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute code (coming soon)
- **[Testing Guide](testing/TESTING_GUIDE.md)** - Comprehensive testing documentation (coming soon)
- **[Code Style](development/code-style.md)** - Code standards and conventions

**Architecture:**
- **[Architecture Overview](architecture/overview.md)** - 10-package layered architecture
- **[SDK Architecture](architecture/sdk-architecture.md)** - UV workspace deep-dive
- **[Multi-Tenant Architecture](architecture/multi-tenant.md)** - Complete tenant isolation
- **[System Flows](architecture/system-flows.md)** - 20+ architectural diagrams
- **[Integration](architecture/integration.md)** - Component integration patterns

**Module Documentation (by Layer):**

*Foundation Layer:*
- **[SDK Module](modules/sdk.md)** - Pure backend interfaces, Document model
- **[Foundation Module](modules/foundation.md)** - Config base, telemetry interfaces (coming soon)

*Core Layer:*
- **[Core Module](modules/common.md)** - Base classes, registries, memory
- **[Evaluation Module](modules/evaluation.md)** - Experiments, metrics, datasets
- **[Telemetry Phoenix Module](modules/telemetry-phoenix.md)** - Phoenix provider plugin (coming soon)

*Implementation Layer:*
- **[Agents Module](modules/agents.md)** - Routing, search, orchestration agents
- **[Vespa Module](modules/backends.md)** - Vespa backend, tenant schemas
- **[Synthetic Module](modules/synthetic.md)** - Synthetic data generation (coming soon)

*Application Layer:*
- **[Runtime Module](modules/runtime.md)** - FastAPI server, ingestion (coming soon)
- **[Dashboard Module](modules/dashboard.md)** - Streamlit UI, analytics (coming soon)

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
- **[Testing Guide](testing/TESTING_GUIDE.md)** - Comprehensive testing (coming soon)
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

### 10-Package Architecture

Cogniverse uses a **UV workspace** with 10 packages in layered architecture:

**Foundation Layer:**
1. [cogniverse-sdk](modules/sdk.md) - Pure backend interfaces (zero dependencies)
2. [cogniverse-foundation](modules/foundation.md) - Config base, telemetry interfaces

**Core Layer:**
3. [cogniverse-core](modules/common.md) - Base classes, registries, memory
4. [cogniverse-evaluation](modules/evaluation.md) - Experiments, metrics, datasets
5. [cogniverse-telemetry-phoenix](modules/telemetry-phoenix.md) - Phoenix provider (plugin)

**Implementation Layer:**
6. [cogniverse-agents](modules/agents.md) - Routing, search, orchestration
7. [cogniverse-vespa](modules/backends.md) - Vespa backend, schemas
8. [cogniverse-synthetic](modules/synthetic.md) - Synthetic data generation

**Application Layer:**
9. [cogniverse-runtime](modules/runtime.md) - FastAPI server, ingestion
10. [cogniverse-dashboard](modules/dashboard.md) - Streamlit UI, analytics

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
| Setup & Installation | ✅ Complete | 2025-11-13 |
| User Guide | ✅ Complete | 2025-11-13 |
| Developer Guide | ✅ Complete | 2025-11-13 |
| Architecture Docs | ✅ Complete | 2025-10-15 |
| SDK Module Doc | ✅ Complete | 2025-11-13 |
| Other Module Docs | ⚠️ Partial | 2025-10-15 |
| Testing Guide | 🚧 In Progress | - |
| Contributing Guide | 🚧 In Progress | - |

**Legend:**
- ✅ Complete and production-ready
- ⚠️ Partially complete, needs updates
- 🚧 In progress
- ❌ Not started

---

**Version**: 2.0.0
**Architecture**: UV Workspace (10 Packages - Layered Architecture)
**Last Updated**: 2025-11-13
**Status**: Production Ready