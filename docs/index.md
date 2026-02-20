# Welcome to Cogniverse

**v0.1.0** · Multi-Agent AI Platform

Cogniverse is a self-optimizing multi-agent platform for intelligent processing and search across multi-modal content (video, audio, images, documents). It features A2A agent orchestration, continuous learning via DSPy, and complete multi-tenant isolation.

---

## Platform Overview

```mermaid
flowchart TD
    %% Main Request Flow
    user(("<span style='color:#000'>User</span>")) --> |request| agentlayer["<span style='color:#000'><b>Agents (A2A)</b><br/>Router · Composing · Search · ...</span>"]
    agentlayer --> |response| user
    agentlayer <--> store[("<span style='color:#000'>Content Store</span>")]
    agentlayer <--> |context| memory["<span style='color:#000'>Memory</span>"]
    agentlayer -.-> model["<span style='color:#000'>Model</span>"]
    config["<span style='color:#000'><b>Config</b><br/>Models · Embeddings</span>"] -.-> model
    config -.-> store

    %% Content Ingestion
    content[/"<span style='color:#000'>Multi-Modal Content</span>"/] --> |ingest| store

    %% Continuous Improvement
    agentlayer -.-> |traces| telemetry["<span style='color:#000'>Telemetry</span>"]
    telemetry -.-> evaluator["<span style='color:#000'>Evaluator</span>"]
    evaluator -.-> optimizer["<span style='color:#000'>Optimizer</span>"]
    optimizer -.-> |improves| agentlayer

    %% Experiments
    telemetry -.-> experiments["<span style='color:#000'>Experiments</span>"]
    experiments -.-> evaluator

    %% Training
    store -.-> |samples| generator["<span style='color:#000'>Synthetic Generator</span>"]
    generator -.-> trainer["<span style='color:#000'>Trainer</span>"]
    evaluator -.-> |annotations| trainer
    trainer -.-> |adapters| model

    style user fill:#90caf9,stroke:#1565c0,color:#000
    style content fill:#a5d6a7,stroke:#388e3c,color:#000
    style agentlayer fill:#ce93d8,stroke:#7b1fa2,color:#000
    style model fill:#81d4fa,stroke:#0288d1,color:#000
    style store fill:#90caf9,stroke:#1565c0,color:#000
    style memory fill:#90caf9,stroke:#1565c0,color:#000

    style telemetry fill:#a5d6a7,stroke:#388e3c,color:#000
    style experiments fill:#a5d6a7,stroke:#388e3c,color:#000
    style evaluator fill:#a5d6a7,stroke:#388e3c,color:#000
    style optimizer fill:#ffcc80,stroke:#ef6c00,color:#000
    style generator fill:#ffcc80,stroke:#ef6c00,color:#000
    style trainer fill:#ffcc80,stroke:#ef6c00,color:#000
    style config fill:#b0bec5,stroke:#546e7a,color:#000
```

---

## Key Features

- **Multi-Modal Processing** — Video, audio, images, documents with unified embeddings
- **Agent Orchestration** — A2A protocol with routing, search, and composing agents
- **Self-Optimization** — Continuous learning via GEPA, MIPRO, and synthetic data
- **LLM Fine-Tuning** — End-to-end pipeline from telemetry to LoRA adapters
- **Memory** — Context persistence across conversations
- **Experiments** — Run and track evaluation experiments with datasets
- **Hybrid Search** — BM25 + dense vectors with multiple ranking strategies
- **Configurable** — Pluggable models, embeddings, and backends
- **Multi-Tenant** — Schema-per-tenant isolation with independent telemetry
- **Production Ready** — OpenTelemetry tracing, metrics, and distributed traces

---

## Quick Start

```bash
# Install
git clone <repository-url> && cd cogniverse
uv sync

# Start services (Vespa, Phoenix, Ollama)
docker compose -f deployment/docker-compose.yml up -d

# Launch dashboard
uv run streamlit run scripts/phoenix_dashboard_standalone.py
# Open http://localhost:8501
```

See [Getting Started](operations/setup-installation.md) for ingestion and configuration.

---

## Built With

| Category | Technologies |
|----------|-------------|
| **Embedding Models** | ColPali, VideoPrism, ColQwen, Whisper |
| **Entity Extraction** | GLiNER (zero-shot NER) |
| **LLM Inference** | Ollama (Llama, Qwen, SmolLM, Gemma) |
| **Remote Infra** | Modal (inference, training) |
| **Search Backend** | Vespa (BM25, dense, hybrid ranking) |
| **Memory** | Mem0 (context persistence) |
| **Telemetry** | Phoenix (OpenTelemetry, tracing, experiments) |
| **Optimization** | DSPy 3.1+ (GEPA, MIPRO, SIMBA, Bootstrap) |
| **Fine-Tuning** | TRL (LoRA, SFT, DPO) |

---

## Documentation

| Audience | Resources |
|----------|-----------|
| **Users** | [User Guide](USER_GUIDE.md) · [Setup](operations/setup-installation.md) · [Configuration](operations/configuration.md) |
| **Developers** | [Developer Guide](DEVELOPER_GUIDE.md) · [Architecture](architecture/overview.md) · [Modules](modules/sdk.md) |
| **DevOps** | [Deployment](operations/deployment.md) · [Docker](operations/docker-deployment.md) · [Kubernetes](operations/kubernetes-deployment.md) |
