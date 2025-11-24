"""
Cogniverse Fine-Tuning Infrastructure.

End-to-end fine-tuning pipeline for LLM agents and embedding models.

Main components:
- Dataset extraction from telemetry (Phoenix/OpenTelemetry)
- Auto-selection of training method (SFT/DPO/contrastive)
- Synthetic data generation with mandatory approval
- LoRA/PEFT training (TRL for LLM, sentence-transformers for embeddings)
- Modal GPU integration (optional)

Quick Start:
    >>> from cogniverse_finetuning import finetune
    >>> from cogniverse_foundation.telemetry import TelemetryManager
    >>>
    >>> # Get telemetry provider
    >>> manager = TelemetryManager()
    >>> provider = manager.get_provider("tenant1", "cogniverse-tenant1")
    >>>
    >>> # Local training
    >>> result = await finetune(
    ...     telemetry_provider=provider,
    ...     tenant_id="tenant1",
    ...     project="cogniverse-tenant1",
    ...     model_type="llm",
    ...     agent_type="routing",
    ...     base_model="HuggingFaceTB/SmolLM-135M",
    ...     backend="local"
    ... )
    >>>
    >>> # Or remote GPU training
    >>> result = await finetune(
    ...     telemetry_provider=provider,
    ...     tenant_id="tenant1",
    ...     project="cogniverse-tenant1",
    ...     model_type="llm",
    ...     agent_type="routing",
    ...     backend="remote",
    ...     backend_provider="modal",
    ...     gpu="A100-40GB"
    ... )
    >>> print(f"Adapter saved to: {result.adapter_path}")
"""

from cogniverse_finetuning.orchestrator import (
    FinetuningOrchestrator,
    OrchestrationConfig,
    OrchestrationResult,
    analyze_dataset_status,
    finetune,
)

__version__ = "0.1.0"

__all__ = [
    "FinetuningOrchestrator",
    "OrchestrationConfig",
    "OrchestrationResult",
    "analyze_dataset_status",
    "finetune",
]
