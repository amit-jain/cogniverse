"""
Adapter Registry Module

Provides adapter lifecycle management for trained LoRA adapters.

Usage:
    >>> from cogniverse_finetuning.registry import AdapterRegistry, AdapterMetadata
    >>> registry = AdapterRegistry()
    >>> adapter_id = registry.register_adapter(
    ...     tenant_id="tenant1",
    ...     name="routing_sft",
    ...     version="1.0.0",
    ...     base_model="SmolLM-135M",
    ...     model_type="llm",
    ...     training_method="sft",
    ...     adapter_path="outputs/adapters/sft_routing/",
    ...     agent_type="routing"
    ... )
    >>> registry.activate_adapter(adapter_id)
"""

from cogniverse_finetuning.registry.adapter_registry import (
    AdapterRegistry,
    generate_adapter_id,
)
from cogniverse_finetuning.registry.inference import (
    AdapterInfo,
    get_active_adapter_for_inference,
    list_available_adapters,
    resolve_adapter_path,
)
from cogniverse_finetuning.registry.models import AdapterMetadata
from cogniverse_finetuning.registry.storage import (
    AdapterStorage,
    HuggingFaceStorage,
    LocalStorage,
    download_adapter,
    get_storage_backend,
    upload_adapter,
)

__all__ = [
    "AdapterInfo",
    "AdapterMetadata",
    "AdapterRegistry",
    "AdapterStorage",
    "HuggingFaceStorage",
    "LocalStorage",
    "download_adapter",
    "generate_adapter_id",
    "get_active_adapter_for_inference",
    "get_storage_backend",
    "list_available_adapters",
    "resolve_adapter_path",
    "upload_adapter",
]
