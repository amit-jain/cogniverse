"""Create requests for ``POST /admin/profiles`` built from a profile definition."""

from typing import Any

from cogniverse_foundation.config.unified_config import BackendProfileConfig


def profile_create_payload(
    profile_name: str,
    profile_def: dict[str, Any],
    tenant_id: str,
    *,
    deploy_schema: bool = True,
) -> dict[str, Any]:
    """The request that creates ``profile_def`` (a ``configs/config.json``
    ``backend.profiles`` entry) for ``tenant_id``, every key of it carried:
    the named fields as fields and the rest as ``extra_config``."""
    profile = BackendProfileConfig.from_dict(profile_name, profile_def)
    return {
        "profile_name": profile_name,
        "tenant_id": tenant_id,
        "type": profile.type,
        "description": profile.description,
        "schema_name": profile.schema_name or profile_name,
        "embedding_model": profile.embedding_model,
        "pipeline_config": profile.pipeline_config,
        "strategies": profile.strategies,
        "embedding_type": profile.embedding_type or "multi_vector",
        "schema_config": profile.schema_config,
        "model_specific": profile.model_specific or None,
        "model_loader": profile.model_loader,
        "process_type": profile.process_type,
        "extra_config": profile.extra_config,
        "deploy_schema": deploy_schema,
    }
