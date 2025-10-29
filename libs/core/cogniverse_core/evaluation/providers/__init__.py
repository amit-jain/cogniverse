"""
Evaluator provider registry and utilities.

This module provides a registry pattern for evaluator providers, allowing
evaluation code to work with different telemetry backends.
"""

from typing import Optional

from .base import (
    AnalyticsProvider,
    EvaluatorFramework,
    EvaluatorProvider,
    MonitoringProvider,
)

__all__ = [
    "EvaluatorProvider",
    "EvaluatorFramework",
    "AnalyticsProvider",
    "MonitoringProvider",
    "get_evaluator_provider",
    "register_evaluator_provider",
]

# Global registry for evaluator providers
_evaluator_provider_registry: dict[str, type[EvaluatorProvider]] = {}
_default_evaluator_provider: Optional[EvaluatorProvider] = None


def register_evaluator_provider(
    name: str, provider_class: type[EvaluatorProvider]
) -> None:
    """
    Register an evaluator provider implementation.

    Args:
        name: Provider name (e.g., 'phoenix', 'langsmith')
        provider_class: Provider class implementing EvaluatorProvider
    """
    _evaluator_provider_registry[name] = provider_class


def get_evaluator_provider(
    provider_name: Optional[str] = None,
    tenant_id: str = "default",
    project_name: Optional[str] = None,
) -> EvaluatorProvider:
    """
    Get evaluator provider instance.

    If no provider_name specified, returns the default provider (Phoenix).
    Provider instances are created on-demand.

    Args:
        provider_name: Name of provider to use (None = default)
        tenant_id: Tenant identifier
        project_name: Project name for telemetry

    Returns:
        EvaluatorProvider instance

    Raises:
        ValueError: If provider not found
    """
    global _default_evaluator_provider

    # If no provider specified, return default
    if provider_name is None:
        if _default_evaluator_provider is None:
            # Lazy import to avoid circular dependencies
            try:
                from cogniverse_telemetry_phoenix.evaluation.provider import (
                    PhoenixEvaluatorProvider,
                )

                from cogniverse_core.telemetry.manager import get_telemetry_manager

                # Get telemetry manager and provider
                telemetry_manager = get_telemetry_manager()
                telemetry_provider = telemetry_manager.get_provider(
                    tenant_id=tenant_id, project_name=project_name
                )

                # Create Phoenix evaluator provider
                _default_evaluator_provider = PhoenixEvaluatorProvider(
                    telemetry_provider=telemetry_provider
                )
            except ImportError as e:
                raise ValueError(
                    f"Default Phoenix evaluator provider not available: {e}"
                ) from e

        return _default_evaluator_provider

    # Get provider from registry
    if provider_name not in _evaluator_provider_registry:
        available = ", ".join(_evaluator_provider_registry.keys())
        raise ValueError(
            f"Evaluator provider '{provider_name}' not found. "
            f"Available providers: {available}"
        )

    provider_class = _evaluator_provider_registry[provider_name]
    return provider_class(tenant_id=tenant_id, project_name=project_name)


def reset_evaluator_provider() -> None:
    """
    Reset default evaluator provider.

    Useful for testing and cleanup.
    """
    global _default_evaluator_provider
    _default_evaluator_provider = None
