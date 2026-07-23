#!/usr/bin/env python3
"""
StrategyFactory - Creates strategy sets from explicit configuration.

Uses actual class names in config - no string mappings or if/elif logic.
"""

import importlib
import inspect
import logging
from typing import Any

from .processing_strategy_set import ProcessingStrategySet
from .processor_base import BaseStrategy

INFERENCE_SERVICE_PARAM = "inference_service"
logger = logging.getLogger(__name__)


class StrategyFactory:
    """Factory for creating strategy sets from explicit configuration."""

    @classmethod
    def create_from_profile_config(
        cls, profile_config: dict[str, Any]
    ) -> ProcessingStrategySet:
        """
        Create strategy set from profile configuration.

        Expects config format:
        {
          "inference_services": {                # optional, profile-level
            "embedding": "vllm_colpali",
            "transcription": "vllm_asr"
          },
          "strategies": {
            "segmentation": {
              "class": "FrameSegmentationStrategy",
              "params": {"fps": 0.5}
            },
            "transcription": {
              "class": "AudioTranscriptionStrategy",
              "params": {"model": "whisper-large-v3"}
            }
          }
        }

        Profile-level ``inference_services[<strategy_type>]`` becomes an
        ``inference_service=`` kwarg on the matching strategy — but only
        when the strategy's constructor declares the parameter (or
        accepts ``**kwargs``). Strategies that don't take it (in-process
        embedding strategies, segmentation, etc.) are not injected.

        Other params from the profile are passed through unchanged: a
        typo like ``"models": "..."`` in the JSON raises ``TypeError`` at
        construction so the misconfiguration is loud, not silent.

        Args:
            profile_config: Profile configuration dict

        Returns:
            ProcessingStrategySet with configured strategies
        """
        strategies_config = profile_config.get("strategies", {})
        inference_services = profile_config.get("inference_services") or {}
        strategies = {}

        for strategy_type, strategy_config in strategies_config.items():
            class_name = strategy_config.get("class")
            if not class_name:
                continue

            params = dict(strategy_config.get("params", {}))
            service_name = inference_services.get(strategy_type)
            if (
                service_name
                and INFERENCE_SERVICE_PARAM not in params
                and cls._strategy_accepts_inference_service(class_name)
            ):
                params[INFERENCE_SERVICE_PARAM] = service_name

            strategy_instance = cls._create_strategy_instance(class_name, params)
            if strategy_instance:
                strategies[strategy_type] = strategy_instance

        return ProcessingStrategySet(**strategies)

    @classmethod
    def _resolve_strategy_class(cls, class_name: str) -> type[BaseStrategy] | None:
        """Look up the strategy class on the strategies module.

        Returns ``None`` if the module or attribute cannot be resolved so
        ``create_from_profile_config`` can skip a single bad entry instead
        of failing the whole profile load.
        """
        try:
            strategies_module = importlib.import_module(
                "cogniverse_runtime.ingestion.strategies"
            )
            return getattr(strategies_module, class_name)
        except (ImportError, AttributeError) as exc:
            logger.error("Failed to resolve strategy class %r: %s", class_name, exc)
            return None

    @classmethod
    def _strategy_accepts_inference_service(cls, class_name: str) -> bool:
        """True when the strategy's ``__init__`` explicitly declares
        ``inference_service``.

        ``**kwargs`` deliberately does NOT count as opt-in: strategies that
        absorbed the kwarg that way silently discarded it, so the profile's
        inference service never reached the processor while the factory
        believed it was delivered. A strategy opts in by naming the
        parameter.
        """
        strategy_class = cls._resolve_strategy_class(class_name)
        if strategy_class is None:
            return False
        try:
            sig = inspect.signature(strategy_class.__init__)
        except (TypeError, ValueError):
            return False
        return INFERENCE_SERVICE_PARAM in sig.parameters

    @classmethod
    def _create_strategy_instance(
        cls, class_name: str, params: dict[str, Any]
    ) -> BaseStrategy:
        """Construct the strategy with the resolved params.

        Returns ``None`` if the class cannot be resolved; lets
        ``TypeError`` from a parameter mismatch propagate so a typo in
        profile config produces a loud failure.
        """
        strategy_class = cls._resolve_strategy_class(class_name)
        if strategy_class is None:
            return None
        # A param typo (e.g. "models" for "model") raises TypeError here; let
        # it propagate so the misconfiguration is loud, not a silently dropped
        # strategy — as the docstrings above promise.
        return strategy_class(**params)
