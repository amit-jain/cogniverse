#!/usr/bin/env python3
"""
VLM Processor - Pluggable VLM description generation.

Generates descriptions for video frames using VLM models.
Delegates to VLMDescriptor for actual Modal VLM service communication.
"""

import logging
from typing import Any

from ..processor_base import BaseProcessor


class VLMProcessor(BaseProcessor):
    """Handles VLM-based description generation via VLMDescriptor."""

    PROCESSOR_NAME = "vlm"

    def __init__(
        self,
        logger: logging.Logger,
        vlm_endpoint: str,
        batch_size: int = 500,
        timeout: int = 10800,
        auto_start: bool = True,
    ):
        """
        Initialize VLM processor.

        Args:
            logger: Logger instance
            vlm_endpoint: URL of the Modal VLM service endpoint
            batch_size: Batch size for frame processing
            timeout: Request timeout in seconds (default 3 hours)
            auto_start: Whether to auto-start the Modal service if not running
        """
        super().__init__(logger)
        self.vlm_endpoint = vlm_endpoint
        self.batch_size = batch_size
        self.timeout = timeout
        self.auto_start = auto_start
        self._descriptor = None

    def _get_descriptor(self):
        """Lazy-init VLMDescriptor on first use."""
        if self._descriptor is None:
            from .vlm_descriptor import VLMDescriptor

            self._descriptor = VLMDescriptor(
                vlm_endpoint=self.vlm_endpoint,
                batch_size=self.batch_size,
                timeout=self.timeout,
                auto_start=self.auto_start,
            )
            self.logger.info(
                f"VLMDescriptor initialized with endpoint: {self.vlm_endpoint}"
            )
        return self._descriptor

    @classmethod
    def from_config(
        cls, config: dict[str, Any], logger: logging.Logger
    ) -> "VLMProcessor":
        """Create VLM processor from configuration."""
        vlm_endpoint = config.get("vlm_endpoint")
        if not vlm_endpoint:
            raise ValueError(
                "VLMProcessor requires 'vlm_endpoint' in config. "
                "Set it in the VLMDescriptionStrategy params in your profile config."
            )
        return cls(
            logger=logger,
            vlm_endpoint=vlm_endpoint,
            batch_size=config.get("batch_size", 500),
            timeout=config.get("timeout", 10800),
            auto_start=config.get("auto_start", True),
        )

    def process(self, *args, **kwargs) -> Any:
        """Process input data (implements BaseProcessor abstract method)."""
        if args and isinstance(args[0], dict):
            return self.generate_descriptions(args[0])
        return self.generate_descriptions(kwargs)

    def generate_descriptions(self, frames_data: dict[str, Any]) -> dict[str, Any]:
        """Generate descriptions for frames via VLMDescriptor."""
        descriptor = self._get_descriptor()
        return descriptor.generate_descriptions(frames_data)

    def cleanup(self):
        """Clean up VLM resources and stop service if needed."""
        if self._descriptor is not None:
            self._descriptor.stop_service()
            self._descriptor = None
