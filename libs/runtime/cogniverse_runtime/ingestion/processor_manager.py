#!/usr/bin/env python3
"""
Pluggable ProcessorManager with auto-discovery - CLEAN VERSION.

No backward compatibility bullshit. Clean, simple, pluggable.
"""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any

from .processor_base import BaseProcessor, BaseStrategy


class ProcessorManager:
    """Plugin-based processor manager with auto-discovery."""

    def __init__(self, logger: logging.Logger, plugin_dir: Path | None = None):
        """
        Initialize processor manager with auto-discovery.

        Args:
            logger: Logger instance
            plugin_dir: Directory to search for processor plugins (defaults to processors/)
        """
        self.logger = logger
        self._processors: dict[str, BaseProcessor] = {}
        self._processor_classes: dict[str, type] = {}

        # Auto-discover processors
        if plugin_dir is None:
            plugin_dir = Path(__file__).parent / "processors"

        self._discover_processors(plugin_dir)
        self.logger.info(
            f"🔍 Discovered {len(self._processor_classes)} processor types"
        )

    def _discover_processors(self, plugin_dir: Path):
        """Auto-discover processor plugins by finding BaseProcessor subclasses."""
        if not plugin_dir.exists():
            self.logger.warning(f"Plugin directory not found: {plugin_dir}")
            return

        package_name = "cogniverse_runtime.ingestion.processors"

        for _importer, modname, ispkg in pkgutil.iter_modules([str(plugin_dir)]):
            if not ispkg:  # Skip packages, only load modules
                full_module_name = f"{package_name}.{modname}"
                try:
                    module = importlib.import_module(full_module_name)

                    # Look for BaseProcessor subclasses
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, BaseProcessor)
                            and attr != BaseProcessor
                            and hasattr(attr, "PROCESSOR_NAME")
                            and attr.PROCESSOR_NAME
                        ):
                            processor_name = attr.PROCESSOR_NAME
                            self._processor_classes[processor_name] = attr
                            self.logger.debug(
                                f"Discovered processor: {processor_name} from {modname}"
                            )

                except Exception as e:
                    self.logger.error(f"Failed to load module {full_module_name}: {e}")

    def initialize_from_strategies(self, strategy_set):
        """Initialize processors dynamically from strategy set."""
        all_requirements = {}

        # Get all strategies from the strategy set
        strategies = strategy_set.get_all_strategies()

        # Collect requirements from all strategies
        for strategy in strategies:
            if isinstance(strategy, BaseStrategy):
                requirements = strategy.get_required_processors()
                all_requirements.update(requirements)

                strategy_name = type(strategy).__name__
                self.logger.info(
                    f"🔧 {strategy_name} requires: {list(requirements.keys())}"
                )

        self._init_from_requirements(all_requirements)

    def _init_from_requirements(self, required_processors: dict[str, dict[str, Any]]):
        """Dynamically create processors from requirements."""
        for processor_name, processor_config in required_processors.items():
            if processor_name in self._processor_classes:
                processor_class = self._processor_classes[processor_name]
                self.logger.info(f"🔧 Creating {processor_name}")

                try:
                    # Add processor name to config for processors that need it
                    config_with_name = processor_config.copy()
                    config_with_name["processor_name"] = processor_name
                    processor = processor_class.from_config(
                        config_with_name, self.logger
                    )
                    self._processors[processor_name] = processor
                    self.logger.info(f"   ✅ {processor_name} initialized successfully")
                except Exception as e:
                    self.logger.error(f"   ❌ Failed to create {processor_name}: {e}")
                    raise
            else:
                available = list(self._processor_classes.keys())
                raise ValueError(
                    f"Unknown processor type: {processor_name}. "
                    f"Available processors: {available}"
                )

    def get_processor(self, name: str) -> BaseProcessor | None:
        """Get a processor by name."""
        return self._processors.get(name)

    def has_processor(self, name: str) -> bool:
        """Check if a processor is available."""
        return name in self._processors

    def list_processors(self) -> list[str]:
        """List all initialized processors."""
        return list(self._processors.keys())

    def list_available_processor_types(self) -> list[str]:
        """List all discoverable processor types."""
        return list(self._processor_classes.keys())

    def cleanup(self):
        """Cleanup all processors."""
        for processor in self._processors.values():
            if hasattr(processor, "cleanup"):
                try:
                    processor.cleanup()
                except Exception as e:
                    self.logger.error(
                        f"Error cleaning up {processor.get_processor_name()}: {e}"
                    )

        self._processors.clear()
