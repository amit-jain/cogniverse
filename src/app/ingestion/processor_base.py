#!/usr/bin/env python3
"""
Base classes for processors in the ingestion pipeline.

Provides foundation for pluggable processor architecture with auto-discovery.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging


class BaseProcessor(ABC):
    """Base class that all processors must extend."""
    
    PROCESSOR_NAME: str = ""  # Must be overridden by subclasses
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize processor with logger and optional parameters.
        
        Args:
            logger: Logger instance for this processor
            **kwargs: Additional processor-specific configuration
        """
        if not self.PROCESSOR_NAME:
            raise ValueError(f"{self.__class__.__name__} must define PROCESSOR_NAME")
        
        self.logger = logger
        self._config = kwargs
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger: logging.Logger) -> 'BaseProcessor':
        """
        Factory method to create processor from configuration using introspection.
        
        Automatically maps config keys to constructor parameters. Subclasses can
        override this if they need custom configuration logic.
        
        Args:
            config: Configuration dictionary for this processor
            logger: Logger instance
            
        Returns:
            Configured processor instance
        """
        import inspect
        
        # Get constructor signature
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        
        # Map config to constructor parameters, skipping 'self' and 'logger'
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'logger'):
                continue
                
            if param_name in config:
                kwargs[param_name] = config[param_name]
            elif param.default is not inspect.Parameter.empty:
                # Use default value if not in config
                kwargs[param_name] = param.default
        
        return cls(logger=logger, **kwargs)
    
    def get_processor_name(self) -> str:
        """Get the name of this processor."""
        return self.PROCESSOR_NAME
    
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration used to create this processor."""
        return self._config.copy()


class BaseStrategy(ABC):
    """Base class that all strategies must extend."""
    
    @abstractmethod
    def get_required_processors(self) -> Dict[str, Dict[str, Any]]:
        """
        Return required processors and their configurations.
        
        Returns:
            Dict mapping processor names to their configuration dictionaries
        """
        pass