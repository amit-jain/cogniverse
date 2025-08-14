"""
Common utilities and configuration for Cogniverse.

Shared code used by all Cogniverse components.
"""

__version__ = "0.1.0"

from .config import Config, get_config

__all__ = ["Config", "get_config"]