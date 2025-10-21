# src/routing/__init__.py
"""
Comprehensive Routing System for Multi-Agent RAG

This module provides a flexible, extensible routing system with multiple strategies
and automatic optimization capabilities.
"""

from .base import RoutingDecision, RoutingMetrics, RoutingStrategy
from .config import RoutingConfig, load_config
from .optimizer import AutoTuningOptimizer, RoutingOptimizer
from .router import ComprehensiveRouter, TieredRouter
from .strategies import (
    EnsembleRoutingStrategy,
    GLiNERRoutingStrategy,
    HybridRoutingStrategy,
    KeywordRoutingStrategy,
    LangExtractRoutingStrategy,
    LLMRoutingStrategy,
)

__all__ = [
    "RoutingStrategy",
    "RoutingDecision",
    "RoutingMetrics",
    "GLiNERRoutingStrategy",
    "LLMRoutingStrategy",
    "KeywordRoutingStrategy",
    "HybridRoutingStrategy",
    "EnsembleRoutingStrategy",
    "LangExtractRoutingStrategy",
    "ComprehensiveRouter",
    "TieredRouter",
    "RoutingOptimizer",
    "AutoTuningOptimizer",
    "RoutingConfig",
    "load_config",
]
