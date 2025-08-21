# src/routing/__init__.py
"""
Comprehensive Routing System for Multi-Agent RAG

This module provides a flexible, extensible routing system with multiple strategies
and automatic optimization capabilities.
"""

from .base import RoutingStrategy, RoutingDecision, RoutingMetrics
from .strategies import (
    GLiNERRoutingStrategy,
    LLMRoutingStrategy,
    KeywordRoutingStrategy,
    HybridRoutingStrategy,
    EnsembleRoutingStrategy,
    LangExtractRoutingStrategy
)
from .router import ComprehensiveRouter, TieredRouter
from .optimizer import RoutingOptimizer, AutoTuningOptimizer
from .config import RoutingConfig, load_config

__all__ = [
    'RoutingStrategy',
    'RoutingDecision',
    'RoutingMetrics',
    'GLiNERRoutingStrategy',
    'LLMRoutingStrategy',
    'KeywordRoutingStrategy',
    'HybridRoutingStrategy',
    'EnsembleRoutingStrategy',
    'LangExtractRoutingStrategy',
    'ComprehensiveRouter',
    'TieredRouter',
    'RoutingOptimizer',
    'AutoTuningOptimizer',
    'RoutingConfig',
    'load_config'
]