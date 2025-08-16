"""Shared model loaders for ingestion and querying."""

from .model_loaders import (
    get_or_load_model,
    ModelLoaderFactory,
    ModelLoader,
    ColPaliModelLoader,
    ColQwenModelLoader,
    VideoPrismModelLoader
)

from .videoprism_loader import (
    VideoPrismLoader,
    VideoPrismGlobalLoader,
    get_videoprism_loader
)

__all__ = [
    'get_or_load_model',
    'ModelLoaderFactory',
    'ModelLoader',
    'ColPaliModelLoader',
    'ColQwenModelLoader',
    'VideoPrismModelLoader',
    'VideoPrismLoader',
    'VideoPrismGlobalLoader',
    'get_videoprism_loader'
]