"""Embedding generation pipeline for video processing."""

from cogniverse_core.common.models import (
    ColBERTModelLoader,
    ColPaliModelLoader,
    ColQwenModelLoader,
    ModelLoader,
    ModelLoaderFactory,
    VideoPrismModelLoader,
    get_or_load_model,
)
from cogniverse_sdk.document import ContentType, Document, ProcessingStatus

from .backend_factory import BackendFactory
from .embedding_generator import (
    BaseEmbeddingGenerator,
    EmbeddingResult,
)
from .embedding_generator_factory import (
    EmbeddingGeneratorFactory,
    create_embedding_generator,
)
from .embedding_generator_impl import EmbeddingGeneratorImpl

__all__ = [
    # Main classes
    "BaseEmbeddingGenerator",
    "EmbeddingGeneratorImpl",
    "EmbeddingGeneratorFactory",
    "EmbeddingResult",
    "Document",
    "ContentType",
    "ProcessingStatus",
    # Model loaders
    "get_or_load_model",
    "ModelLoaderFactory",
    "ModelLoader",
    "ColBERTModelLoader",
    "ColPaliModelLoader",
    "ColQwenModelLoader",
    "VideoPrismModelLoader",
    # Processors and factories
    "BackendFactory",
    # Factory function
    "create_embedding_generator",
]

__version__ = "1.0.0"
