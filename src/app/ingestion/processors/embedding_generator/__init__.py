"""Embedding generation pipeline for video processing."""

# Import new generic Document only
from src.common.document import ContentType, Document, ProcessingStatus
from src.common.models import (ColPaliModelLoader, ColQwenModelLoader,
                               ModelLoader, ModelLoaderFactory,
                               VideoPrismModelLoader, get_or_load_model)

from .backend_factory import BackendFactory
from .embedding_generator import (EmbeddingGenerator, EmbeddingResult,
                                  ProcessingConfig)
from .embedding_generator_factory import (EmbeddingGeneratorFactory,
                                          create_embedding_generator)
from .embedding_generator_impl import EmbeddingGeneratorImpl
from .embedding_processors import EmbeddingProcessor

# Document builders no longer needed - backend handles this internally



# VespaPyClient is now in backends/vespa/ingestion_client.py



__all__ = [
    # Main classes
    "EmbeddingGenerator",
    "EmbeddingGeneratorImpl",
    "EmbeddingGeneratorFactory",
    "EmbeddingResult",
    "ProcessingConfig",
    "Document",
    "ContentType",
    "ProcessingStatus",
    # Model loaders
    "get_or_load_model",
    "ModelLoaderFactory",
    "ModelLoader",
    "ColPaliModelLoader",
    "ColQwenModelLoader",
    "VideoPrismModelLoader",
    # Processors and factories
    "EmbeddingProcessor",
    "BackendFactory",
    # Factory function
    "create_embedding_generator",
]

__version__ = "1.0.0"
