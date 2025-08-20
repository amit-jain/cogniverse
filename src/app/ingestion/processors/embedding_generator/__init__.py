"""Embedding generation pipeline for video processing."""

from .embedding_generator import (
    EmbeddingResult,
    ProcessingConfig,
    EmbeddingGenerator
)

# Import new generic Document only
from src.common.document import Document, ContentType, ProcessingStatus

from src.common.models import (
    get_or_load_model,
    ModelLoaderFactory,
    ModelLoader,
    ColPaliModelLoader,
    ColQwenModelLoader,
    VideoPrismModelLoader
)

# Document builders no longer needed - backend handles this internally

from .embedding_processors import EmbeddingProcessor

from .backend_factory import BackendFactory

# VespaPyClient is now in backends/vespa/ingestion_client.py

from .embedding_generator_impl import EmbeddingGeneratorImpl

from .embedding_generator_factory import (
    EmbeddingGeneratorFactory,
    create_embedding_generator
)

__all__ = [
    # Main classes
    'EmbeddingGenerator',
    'EmbeddingGeneratorImpl',
    'EmbeddingResult',
    'ProcessingConfig',
    'Document',
    'ContentType', 
    'ProcessingStatus',
    
    # Model loaders
    'get_or_load_model',
    'ModelLoaderFactory',
    'ModelLoader',
    'ColPaliModelLoader',
    'ColQwenModelLoader',
    'VideoPrismModelLoader',
    
    # Processors and factories
    'EmbeddingProcessor',
    'BackendFactory',
    
    # Factory function
    'create_embedding_generator'
]

__version__ = '1.0.0'