"""Embedding generation pipeline for video processing."""

from .embedding_generator import (
    EmbeddingResult,
    ProcessingConfig,
    EmbeddingGenerator
)

# Import from core
from src.core import Document, MediaType, TemporalInfo, SegmentInfo

from src.models import (
    get_or_load_model,
    ModelLoaderFactory,
    ModelLoader,
    ColPaliModelLoader,
    ColQwenModelLoader,
    VideoPrismModelLoader
)

# Document builders no longer needed - backend handles this internally

from .embedding_processors import EmbeddingProcessor

from .backend_client import BackendClient
from .backend_factory import BackendFactory

from .vespa_pyvespa_client import VespaPyClient

from .embedding_generator_impl import EmbeddingGeneratorImpl

# Export implementation as EmbeddingGenerator for backward compatibility
EmbeddingGenerator = EmbeddingGeneratorImpl

from .embedding_generator_factory import (
    EmbeddingGeneratorFactory,
    create_embedding_generator
)

__all__ = [
    # Main classes (EmbeddingGenerator is the implementation)
    'EmbeddingGenerator',
    'EmbeddingResult',
    'ProcessingConfig',
    'Document',
    'MediaType',
    'TemporalInfo',
    'SegmentInfo',
    
    # Model loaders
    'get_or_load_model',
    'ModelLoaderFactory',
    'ModelLoader',
    'ColPaliModelLoader',
    'ColQwenModelLoader',
    'VideoPrismModelLoader',
    
    # Processors and clients
    'EmbeddingProcessor',
    'BackendClient',
    'BackendFactory',
    'VespaPyClient',
    
    # Factory function
    'create_embedding_generator'
]

__version__ = '1.0.0'