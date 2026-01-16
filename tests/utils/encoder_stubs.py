"""
Lightweight encoder stubs for testing ensemble search with different embedding dimensions.

These stubs simulate real encoder behavior without loading heavy model weights:
- Correct embedding dimensions for each model type
- Realistic initialization time (simulating model loading)
- Deterministic output for reproducible tests
"""

import time
from typing import Any, Dict

import numpy as np


class EncoderStub:
    """Base class for encoder stubs"""

    def __init__(self, model_name: str, embedding_dim: int, load_time_ms: float = 10.0):
        """
        Initialize encoder stub.

        Args:
            model_name: Name of the model being simulated
            embedding_dim: Embedding dimension to output
            load_time_ms: Simulated model loading time in milliseconds
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        # Simulate model loading time
        time.sleep(load_time_ms / 1000.0)

    def encode(self, query: str, **kwargs) -> np.ndarray:
        """
        Encode query to embeddings.

        Returns deterministic embeddings based on query hash for reproducibility.
        """
        # Use query hash as seed for reproducible embeddings
        seed = hash(query) % (2**32)
        rng = np.random.RandomState(seed)

        # Return embeddings with correct dimension
        return rng.randn(self.embedding_dim).astype(np.float32)


class ColPaliStub(EncoderStub):
    """Stub for ColPali encoder (128-dim patch-based embeddings)"""

    def __init__(self, model_name: str = "vidore/colpali-v1.2", **kwargs):
        super().__init__(model_name, embedding_dim=128, load_time_ms=15.0)

    def encode(self, query: str, **kwargs) -> np.ndarray:
        """Return 16x128 patch embeddings"""
        seed = hash(query) % (2**32)
        rng = np.random.RandomState(seed)

        # ColPali returns patch-based embeddings (num_patches x dim)
        return rng.randn(16, 128).astype(np.float32)


class VideoPrismBaseStub(EncoderStub):
    """Stub for VideoPrism Base encoder (768-dim global embeddings)"""

    def __init__(self, model_name: str = "google/videoprism-base", **kwargs):
        super().__init__(model_name, embedding_dim=768, load_time_ms=20.0)

    def encode(self, query: str, **kwargs) -> np.ndarray:
        """Return 16x768 patch embeddings for consistency with ColPali"""
        seed = hash(query) % (2**32)
        rng = np.random.RandomState(seed)

        # Return patch format for consistency with other models
        return rng.randn(16, 768).astype(np.float32)


class VideoPrismLargeStub(EncoderStub):
    """Stub for VideoPrism Large encoder (1024-dim global embeddings)"""

    def __init__(self, model_name: str = "google/videoprism-large", **kwargs):
        super().__init__(model_name, embedding_dim=1024, load_time_ms=25.0)

    def encode(self, query: str, **kwargs) -> np.ndarray:
        """Return 16x1024 patch embeddings for consistency"""
        seed = hash(query) % (2**32)
        rng = np.random.RandomState(seed)

        return rng.randn(16, 1024).astype(np.float32)


class ColQwenStub(EncoderStub):
    """Stub for ColQwen encoder (128-dim patch-based embeddings)"""

    def __init__(self, model_name: str = "vidore/colqwen2-v1.0", **kwargs):
        super().__init__(model_name, embedding_dim=128, load_time_ms=15.0)

    def encode(self, query: str, **kwargs) -> np.ndarray:
        """Return 16x128 patch embeddings"""
        seed = hash(query) % (2**32)
        rng = np.random.RandomState(seed)

        return rng.randn(16, 128).astype(np.float32)


def create_encoder_stub(model_name: str, **kwargs) -> EncoderStub:
    """
    Factory function to create appropriate encoder stub based on model name.

    Args:
        model_name: Model identifier
        **kwargs: Additional arguments passed to encoder

    Returns:
        Appropriate encoder stub instance
    """
    model_name_lower = model_name.lower()

    if "colpali" in model_name_lower:
        return ColPaliStub(model_name, **kwargs)
    elif "videoprism" in model_name_lower:
        if "large" in model_name_lower:
            return VideoPrismLargeStub(model_name, **kwargs)
        else:
            return VideoPrismBaseStub(model_name, **kwargs)
    elif "colqwen" in model_name_lower:
        return ColQwenStub(model_name, **kwargs)
    else:
        # Default to 128-dim for unknown models
        return EncoderStub(model_name, embedding_dim=128, load_time_ms=10.0)


def mock_encoder_factory(profile_name: str, model_name: str, config: Dict[str, Any] = None):
    """
    Mock for QueryEncoderFactory.create_encoder that returns encoder stubs.

    This can be used with unittest.mock.patch to replace real encoder loading
    with lightweight stubs for testing.

    Example:
        with patch('cogniverse_agents.query.encoders.QueryEncoderFactory.create_encoder',
                   side_effect=mock_encoder_factory):
            # Test code that uses encoders
            pass
    """
    return create_encoder_stub(model_name)
