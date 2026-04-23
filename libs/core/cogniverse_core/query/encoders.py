"""Query encoders for different video processing profiles."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from cogniverse_core.common.models import get_or_load_model

if TYPE_CHECKING:
    from cogniverse_foundation.config.system_config import SystemConfig

logger = logging.getLogger(__name__)


class QueryEncoder(ABC):
    """Abstract base class for query encoders"""

    @abstractmethod
    def encode(self, query: str) -> np.ndarray:
        """Encode a text query into embeddings"""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension"""
        pass


class ColBERTQueryEncoder(QueryEncoder):
    """Query encoder for ColBERT models (text-only, per-token embeddings).

    ``embedding_dim`` must match the deployed Vespa schema's per-token tensor
    size. The factory reads it from the profile's ``schema_config.embedding_dim``.
    ``inference_service_url``, when set, switches to the remote HTTP loader
    pointing at the deployed service.
    """

    def __init__(
        self,
        model_name: str = "lightonai/LateOn",
        *,
        embedding_dim: int,
        inference_service_url: Optional[str] = None,
    ):
        config = {
            "embedding_model": model_name,
            "embedding_type": "multi_vector",
            "model_loader": "colbert",
        }
        if inference_service_url:
            config["remote_inference_url"] = inference_service_url

        self.model, _ = get_or_load_model(model_name, config, logger)
        self.embedding_dim = embedding_dim
        logger.info(
            f"Loaded ColBERT query encoder: {model_name} (dim={embedding_dim}, "
            f"remote={'yes' if inference_service_url else 'no'})"
        )

    def encode(self, query: str) -> np.ndarray:
        """Encode query to per-token embeddings."""
        result = self.model.encode([query], is_query=True)
        return np.array(result[0], dtype=np.float32)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class ColPaliQueryEncoder(QueryEncoder):
    """Query encoder for ColPali models"""

    def __init__(self, model_name: str = "vidore/colsmol-500m"):
        config = {
            "colpali_model": model_name,
            "embedding_type": "multi_vector",
            "model_loader": "colpali",
        }
        self.model, self.processor = get_or_load_model(model_name, config, logger)

        self.device = next(self.model.parameters()).device
        self.embedding_dim = 128
        logger.info(f"Loaded ColPali query encoder: {model_name} on {self.device}")

    def encode(self, query: str) -> np.ndarray:
        """Encode query to multi-vector embeddings"""
        batch_queries = self.processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)
        # Return as numpy array [num_patches, embedding_dim]
        return query_embeddings.cpu().numpy().squeeze(0)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class ColQwenQueryEncoder(QueryEncoder):
    """Query encoder for ColQwen models"""

    def __init__(self, model_name: str = "vidore/colqwen-omni-v0.1"):
        config = {
            "colpali_model": model_name,
            "embedding_type": "multi_vector",
            "model_loader": "colqwen",
        }
        self.model, self.processor = get_or_load_model(model_name, config, logger)

        self.device = next(self.model.parameters()).device
        self.embedding_dim = 128
        logger.info(f"Loaded ColQwen query encoder: {model_name} on {self.device}")

    def encode(self, query: str) -> np.ndarray:
        """Encode query to multi-vector embeddings"""
        batch_queries = self.processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)
        # Return as numpy array [num_patches, embedding_dim]
        return query_embeddings.cpu().numpy().squeeze(0)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class VideoPrismQueryEncoder(QueryEncoder):
    """Query encoder for VideoPrism models"""

    def __init__(self, model_name: str = "videoprism_public_v1_base_hf"):
        self.model_name = model_name

        # VideoPrism dimensions
        self.is_global = "lvt" in model_name.lower() or "global" in model_name.lower()
        if "large" in model_name:
            self.embedding_dim = 1024
            self.num_patches = 2048  # For patch-based models
        else:
            self.embedding_dim = 768
            self.num_patches = 4096  # For patch-based models

        # Use v2 model loader - it returns the videoprism loader instance
        config = {
            "colpali_model": model_name,
            "model_name": model_name,
            "embedding_type": "single_vector",
            "model_loader": "videoprism",
        }
        logger.info(f"Creating VideoPrism loader for model: {model_name}")
        self.videoprism_loader, _ = get_or_load_model(model_name, config, logger)
        logger.info(f"Got loader type: {type(self.videoprism_loader).__name__}")
        logger.info(
            f"Loader has encode_text: {hasattr(self.videoprism_loader, 'encode_text')}"
        )
        logger.info(
            f"Loader has text_tokenizer: {hasattr(self.videoprism_loader, 'text_tokenizer')}"
        )

        # Get text encoding components from the loader
        if hasattr(self.videoprism_loader, "text_tokenizer"):
            self.text_tokenizer = self.videoprism_loader.text_tokenizer
            self.forward_fn = (
                self.videoprism_loader.text_forward_fn
                if hasattr(self.videoprism_loader, "text_forward_fn")
                else None
            )
            logger.info(f"Loaded VideoPrism query encoder: {model_name}")
        else:
            logger.warning("VideoPrism loader doesn't have text encoder support")
            self.text_tokenizer = None
            self.forward_fn = None

    def encode(self, query: str) -> np.ndarray:
        """Encode text query to embeddings matching VideoPrism format"""
        if not hasattr(self.videoprism_loader, "encode_text"):
            raise RuntimeError(
                "VideoPrism loader doesn't have text encoding support. "
                "Ensure the model supports text encoding (LVT models)."
            )

        try:
            # Use the videoprism loader's text encoding method
            embeddings_np = self.videoprism_loader.encode_text(query)

            logger.info(f"Generated text embeddings shape: {embeddings_np.shape}")

            # For global embedding models, return as-is (single vector)
            if self.is_global:
                logger.info("Using global embedding format")
                # Ensure we return a 1D array for global models
                if len(embeddings_np.shape) > 1:
                    embeddings_np = embeddings_np.flatten()
                return embeddings_np

            # For patch-based models, the loader should handle the format
            return embeddings_np

        except Exception as e:
            logger.error(f"Failed to encode text query: {e}")
            raise

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


def _build_colbert_encoder(
    model_name: str,
    profile: str,
    profile_config: dict,
    system_config: "SystemConfig",
) -> "ColBERTQueryEncoder":
    """Construct a ColBERTQueryEncoder, wiring dim + remote URL from config."""
    schema_config = profile_config.get("schema_config", {})
    embedding_dim = schema_config.get("embedding_dim")
    if embedding_dim is None:
        raise ValueError(
            f"Profile {profile!r} routes to ColBERTQueryEncoder but is missing "
            f"schema_config.embedding_dim. Add it to configs/config.json."
        )

    service_name = profile_config.get("inference_service")
    inference_url: Optional[str] = None
    if service_name:
        service_urls = getattr(system_config, "inference_service_urls", {}) or {}
        inference_url = service_urls.get(service_name)
        if not inference_url:
            available = sorted(service_urls)
            raise ValueError(
                f"Profile {profile!r} specifies inference_service={service_name!r} "
                f"but no URL is configured. Deployed services: {available}. "
                f"Enable inference.{service_name} in the Helm values or remove "
                f"the inference_service field to fall back to local loading."
            )
    return ColBERTQueryEncoder(
        model_name,
        embedding_dim=embedding_dim,
        inference_service_url=inference_url,
    )


class QueryEncoderFactory:
    """Factory to create appropriate query encoder based on profile.

    Caches encoders by model_name so each model is loaded exactly once.
    """

    # (model_name, inference_service, embedding_dim) → QueryEncoder instance
    _encoder_cache: dict = {}

    @classmethod
    def create_encoder(
        cls,
        profile: str,
        model_name: Optional[str] = None,
        config: Optional["SystemConfig"] = None,
    ) -> QueryEncoder:
        """Create or retrieve cached query encoder for the given profile.

        Dynamically determines the encoder based on config.json backend.profiles.
        Returns cached encoder if model_name was already loaded.

        Args:
            profile: Profile name to use
            model_name: Optional model name override
            config: SystemConfig instance (required for dependency injection)
        """
        if config is None:
            raise ValueError(
                "config is required for QueryEncoderFactory.create_encoder(). "
                "Pass SystemConfig instance explicitly."
            )

        backend_config = config.get("backend", {})
        video_profiles = backend_config.get("profiles", {})

        # Check if profile exists in config
        if profile not in video_profiles:
            raise ValueError(
                f"Unknown profile: {profile}. Available profiles: {list(video_profiles.keys())}"
            )

        profile_config = video_profiles[profile]

        # Use provided model_name or get from config
        if not model_name:
            model_name = profile_config.get("embedding_model")
            if not model_name:
                raise ValueError(f"No embedding_model specified for profile: {profile}")

        # Cache key includes per-profile routing knobs so profiles sharing a
        # model but declaring different inference services or embedding dims
        # do not collapse onto the first-constructed encoder.
        schema_config = profile_config.get("schema_config", {}) or {}
        cache_key = (
            model_name,
            profile_config.get("inference_service"),
            schema_config.get("embedding_dim"),
        )

        if cache_key in cls._encoder_cache:
            logger.info(f"Reusing cached encoder for key: {cache_key}")
            return cls._encoder_cache[cache_key]

        encoder = cls._create_encoder_instance(
            model_name, profile, profile_config, config
        )
        cls._encoder_cache[cache_key] = encoder
        logger.info(f"Cached encoder for key {cache_key} ({type(encoder).__name__})")
        return encoder

    @staticmethod
    def _create_encoder_instance(
        model_name: str,
        profile: str,
        profile_config: dict,
        system_config: "SystemConfig",
    ) -> QueryEncoder:
        """Create a new encoder instance based on model loader / name.

        Preference order:
          1. ``profile_config["model_loader"]`` (authoritative when set).
          2. Model-name substring match.
          3. Profile-name substring match.
        """
        model_loader = profile_config.get("model_loader")

        if model_loader == "colbert":
            return _build_colbert_encoder(
                model_name, profile, profile_config, system_config
            )
        if model_loader == "colpali":
            return ColPaliQueryEncoder(model_name)
        if model_loader == "colqwen":
            return ColQwenQueryEncoder(model_name)
        if model_loader == "videoprism":
            return VideoPrismQueryEncoder(model_name)

        name = model_name.lower()
        if "colpali" in name or "colsmol" in name:
            return ColPaliQueryEncoder(model_name)
        if "colqwen" in name:
            return ColQwenQueryEncoder(model_name)
        if "videoprism" in name:
            return VideoPrismQueryEncoder(model_name)
        if "colbert" in name or "lateon" in name:
            return _build_colbert_encoder(
                model_name, profile, profile_config, system_config
            )

        profile_lower = profile.lower()
        if "colpali" in profile_lower:
            return ColPaliQueryEncoder(model_name)
        if "colqwen" in profile_lower:
            return ColQwenQueryEncoder(model_name)
        if "videoprism" in profile_lower:
            return VideoPrismQueryEncoder(model_name)
        if "colbert" in profile_lower or "document" in profile_lower:
            return _build_colbert_encoder(
                model_name, profile, profile_config, system_config
            )

        raise ValueError(
            f"Cannot determine encoder type for model: {model_name} in profile: {profile}"
        )

    @staticmethod
    def get_supported_profiles(config: Optional["SystemConfig"] = None) -> list:
        """Return list of supported profiles from config.json

        Args:
            config: SystemConfig instance (required for dependency injection)
        """
        if config is None:
            raise ValueError(
                "config is required for QueryEncoderFactory.get_supported_profiles(). "
                "Pass SystemConfig instance explicitly."
            )

        backend_config = config.get("backend", {})
        video_profiles = backend_config.get("profiles", {})
        return list(video_profiles.keys())
