#!/usr/bin/env python3
"""
Model Loaders - Handles loading of different embedding models.

Supports both local model loading and remote inference providers:
- Local: Load models using HuggingFace transformers
- Remote: Use inference endpoints (Infinity, Modal, custom APIs)

Remote providers allow offloading model inference to dedicated services,
reducing memory usage and enabling better scaling.
"""

import logging
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
import torch

from cogniverse_core.common.utils.retry import RetryConfig, retry_with_backoff


class ModelLoader(ABC):
    """Abstract base class for model loaders"""

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ):
        self.model_name = model_name
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self) -> Tuple[Any, Any]:
        """Load model and processor"""
        pass

    def get_device(self) -> str:
        """Get the appropriate device for the model"""
        # Check config override first
        if "device" in self.config:
            return self.config["device"]

        # Auto-detect device
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def get_dtype(self) -> torch.dtype:
        """Get appropriate dtype for the device"""
        device = self.get_device()
        if device == "cuda":
            return torch.bfloat16
        else:
            return torch.float32


class RemoteInferenceClient:
    """
    Client for remote model inference providers.

    Supports various inference endpoints:
    - Infinity: For ColPali and similar models
    - Modal: For custom deployed models
    - Custom REST APIs: Any HTTP endpoint following the standard format

    The client handles request/response formatting and error handling.
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.session = requests.Session()

        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    @retry_with_backoff(
        config=RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            exceptions=(requests.RequestException, ConnectionError, TimeoutError),
        )
    )
    def process_images(self, images: list, **kwargs) -> Dict[str, Any]:
        """
        Send images to remote inference endpoint with retry logic.

        Args:
            images: List of image paths or PIL images
            **kwargs: Additional parameters for the model

        Returns:
            Dict with inference results (embeddings, etc.)
        """
        try:
            # Prepare request data
            import base64
            import io

            from PIL import Image

            # Convert images to base64
            image_data = []
            for img in images:
                if isinstance(img, str) or isinstance(img, Path):
                    # Load from file
                    with Image.open(img) as pil_img:
                        buffer = io.BytesIO()
                        pil_img.save(buffer, format="PNG")
                        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        image_data.append(img_base64)
                elif hasattr(img, "save"):  # PIL Image
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    image_data.append(img_base64)
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")

            # Prepare request payload
            payload = {
                "images": image_data,
                "model": kwargs.get("model_name", "colpali"),
                **kwargs,
            }

            # Send request
            response = self.session.post(
                f"{self.endpoint_url}/v1/embeddings",
                json=payload,
                timeout=300,  # 5 minutes for large batches
            )
            response.raise_for_status()

            # Parse response
            result = response.json()

            # Convert embeddings to numpy array
            embeddings = np.array(result.get("embeddings", []))

            return {
                "embeddings": embeddings,
                "processing_time": result.get("processing_time", 0.0),
                "model": result.get("model"),
                "usage": result.get("usage", {}),
            }

        except Exception as e:
            self.logger.error(f"Remote inference failed: {e}")
            raise

    @retry_with_backoff(
        config=RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            exceptions=(
                requests.RequestException,
                ConnectionError,
                TimeoutError,
                subprocess.CalledProcessError,
            ),
        )
    )
    def process_video_segment(
        self, video_path: Path, start_time: float, end_time: float, **kwargs
    ) -> Dict[str, Any]:
        """
        Send video segment to remote inference endpoint with retry logic.

        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            **kwargs: Additional parameters

        Returns:
            Dict with inference results
        """
        try:
            import base64
            import subprocess
            import tempfile

            # Extract video segment to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                tmp_path = tmp_file.name

                # Use ffmpeg to extract segment
                duration = end_time - start_time
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-ss",
                    str(start_time),
                    "-t",
                    str(duration),
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-y",
                    tmp_path,
                ]

                subprocess.run(cmd, capture_output=True, check=True)

                # Read video file and encode to base64
                with open(tmp_path, "rb") as f:
                    video_base64 = base64.b64encode(f.read()).decode("utf-8")

                # Clean up temp file
                import os

                os.unlink(tmp_path)

            # Prepare request payload
            payload = {
                "video": video_base64,
                "start_time": start_time,
                "end_time": end_time,
                "model": kwargs.get("model_name", "videoprism"),
                **kwargs,
            }

            # Send request
            response = self.session.post(
                f"{self.endpoint_url}/v1/video/embeddings",
                json=payload,
                timeout=600,  # 10 minutes for video processing
            )
            response.raise_for_status()

            # Parse response
            result = response.json()

            # Convert embeddings to numpy array
            embeddings = np.array(result.get("embeddings", []))

            return {
                "embeddings": embeddings,
                "processing_time": result.get("processing_time", 0.0),
                "model": result.get("model"),
                "frames_processed": result.get("frames_processed", 0),
            }

        except Exception as e:
            self.logger.error(f"Remote video inference failed: {e}")
            raise


class RemoteColPaliLoader(ModelLoader):
    """
    Remote ColPali model loader using inference endpoints.

    Instead of loading the model locally, this loader sends requests
    to a remote inference service (e.g., Infinity).
    """

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(model_name, config, logger)

        # Remote inference config (fields defined in configs/config.json)
        self.remote_url = config.get("remote_inference_url")
        self.api_key = config.get("remote_inference_api_key")

        if not self.remote_url:
            raise ValueError("remote_inference_url required for remote model loader")

        self.client = RemoteInferenceClient(self.remote_url, self.api_key, self.logger)

    def load_model(self) -> Tuple[Any, Any]:
        """
        For remote inference, return the client as both model and processor.

        The client handles both preprocessing (processor) and inference (model).
        """
        self.logger.info(f"Initialized remote ColPali inference at {self.remote_url}")
        return self.client, self.client


class RemoteVideoPrismLoader(ModelLoader):
    """
    Remote VideoPrism model loader using inference endpoints.

    Sends video segments to remote service for processing.
    """

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(model_name, config, logger)

        self.remote_url = config.get("remote_inference_url")
        self.api_key = config.get("remote_inference_api_key")

        if not self.remote_url:
            raise ValueError("remote_inference_url required for remote model loader")

        self.client = RemoteInferenceClient(self.remote_url, self.api_key, self.logger)

    def load_model(self) -> Tuple[Any, Any]:
        """Return remote client for VideoPrism inference."""
        self.logger.info(
            f"Initialized remote VideoPrism inference at {self.remote_url}"
        )

        # Create a wrapper that matches VideoPrism interface
        class VideoPrismRemoteWrapper:
            def __init__(self, client):
                self.client = client

            def process_video_segment(
                self, video_path: Path, start_time: float, end_time: float
            ) -> Dict[str, Any]:
                result = self.client.process_video_segment(
                    video_path, start_time, end_time
                )
                # Convert to VideoPrism expected format
                return {
                    "embeddings_np": result["embeddings"],
                    "processing_time": result.get("processing_time", 0),
                }

        wrapper = VideoPrismRemoteWrapper(self.client)
        return wrapper, None  # No separate processor for VideoPrism


class ColPaliModelLoader(ModelLoader):
    """Loader for ColPali models"""

    @retry_with_backoff(
        config=RetryConfig(
            max_attempts=3,
            initial_delay=2.0,
            exceptions=(Exception,),  # Retry on any exception during model loading
        )
    )
    def load_model(self) -> Tuple[Any, Any]:
        """Load ColPali model and processor with retry logic"""
        try:
            self.logger.info(f"Loading ColPali model: {self.model_name}")
            from colpali_engine.models import ColIdefics3, ColIdefics3Processor

            device = self.get_device()
            dtype = self.get_dtype()

            self.logger.info(f"Using device: {device}, dtype: {dtype}")

            # Load model
            if device == "mps":
                # For MPS, load to CPU first then move
                model = ColIdefics3.from_pretrained(
                    self.model_name, torch_dtype=dtype, device_map="cpu"
                ).eval()
                model = model.to(device)
            else:
                model = ColIdefics3.from_pretrained(
                    self.model_name, torch_dtype=dtype, device_map=device
                ).eval()

            # Load processor
            processor = ColIdefics3Processor.from_pretrained(self.model_name)

            self.model = model
            self.processor = processor

            self.logger.info("ColPali model loaded successfully")
            return model, processor

        except Exception as e:
            self.logger.error(f"Failed to load ColPali model: {e}")
            raise  # Re-raise for retry


class ColQwenModelLoader(ModelLoader):
    """Loader for ColQwen models"""

    @retry_with_backoff(
        config=RetryConfig(
            max_attempts=3,
            initial_delay=2.0,
            exceptions=(Exception,),  # Retry on any exception during model loading
        )
    )
    def load_model(self) -> Tuple[Any, Any]:
        """Load ColQwen model and processor with retry logic"""
        try:
            self.logger.info(f"Loading ColQwen model: {self.model_name}")

            # Determine model variant
            if "omni" in self.model_name.lower():
                from colpali_engine.models import (
                    ColQwen2_5Omni,
                    ColQwen2_5OmniProcessor,
                )

                model_class = ColQwen2_5Omni
                processor_class = ColQwen2_5OmniProcessor
            else:
                from colpali_engine.models import ColQwen2, ColQwen2Processor

                model_class = ColQwen2
                processor_class = ColQwen2Processor

            device = self.get_device()
            dtype = self.get_dtype()

            # Force CPU for ColQwen on Mac due to MPS memory limitations
            import platform

            if platform.system() == "Darwin" and "colqwen" in self.model_name.lower():
                device = "cpu"
                self.logger.info(
                    "Forcing CPU for ColQwen on Mac due to MPS memory limitations"
                )

            # Check for flash attention
            attn_implementation = None
            if device != "mps" and device != "cpu":
                try:
                    from transformers.utils import is_flash_attn_2_available

                    if is_flash_attn_2_available():
                        attn_implementation = "flash_attention_2"
                except (ImportError, AttributeError):
                    pass

            self.logger.info(
                f"Using device: {device}, dtype: {dtype}, attention: {attn_implementation}"
            )

            # Load model
            model = model_class.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=device,
                attn_implementation=attn_implementation,
            ).eval()

            # Load processor
            # The Omni processor already handles audio, no need for custom processor
            processor = processor_class.from_pretrained(self.model_name)

            if "omni" in self.model_name.lower():
                self.logger.info("Using ColQwen2.5-Omni processor with audio support")

            self.model = model
            self.processor = processor

            self.logger.info("ColQwen model loaded successfully")
            return model, processor

        except Exception as e:
            self.logger.error(f"Failed to load ColQwen model: {e}")
            raise  # Re-raise for retry


class VideoPrismModelLoader(ModelLoader):
    """Loader for VideoPrism models with production fixes"""

    @retry_with_backoff(
        config=RetryConfig(
            max_attempts=3,
            initial_delay=2.0,
            exceptions=(Exception,),  # Retry on any exception during model loading
        )
    )
    def load_model(self) -> Tuple[Any, Any]:
        """Load VideoPrism model with JAX platform fix and text encoder support with retry logic"""
        try:
            self.logger.info(f"Loading VideoPrism model: {self.model_name}")

            # JAX_PLATFORM_NAME must be set before importing JAX (at startup boundary,
            # e.g. via JAX_PLATFORM_NAME=cpu env var or in __main__ before model loading).

            from .videoprism_loader import get_videoprism_loader as videoprism_loader

            # Get loader instance with proper config
            loader_config = self.config.copy()
            loader_config["model_name"] = self.model_name

            # Check if this is a global model that needs text encoder
            if "global" in self.model_name.lower() or "lvt" in self.model_name.lower():
                loader_config["load_text_encoder"] = True
                self.logger.info(
                    "Loading VideoPrism with text encoder for global embeddings"
                )

            loader = videoprism_loader(self.model_name, loader_config)
            loader.load_model()

            if loader_config.get("load_text_encoder"):
                if not hasattr(loader, "load_text_encoder"):
                    raise AttributeError(
                        f"VideoPrism loader {type(loader).__name__} does not implement "
                        "load_text_encoder(), which is required for global/lvt models."
                    )
                loader.load_text_encoder()
                self.logger.info("VideoPrism text encoder loaded successfully")

            self.model = loader
            self.processor = None  # VideoPrism doesn't use a separate processor

            self.logger.info("VideoPrism model loaded successfully")
            return loader, None

        except Exception as e:
            self.logger.error(f"Failed to load VideoPrism model: {e}")
            raise  # Re-raise for retry


class ColBERTModelLoader(ModelLoader):
    """Loader for ColBERT multi-vector models (e.g., lightonai/GTE-ModernColBERT-v1).

    Uses PyLate internally for per-token embedding extraction with the model's
    built-in linear projection (768 → 128 dims).
    """

    @retry_with_backoff(
        config=RetryConfig(
            max_attempts=3,
            initial_delay=2.0,
            exceptions=(Exception,),
        )
    )
    def load_model(self) -> Tuple[Any, Any]:
        """Load ColBERT model via PyLate and return (model, None)."""
        try:
            self.logger.info(f"Loading ColBERT model: {self.model_name}")
            from pylate import models as pylate_models

            device = self.get_device()
            self.logger.info(f"Using device: {device}")

            model = pylate_models.ColBERT(self.model_name, device=device)

            self.model = model
            self.logger.info("ColBERT model loaded successfully")
            return model, None

        except Exception as e:
            self.logger.error(f"Failed to load ColBERT model: {e}")
            raise


class ModelLoaderFactory:
    """Factory for creating model loaders based on model_loader key from config."""

    LOADERS: Dict[str, type] = {
        "colpali": ColPaliModelLoader,
        "colqwen": ColQwenModelLoader,
        "videoprism": VideoPrismModelLoader,
        "colbert": ColBERTModelLoader,
    }

    REMOTE_LOADERS: Dict[str, type] = {
        "colpali": RemoteColPaliLoader,
        "colqwen": RemoteColPaliLoader,
        "videoprism": RemoteVideoPrismLoader,
    }

    @staticmethod
    def create_loader(
        model_name: str, config: Dict[str, Any], logger: Optional[logging.Logger] = None
    ) -> ModelLoader:
        """
        Create model loader based on config["model_loader"].

        Raises ValueError if model_loader is missing or unrecognized.
        If remote_inference_url is present, creates a remote loader.
        """
        loader_key = config.get("model_loader")
        if not loader_key:
            raise ValueError(
                f"Config must contain 'model_loader' to select model loader. "
                f"Got config keys: {sorted(config.keys())}. "
                f"Valid model_loaders: {sorted(ModelLoaderFactory.LOADERS.keys())}"
            )

        if config.get("remote_inference_url"):
            if logger:
                logger.info(
                    f"Using remote inference for {model_name} at {config['remote_inference_url']}"
                )
            remote_cls = ModelLoaderFactory.REMOTE_LOADERS.get(loader_key)
            if not remote_cls:
                raise ValueError(
                    f"No remote loader for model_loader={loader_key!r}. "
                    f"Available: {sorted(ModelLoaderFactory.REMOTE_LOADERS.keys())}"
                )
            return remote_cls(model_name, config, logger)

        loader_cls = ModelLoaderFactory.LOADERS.get(loader_key)
        if not loader_cls:
            raise ValueError(
                f"Unknown model_loader={loader_key!r}. "
                f"Valid loaders: {sorted(ModelLoaderFactory.LOADERS.keys())}"
            )
        return loader_cls(model_name, config, logger)


# Global model cache to avoid reloading
_model_cache: Dict[str, Tuple[Any, Any]] = {}


def get_or_load_model(
    model_name: str,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    force_reload: bool = False,
) -> Tuple[Any, Any]:
    """
    Get model from cache or load it.

    For remote inference, caching is based on model_name + remote_url to
    allow different endpoints for the same model.
    """

    # Create cache key that includes remote URL if present
    cache_key = model_name
    if config.get("remote_inference_url"):
        cache_key = f"{model_name}@{config['remote_inference_url']}"

    if not force_reload and cache_key in _model_cache:
        if logger:
            logger.info(f"Using cached model: {cache_key}")
        return _model_cache[cache_key]

    # Create loader and load model
    loader = ModelLoaderFactory.create_loader(model_name, config, logger)
    model, processor = loader.load_model()

    # Cache the model
    _model_cache[cache_key] = (model, processor)

    return model, processor
