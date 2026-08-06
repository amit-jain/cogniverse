#!/usr/bin/env python3
"""
Model Loaders - Handles loading of different embedding models.

Supports both local model loading and remote inference providers:
- Local: Load models using HuggingFace transformers
- Remote: Use inference endpoints (Infinity, Modal, custom APIs)

Remote providers allow offloading model inference to dedicated services,
reducing memory usage and enabling better scaling.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import threading
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np
import requests

from cogniverse_core.common.utils.retry import RetryConfig, retry_with_backoff
from cogniverse_foundation.config.inference_auth import inference_headers


@runtime_checkable
class _CacheResource(Protocol):
    def _close(self) -> None: ...


def _close_cached_value(value: Any) -> None:
    resources = value if isinstance(value, tuple) else (value,)
    closed: set[int] = set()
    for resource in resources:
        identity = id(resource)
        if identity in closed:
            continue
        closed.add(identity)
        if isinstance(resource, _CacheResource):
            resource._close()


def _store_bounded_cache(
    cache: OrderedDict,
    key: Any,
    value: Any,
    *,
    capacity: int,
    label: str,
) -> None:
    displaced = None
    displaced_key = key
    restore_as_lru = False
    if key in cache:
        displaced = cache.pop(key)
    elif len(cache) >= capacity:
        displaced_key, displaced = cache.popitem(last=False)
        restore_as_lru = True

    if displaced is not None:
        try:
            _close_cached_value(displaced)
        except Exception as exc:
            cache[displaced_key] = displaced
            if restore_as_lru:
                cache.move_to_end(displaced_key, last=False)
            replacement_error = None
            try:
                _close_cached_value(value)
            except Exception as cleanup_exc:
                replacement_error = cleanup_exc
            detail = f"{label} eviction failed to close {displaced_key!r}: {exc}"
            if replacement_error is not None:
                detail += f"; replacement cleanup failed: {replacement_error}"
            raise RuntimeError(detail) from exc

    cache[key] = value
    cache.move_to_end(key)


def _resolved_inference_headers(
    endpoint_url: str,
    api_key: Optional[str],
) -> Mapping[str, str]:
    configured_headers = inference_headers(endpoint_url)
    if configured_headers:
        if api_key is not None:
            raise ValueError("api_key must not be supplied for a Modal endpoint")
        return configured_headers
    if api_key is None:
        return {}
    if not api_key or api_key != api_key.strip():
        raise ValueError("api_key must be a non-empty canonical value")
    return MappingProxyType({"Authorization": f"Bearer {api_key}"})


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

        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def get_dtype(self) -> "torch.dtype":  # noqa: F821
        """Get appropriate dtype for the device"""
        import torch

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
        *,
        _resolved_headers: Optional[Mapping[str, str]] = None,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.session = requests.Session()
        # Bounds the per-query text-encode POST on the search hot path. Image
        # ingestion keeps its own 1800s budget; a single text forward pass is
        # tens of ms, so 30s is a generous ceiling that fails fast under outage.
        self.query_encode_timeout_s: float = 30.0

        if _resolved_headers is not None:
            if api_key is not None:
                raise ValueError("api_key and _resolved_headers are mutually exclusive")
            resolved_headers = _resolved_headers
        else:
            resolved_headers = _resolved_inference_headers(
                self.endpoint_url, self.api_key
            )
        self.session.headers.update(resolved_headers)

        # Per-endpoint breaker: a down inference pod trips it so calls fail fast
        # (CircuitOpenError) instead of burning their retry budget each time.
        from cogniverse_core.common.utils.circuit_breaker import (
            BreakerConfig,
            CircuitBreaker,
        )

        self._breaker = CircuitBreaker.get(
            BreakerConfig(
                name=f"inference:{self.endpoint_url}",
                failure_threshold=5,
                reset_timeout_s=15.0,
                counted_exceptions=(
                    requests.RequestException,
                    ConnectionError,
                    TimeoutError,
                ),
            )
        )

    def _close(self) -> None:
        self.session.close()

    def process_images(self, images: list, **kwargs) -> Dict[str, Any]:
        """Send images to the inference endpoint, guarded by the breaker.

        The breaker wraps the retried call, so a down endpoint fails fast with
        CircuitOpenError once tripped instead of retrying every request.
        """
        return self._breaker.call(self._process_images_retried, images, **kwargs)

    @retry_with_backoff(
        config=RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            exceptions=(requests.RequestException, ConnectionError, TimeoutError),
        )
    )
    def _process_images_retried(self, images: list, **kwargs) -> Dict[str, Any]:
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

            # Send request. CPU-only ColPali on a single keyframe sequence
            # routinely runs 5-15 min on a laptop (no GPU); 300s wasn't
            # enough to clear even a small image batch and the retry loop
            # would burn 15 min before giving up. 1800s (30 min) covers
            # batched video keyframes; tighter budgets manifest as silent
            # data loss (0 documents fed) under load.
            response = self.session.post(
                f"{self.endpoint_url}/v1/embeddings",
                json=payload,
                timeout=1800,
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
            exceptions=(requests.RequestException, ConnectionError, TimeoutError),
        )
    )
    def process_images_vllm(self, images: list, **kwargs) -> Dict[str, Any]:
        """POST images to vLLM's ``/pooling`` endpoint (one request per image,
        issued concurrently) and return per-token multi-vector embeddings.

        vLLM 0.20+ doesn't register ``/v1/embeddings`` for ColPali's
        architecture — it stays on ``/pooling`` regardless of runner
        flags — and only the chat-style ``messages`` shape accepts
        image_url content (the ``input`` shape fails validation). The
        endpoint takes one image per request, so concurrent requests are
        the batching mechanism: vLLM's continuous batching coalesces them
        into shared forward passes server-side.
        """
        import base64
        import io
        from concurrent.futures import ThreadPoolExecutor

        from PIL import Image

        def encode_and_post(img) -> Dict[str, Any]:
            if isinstance(img, (str, Path)):
                with Image.open(img) as pil_img:
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            elif hasattr(img, "save"):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

            payload = {
                "model": kwargs.get("model_name", kwargs.get("model", "")),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            }
                        ],
                    }
                ],
            }
            response = self.session.post(
                f"{self.endpoint_url}/pooling",
                json=payload,
                timeout=1800,
            )
            response.raise_for_status()
            return response.json()

        if len(images) <= 1:
            results = [encode_and_post(img) for img in images]
        else:
            with ThreadPoolExecutor(max_workers=min(8, len(images))) as pool:
                results = list(pool.map(encode_and_post, images))

        per_image = [np.array(r.get("data", [{}])[0].get("data", [])) for r in results]
        result = results[-1] if results else {}

        embeddings = (
            per_image[0] if len(per_image) == 1 else np.array(per_image, dtype=object)
        )

        return {
            "embeddings": embeddings,
            "processing_time": result.get("processing_time", 0.0),
            "model": result.get("model"),
            "usage": result.get("usage", {}),
        }

    def process_queries_vllm(self, queries: list, **kwargs) -> Dict[str, Any]:
        """Encode text queries via vLLM, guarded by the endpoint breaker."""
        return self._breaker.call(self._process_queries_vllm_impl, queries, **kwargs)

    def _process_queries_vllm_impl(self, queries: list, **kwargs) -> Dict[str, Any]:
        """POST one text query at a time to vLLM's ``/pooling`` endpoint
        and return per-token multi-vector embeddings.

        Mirrors ``process_images_vllm`` but with ``type=text`` content
        for ColPali / ColQwen text-side query encoding.
        """
        per_query: list[np.ndarray] = []
        result: Dict[str, Any] = {}
        for query in queries:
            payload = {
                "model": kwargs.get("model_name", kwargs.get("model", "")),
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": str(query)}],
                    }
                ],
            }
            response = self.session.post(
                f"{self.endpoint_url}/pooling",
                json=payload,
                timeout=self.query_encode_timeout_s,
            )
            response.raise_for_status()
            result = response.json()
            tokens = result.get("data", [{}])[0].get("data", [])
            per_query.append(np.array(tokens))

        embeddings = (
            per_query[0] if len(per_query) == 1 else np.array(per_query, dtype=object)
        )

        return {
            "embeddings": embeddings,
            "processing_time": result.get("processing_time", 0.0),
            "model": result.get("model"),
            "usage": result.get("usage", {}),
        }

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
    Remote ColPali multi-vector loader.

    Talks to a vLLM ``ColPaliForRetrieval`` instance serving
    ``TomoroAI/tomoro-colqwen3-embed-4b`` (or any colpali-engine HF
    variant vLLM accepts) over the OpenAI-compatible /v1/embeddings
    endpoint with the ``token_embed`` pooling task. Returns per-token
    embeddings (shape ``[num_patches, 320]`` for tomoro-colqwen3-embed-4b).

    ``RemoteInferenceClient.process_images_vllm`` constructs the
    OpenAI-compat payload.
    """

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        *,
        _resolved_headers: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(model_name, config, logger)

        # Remote inference config (fields defined in configs/config.json)
        self.remote_url = config.get("remote_inference_url")
        self.api_key = config.get("remote_inference_api_key")

        if not self.remote_url:
            raise ValueError("remote_inference_url required for remote model loader")

        resolved_headers = _resolved_headers or _resolved_inference_headers(
            self.remote_url, self.api_key
        )
        self.client = RemoteInferenceClient(
            self.remote_url,
            logger=self.logger,
            _resolved_headers=resolved_headers,
        )
        # Bind the OpenAI-compat path so callers that only see the
        # client surface (model, processor) hit the vLLM contract.
        self.client.process_images = self.client.process_images_vllm  # type: ignore[method-assign]
        self.client.process_queries = self.client.process_queries_vllm  # type: ignore[method-assign]

    def load_model(self) -> Tuple[Any, Any]:
        """
        For remote inference, return the client as both model and processor.

        The client handles both preprocessing (processor) and inference (model).
        """
        self.logger.info(
            f"Initialized vLLM ColPali inference at {self.remote_url} "
            f"(model={self.model_name})"
        )
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
        *,
        _resolved_headers: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(model_name, config, logger)

        self.remote_url = config.get("remote_inference_url")
        self.api_key = config.get("remote_inference_api_key")

        if not self.remote_url:
            raise ValueError("remote_inference_url required for remote model loader")

        resolved_headers = _resolved_headers or _resolved_inference_headers(
            self.remote_url, self.api_key
        )
        self.client = RemoteInferenceClient(
            self.remote_url,
            logger=self.logger,
            _resolved_headers=resolved_headers,
        )

    def load_model(self) -> Tuple[Any, Any]:
        """Return remote client for VideoPrism inference."""
        self.logger.info(
            f"Initialized remote VideoPrism inference at {self.remote_url}"
        )

        # Create a wrapper that matches VideoPrism interface
        class VideoPrismRemoteWrapper:
            def __init__(self, client, model_name: str):
                self.client = client
                self.model_name = model_name

            def process_video_segment(
                self, video_path: Path, start_time: float, end_time: float
            ) -> Dict[str, Any]:
                result = self.client.process_video_segment(
                    video_path,
                    start_time,
                    end_time,
                    model_name=self.model_name,
                )
                # Convert to VideoPrism expected format
                return {
                    "embeddings_np": result["embeddings"],
                    "processing_time": result.get("processing_time", 0),
                }

            def _close(self) -> None:
                self.client._close()

        wrapper = VideoPrismRemoteWrapper(self.client, self.model_name)
        return wrapper, None  # No separate processor for VideoPrism


class RemoteColBERTLoader(ModelLoader):
    """Remote ColBERT model loader against a PyLate inference service.

    Returns a wrapper with an .encode() method matching pylate.models.ColBERT,
    so EmbeddingGeneratorImpl can use it interchangeably with local ColBERT.
    The service (``cogniverse_cli/modal_inference/servers/pylate.py``) runs
    the canonical PyLate tokenization and forward pass — query expansion with
    masked attention and the document punctuation skiplist — so the client
    sends raw text plus ``is_query`` and returns the per-token matrices
    unchanged.
    """

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        *,
        _resolved_headers: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(model_name, config, logger)
        self.remote_url = config.get("remote_inference_url")
        self.api_key = config.get("remote_inference_api_key")
        if not self.remote_url:
            raise ValueError("remote_inference_url required for remote ColBERT loader")
        self._resolved_headers = _resolved_headers or _resolved_inference_headers(
            self.remote_url, self.api_key
        )

    def load_model(self) -> Tuple[Any, Any]:
        """Return a ColBERT-compatible wrapper that calls the remote endpoint."""
        self.logger.info(f"Initialized remote ColBERT inference at {self.remote_url}")

        class ColBERTRemoteWrapper:
            def __init__(self, endpoint_url, headers, model_name):
                self.endpoint_url = endpoint_url.rstrip("/")
                self.model_name = model_name
                self.session = requests.Session()
                self.session.headers.update(headers)

            def _close(self) -> None:
                self.session.close()

            def encode(
                self,
                texts: list,
                is_query: bool = False,
                batch_size: int = 32,
                **kwargs,
            ) -> list:
                """Encode texts via the PyLate service's ``/pooling`` route.

                Matches ``pylate.models.ColBERT.encode()``. The service owns
                the canonical PyLate behavior for both directions — query
                expansion with masked attention and the document punctuation
                skiplist — so raw text plus ``is_query`` goes over the wire
                and the per-token matrices come back unchanged.
                """
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    chunk = texts[i : i + batch_size]
                    payload = {
                        "input": chunk,
                        "model": self.model_name,
                        "is_query": is_query,
                    }
                    try:
                        resp = self.session.post(
                            f"{self.endpoint_url}/pooling",
                            json=payload,
                            timeout=120,
                        )
                        resp.raise_for_status()
                    except requests.RequestException as exc:
                        raise RuntimeError(
                            "remote ColBERT pooling failed for model "
                            f"{self.model_name!r} at {self.endpoint_url}"
                        ) from exc
                    try:
                        response_payload = resp.json()
                    except ValueError as exc:
                        raise RuntimeError(
                            "remote ColBERT pooling returned invalid JSON for model "
                            f"{self.model_name!r} at {self.endpoint_url}"
                        ) from exc
                    if not isinstance(response_payload, dict):
                        raise RuntimeError(
                            "remote ColBERT pooling returned a non-object payload for "
                            f"model {self.model_name!r} at {self.endpoint_url}"
                        )
                    items = response_payload.get("data")
                    if not isinstance(items, list) or len(items) != len(chunk):
                        item_count = (
                            len(items) if isinstance(items, list) else "non-list"
                        )
                        raise RuntimeError(
                            "remote ColBERT pooling returned "
                            f"{item_count} embeddings for {len(chunk)} inputs from "
                            f"model {self.model_name!r} at {self.endpoint_url}"
                        )
                    for item in items:
                        if not isinstance(item, dict):
                            raise RuntimeError(
                                "remote ColBERT pooling returned a non-object embedding "
                                f"for model {self.model_name!r} at {self.endpoint_url}"
                            )
                        matrix = item.get("data")
                        if not isinstance(matrix, list):
                            raise RuntimeError(
                                "remote ColBERT pooling returned a non-list embedding for "
                                f"model {self.model_name!r} at {self.endpoint_url}"
                            )
                        all_embeddings.append(matrix)

                return all_embeddings

        wrapper = ColBERTRemoteWrapper(
            self.remote_url,
            self._resolved_headers,
            self.model_name,
        )
        return wrapper, None


class RemoteWhisperLoader(ModelLoader):
    """Remote Whisper ASR loader against a vLLM /v1/audio/transcriptions
    endpoint.

    The wrapper exposes ``.transcribe(audio_path, language=...)`` so it
    drops into AudioTranscriptionStrategy in place of an in-process
    faster-whisper model.
    """

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        *,
        _resolved_headers: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(model_name, config, logger)
        self.remote_url = config.get("remote_inference_url")
        self.api_key = config.get("remote_inference_api_key")
        if not self.remote_url:
            raise ValueError("remote_inference_url required for remote Whisper loader")
        self._resolved_headers = _resolved_headers or _resolved_inference_headers(
            self.remote_url, self.api_key
        )

    def load_model(self) -> Tuple[Any, Any]:
        """Return a Whisper-compatible wrapper that calls the remote endpoint."""
        self.logger.info(
            f"Initialized vLLM ASR inference at {self.remote_url} "
            f"(model={self.model_name})"
        )

        class WhisperRemoteWrapper:
            def __init__(self, endpoint_url, headers, model_name, logger):
                self.endpoint_url = endpoint_url.rstrip("/")
                self.model_name = model_name
                self.logger = logger
                self.session = requests.Session()
                self.session.headers.update(headers)

            def _close(self) -> None:
                self.session.close()

            def transcribe(
                self, audio_path: str, language: Optional[str] = None, **kwargs
            ) -> Dict[str, Any]:
                """Transcribe an audio file via vLLM /v1/audio/transcriptions.

                Mirrors the OpenAI Whisper API contract: multipart upload
                with ``file``, ``model``, optional ``language``.
                """
                with open(audio_path, "rb") as f:
                    files = {"file": (Path(audio_path).name, f, "audio/wav")}
                    data: Dict[str, Any] = {"model": self.model_name}
                    if language and language != "auto":
                        data["language"] = language
                    resp = self.session.post(
                        f"{self.endpoint_url}/v1/audio/transcriptions",
                        files=files,
                        data=data,
                        timeout=600,
                    )
                resp.raise_for_status()
                return resp.json()

        wrapper = WhisperRemoteWrapper(
            self.remote_url, self._resolved_headers, self.model_name, self.logger
        )
        return wrapper, None


_REMOTE_ONLY_MESSAGE = (
    "ColQwen3/Tomoro models are remote-only — serve via vLLM and set "
    "inference_service_url (profile inference_services.embedding). Local "
    "in-process loading is unsupported (requires transformers>=4.57, blocked "
    "by the pylate cap)."
)


def _is_colqwen3(model_name: str) -> bool:
    """True for ColQwen3/Tomoro models, which have no local loader path.

    Matches by model name (``colqwen3``/``tomoro``). The architecture is
    ``qwen3_vl``, which the pinned ``transformers`` (4.56.2, capped by pylate)
    cannot build and ``colpali_engine`` mis-maps to ``idefics3``.
    """
    name = model_name.lower()
    return "colqwen3" in name or "tomoro" in name


def is_remote_only_model(model_name: str) -> bool:
    """True when the model can only be served remotely (vLLM sidecar) —
    in-process loading raises. Public capability probe so callers can gate
    on the contract instead of matching the error message."""
    return _is_colqwen3(model_name)


def _raise_if_qwen3_vl(model_name: str, error: Exception) -> None:
    """Re-raise a local-load failure as the clear remote-only error when it is
    the ``qwen3_vl`` unsupported-architecture signature."""
    text = str(error).lower()
    if "qwen3_vl" in text or "qwen3_vl_text" in text:
        raise RuntimeError(_REMOTE_ONLY_MESSAGE) from error


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
        if _is_colqwen3(self.model_name):
            raise RuntimeError(_REMOTE_ONLY_MESSAGE)
        try:
            self.logger.info(f"Loading ColPali model: {self.model_name}")
            from colpali_engine.models import ColIdefics3, ColIdefics3Processor

            device = self.get_device()
            dtype = self.get_dtype()

            self.logger.info(f"Using device: {device}, dtype: {dtype}")

            # Load model — avoid device_map parameter which uses accelerate's
            # meta tensor dispatch and causes NotImplementedError on repeated
            # loads in the same process.
            model = ColIdefics3.from_pretrained(self.model_name, torch_dtype=dtype)
            model.eval()  # PyTorch evaluation mode (no dropout/batchnorm)
            if device != "cpu":
                model = model.to(device)

            # Load processor
            processor = ColIdefics3Processor.from_pretrained(self.model_name)

            self.model = model
            self.processor = processor

            self.logger.info("ColPali model loaded successfully")
            return model, processor

        except Exception as e:
            _raise_if_qwen3_vl(self.model_name, e)
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
        if _is_colqwen3(self.model_name):
            raise RuntimeError(_REMOTE_ONLY_MESSAGE)
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
            _raise_if_qwen3_vl(self.model_name, e)
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
            if (
                "global" in self.model_name.lower()
                or "_lvt_" in self.model_name.lower()
            ):
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
    """Loader for ColBERT multi-vector models (e.g., lightonai/Reason-ModernColBERT).

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
            try:
                from pylate import models as pylate_models
            except ImportError as e:
                raise ImportError(
                    "Local ColBERT loading requires the optional 'pylate' "
                    "dependency (install the project's [test] extra). For "
                    "production, serve ColBERT via vLLM by setting "
                    "inference_services.embedding on the profile (routes to "
                    "RemoteColBERTLoader)."
                ) from e

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
        "colbert": RemoteColBERTLoader,
        "whisper": RemoteWhisperLoader,
    }

    @staticmethod
    def create_loader(
        model_name: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        *,
        _resolved_headers: Optional[Mapping[str, str]] = None,
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
            resolved_headers = _resolved_headers or _resolved_inference_headers(
                config["remote_inference_url"],
                config.get("remote_inference_api_key"),
            )
            return remote_cls(
                model_name,
                config,
                logger,
                _resolved_headers=resolved_headers,
            )

        loader_cls = ModelLoaderFactory.LOADERS.get(loader_key)
        if not loader_cls:
            raise ValueError(
                f"Unknown model_loader={loader_key!r}. "
                f"Valid loaders: {sorted(ModelLoaderFactory.LOADERS.keys())}"
            )
        return loader_cls(model_name, config, logger)


# Global bounded LRU model cache to avoid reloading.
# Thread lock prevents concurrent from_pretrained calls which cause
# meta tensor corruption in accelerate's dispatch hooks.
_MODEL_CACHE_CAPACITY = 16
_GLINER_CACHE_CAPACITY = 16
_model_cache: OrderedDict[str, Tuple[Any, Any]] = OrderedDict()
_model_lock = threading.Lock()
# Weak entries live only while a load holds the lock, so distinct historical
# endpoint/credential keys cannot accumulate lock objects indefinitely.
_model_key_locks: weakref.WeakValueDictionary[str, threading.Lock] = (
    weakref.WeakValueDictionary()
)


def get_or_load_model(
    model_name: str,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    force_reload: bool = False,
) -> Tuple[Any, Any]:
    """
    Get model from cache or load it.

    Thread-safe: concurrent from_pretrained calls can corrupt PyTorch/accelerate
    global state (meta tensor dispatch hooks). The lock serializes model loads.
    """
    cache_key = model_name
    resolved_headers: Optional[Mapping[str, str]] = None
    if config.get("remote_inference_url"):
        resolved_headers = _resolved_inference_headers(
            config["remote_inference_url"],
            config.get("remote_inference_api_key"),
        )
        authorization = resolved_headers.get("Authorization", "")
        credential_fingerprint = hashlib.sha256(authorization.encode()).hexdigest()
        cache_key = (
            f"{model_name}@{config['remote_inference_url']}|{credential_fingerprint}"
        )

    def _cached_entry():
        """Return the valid cached pair, evicting invalid entries. Caller
        must hold ``_model_lock``."""
        if force_reload or cache_key not in _model_cache:
            return None
        cached_model, cached_processor = _model_cache[cache_key]
        try:
            if hasattr(cached_model, "parameters"):
                param = next(cached_model.parameters(), None)
                if param is not None and param.device.type == "meta":
                    if logger:
                        logger.warning(
                            f"Cached model {cache_key} has meta tensors, reloading"
                        )
                    del _model_cache[cache_key]
                    return None
            if logger:
                logger.info(f"Using cached model: {cache_key}")
            _model_cache.move_to_end(cache_key)
            return cached_model, cached_processor
        except (StopIteration, RuntimeError):
            if logger:
                logger.warning(f"Cached model {cache_key} invalid, reloading")
            del _model_cache[cache_key]
            return None

    # Two-level locking: the global lock only guards the cache dict and the
    # per-key lock registry; the (minutes-long) load itself runs under a
    # per-key lock so a cold load of one model does not block cache hits
    # and loads of every other model.
    with _model_lock:
        cached = _cached_entry()
        if cached is not None:
            return cached
        key_lock = _model_key_locks.setdefault(cache_key, threading.Lock())

    with key_lock:
        with _model_lock:
            cached = _cached_entry()
            if cached is not None:
                return cached

        # Load outside the global lock — only same-key callers wait here.
        loader = ModelLoaderFactory.create_loader(
            model_name,
            config,
            logger,
            _resolved_headers=resolved_headers,
        )
        model, processor = loader.load_model()

        with _model_lock:
            _store_bounded_cache(
                _model_cache,
                cache_key,
                (model, processor),
                capacity=_MODEL_CACHE_CAPACITY,
                label="model cache",
            )
        return model, processor


# Module-level GLiNER cache. Local models share by model/device; remote clients
# additionally include endpoint and a one-way credential fingerprint.
_gliner_cache: OrderedDict[Tuple[str, str, str, str], Any] = OrderedDict()


class RemoteGlinerClient:
    """HTTP client for the GLiNER inference service.

    Exposes the same ``predict_entities(text, labels, threshold=...)``
    surface the in-process ``GLiNER`` class does so GatewayAgent can
    treat local + remote loaders interchangeably.
    """

    _ENTITY_FIELDS = ("text", "label", "score", "start", "end")

    def __init__(
        self,
        url: str,
        model_name: str,
        api_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        *,
        _resolved_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._model_name = model_name
        self._logger = logger or logging.getLogger(__name__)
        self._session = requests.Session()
        if _resolved_headers is not None:
            if api_key is not None:
                raise ValueError("api_key and _resolved_headers are mutually exclusive")
            resolved_headers = _resolved_headers
        else:
            resolved_headers = _resolved_inference_headers(self._url, api_key)
        self._session.headers.update(resolved_headers)

    def _close(self) -> None:
        self._session.close()

    def _validated_entities(self, data: Any) -> List[Dict[str, Any]]:
        endpoint = f"{self._url}/predict_entities"
        prefix = f"Remote GLiNER response from {endpoint}"
        if not isinstance(data, dict):
            raise ValueError(f"{prefix} must be a JSON object")
        if "entities" not in data:
            raise ValueError(f"{prefix} must contain 'entities'")

        entities = data["entities"]
        if not isinstance(entities, list):
            raise ValueError(f"{prefix} 'entities' must be a list")

        expected_fields = set(self._ENTITY_FIELDS)
        for index, entity in enumerate(entities):
            item_prefix = f"{prefix} entity at index {index}"
            if not isinstance(entity, dict):
                raise ValueError(f"{item_prefix} must be an object")
            if set(entity) != expected_fields:
                raise ValueError(
                    f"{item_prefix} must have exactly fields "
                    f"{list(self._ENTITY_FIELDS)}"
                )
            for field in ("text", "label"):
                if not isinstance(entity[field], str):
                    raise ValueError(f"{item_prefix} field '{field}' must be a string")
            if isinstance(entity["score"], bool) or not isinstance(
                entity["score"], (int, float)
            ):
                raise ValueError(f"{item_prefix} field 'score' must be a number")
            for field in ("start", "end"):
                value = entity[field]
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int)
                ):
                    raise ValueError(
                        f"{item_prefix} field '{field}' must be an integer or null"
                    )

        return entities

    def predict_entities(
        self, text: str, labels: List[str], threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        payload = {
            "text": text,
            "labels": labels,
            "threshold": threshold,
            "model": self._model_name,
        }
        try:
            resp = self._session.post(
                f"{self._url}/predict_entities",
                json=payload,
                # First request per (inference service, model) cold-loads HF
                # weights; on CPU that takes ~30-60s for medium and
                # ~90s for large. Subsequent requests are sub-second.
                timeout=240,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            # An inference service outage (down / 5xx / timeout / connection
            # reset) is NOT a genuine "no entities" result. Swallowing it to []
            # made the gateway's entity_extraction_failed branch unreachable on
            # the remote path, so an outage read as a low-confidence route. Raise
            # so the caller can flag the routing decision as service-degraded; a
            # real HTTP-200 with an empty entity list still returns [].
            self._logger.error(
                "Remote GLiNER prediction failed (url=%s): %s", self._url, exc
            )
            raise
        return self._validated_entities(data)


def get_or_load_gliner(
    model_name: str,
    logger: Optional[logging.Logger] = None,
    inference_url: Optional[str] = None,
    device: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """Return a cached GLiNER instance, loading once per model name.

    When ``inference_url`` is provided, return a ``RemoteGlinerClient``
    that POSTs to the GLiNER inference service implemented by
    ``cogniverse_cli.modal_inference.servers.gliner``. Modal URLs obtain their
    bearer credential from ``COGNIVERSE_INFERENCE_API_KEY``; ``api_key``
    authenticates non-Modal endpoints. The credential participates in cache
    isolation only through a one-way fingerprint. Local mode loads via
    ``gliner.GLiNER.from_pretrained`` and requires the heavy torch stack the
    runtime image normally omits.

    ``device`` moves a locally-loaded model onto the given torch device
    (e.g. ``"cuda"``); ``"cpu"`` / None leaves it where from_pretrained put
    it. Ignored for the remote client (the inference service owns its device).

    Local load and device-placement failures raise ``RuntimeError`` with the
    exact model and requested device rather than returning an unusable model.
    """
    if api_key is not None and not inference_url:
        raise ValueError("api_key requires inference_url")
    authorization = ""
    resolved_headers: Optional[Mapping[str, str]] = None
    if inference_url:
        resolved_headers = _resolved_inference_headers(inference_url, api_key)
        authorization = resolved_headers.get("Authorization", "")
    credential_fingerprint = hashlib.sha256(authorization.encode()).hexdigest()
    cache_key = (
        model_name,
        inference_url or "_local_",
        device or "default",
        credential_fingerprint,
    )
    with _model_lock:
        cached = _gliner_cache.get(cache_key)
        if cached is not None:
            _gliner_cache.move_to_end(cache_key)
            if logger:
                logger.info(
                    f"Using cached GLiNER model: {model_name} "
                    f"({'remote' if inference_url else 'local'})"
                )
            return cached
        if inference_url:
            instance = RemoteGlinerClient(
                inference_url,
                model_name,
                logger=logger,
                _resolved_headers=resolved_headers,
            )
            _store_bounded_cache(
                _gliner_cache,
                cache_key,
                instance,
                capacity=_GLINER_CACHE_CAPACITY,
                label="GLiNER cache",
            )
            if logger:
                logger.info(
                    f"Initialised remote GLiNER client: {model_name} "
                    f"via {inference_url}"
                )
            return instance
        try:
            from gliner import GLiNER

            if logger:
                logger.info(f"Loading GLiNER model: {model_name}")
            instance = GLiNER.from_pretrained(model_name)
            if instance is None:
                raise RuntimeError("GLiNER.from_pretrained returned no model")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local GLiNER model '{model_name}': {exc}"
            ) from exc

        if device and device.lower() != "cpu":
            try:
                moved_instance = instance.to(device)
                if moved_instance is None:
                    raise RuntimeError("model.to returned no model")
                instance = moved_instance
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to move local GLiNER model '{model_name}' to device "
                    f"'{device}': {exc}"
                ) from exc
        _store_bounded_cache(
            _gliner_cache,
            cache_key,
            instance,
            capacity=_GLINER_CACHE_CAPACITY,
            label="GLiNER cache",
        )
        if logger:
            logger.info(f"GLiNER loaded: {model_name}")
        return instance
