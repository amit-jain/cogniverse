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

import logging
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

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
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.session = requests.Session()
        # Bounds the per-query text-encode POST on the search hot path. Image
        # ingestion keeps its own 1800s budget; a single text forward pass is
        # tens of ms, so 30s is a generous ceiling that fails fast under outage.
        self.query_encode_timeout_s: float = 30.0

        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"

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
    ):
        super().__init__(model_name, config, logger)

        # Remote inference config (fields defined in configs/config.json)
        self.remote_url = config.get("remote_inference_url")
        self.api_key = config.get("remote_inference_api_key")

        if not self.remote_url:
            raise ValueError("remote_inference_url required for remote model loader")

        self.client = RemoteInferenceClient(self.remote_url, self.api_key, self.logger)
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


class RemoteColBERTLoader(ModelLoader):
    """Remote ColBERT model loader using text embedding inference endpoints.

    Returns a wrapper with an .encode() method matching pylate.models.ColBERT,
    so EmbeddingGeneratorImpl can use it interchangeably with local ColBERT.
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
            raise ValueError("remote_inference_url required for remote ColBERT loader")

    def load_model(self) -> Tuple[Any, Any]:
        """Return a ColBERT-compatible wrapper that calls the remote endpoint."""
        self.logger.info(f"Initialized remote ColBERT inference at {self.remote_url}")

        class ColBERTRemoteWrapper:
            def __init__(
                self,
                endpoint_url,
                api_key,
                model_name,
                logger,
                query_prefix="[Q] ",
                document_prefix="[D] ",
            ):
                self.endpoint_url = endpoint_url.rstrip("/")
                self.model_name = model_name
                self.logger = logger
                self.query_prefix = query_prefix
                self.document_prefix = document_prefix
                self.session = requests.Session()
                if api_key:
                    self.session.headers["Authorization"] = f"Bearer {api_key}"
                self._tokenizer = None
                self._skiplist_ids = None

            def _load_tokenizer(self):
                """Lazily load the model tokenizer and build the document skiplist.

                The ``[Q] ``/``[D] `` markers are single tokens in the tokenizer
                vocabulary, so prepending the literal text reproduces pylate's
                marker insertion exactly. The document skiplist is the punctuation
                token ids pylate drops from document embeddings via
                ``ColBERT.skiplist_mask``.
                """
                if self._tokenizer is not None:
                    return
                import string

                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._skiplist_ids = {
                    self._tokenizer.convert_tokens_to_ids(word)
                    for word in string.punctuation
                }

            def encode(
                self,
                texts: list,
                is_query: bool = False,
                batch_size: int = 32,
                **kwargs,
            ) -> list:
                """Encode texts via remote /pooling, reproducing pylate's contract.

                Matches ``pylate.models.ColBERT.encode()``. The ``[Q] ``/``[D] ``
                marker is prepended client-side as literal text (each marker is a
                single vocabulary token, identical to pylate's marker insertion);
                vLLM ``/pooling`` rejects unknown fields such as ``is_query``.

                For documents (``is_query=False``) pylate drops punctuation tokens
                from the per-token matrix (``ColBERT.skiplist_mask``); vLLM
                ``/pooling`` returns one embedding per token in tokenizer order, so
                we drop the same rows client-side. Queries keep all tokens.
                """
                self._load_tokenizer()
                prefix = self.query_prefix if is_query else self.document_prefix
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    chunk = texts[i : i + batch_size]
                    batch = [f"{prefix}{t}" for t in chunk]
                    resp = self.session.post(
                        f"{self.endpoint_url}/pooling",
                        json={"input": batch, "model": self.model_name},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    items = resp.json().get("data", [])
                    for text, item in zip(batch, items):
                        matrix = item.get("data", item.get("embedding", []))
                        if not is_query:
                            matrix = self._drop_skiplist_rows(text, matrix)
                        all_embeddings.append(matrix)

                return all_embeddings

            def _drop_skiplist_rows(self, prefixed_text, matrix):
                """Remove punctuation-token rows from a document per-token matrix.

                ``/pooling`` returns embeddings aligned with the tokenizer's
                ``input_ids`` (CLS first, SEP last, specials included), so the
                token ids re-derived from the same prefixed string select exactly
                the rows pylate's skiplist mask removes.
                """
                token_ids = self._tokenizer(prefixed_text, add_special_tokens=True)[
                    "input_ids"
                ]
                if len(token_ids) != len(matrix):
                    self.logger.warning(
                        "Token/embedding length mismatch for remote ColBERT "
                        "document (%d ids vs %d rows); returning unmasked matrix",
                        len(token_ids),
                        len(matrix),
                    )
                    return matrix
                return [
                    row
                    for tid, row in zip(token_ids, matrix)
                    if tid not in self._skiplist_ids
                ]

        wrapper = ColBERTRemoteWrapper(
            self.remote_url,
            self.api_key,
            self.model_name,
            self.logger,
            query_prefix=self.config.get("query_prefix", "[Q] "),
            document_prefix=self.config.get("document_prefix", "[D] "),
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
    ):
        super().__init__(model_name, config, logger)
        self.remote_url = config.get("remote_inference_url")
        self.api_key = config.get("remote_inference_api_key")
        if not self.remote_url:
            raise ValueError("remote_inference_url required for remote Whisper loader")

    def load_model(self) -> Tuple[Any, Any]:
        """Return a Whisper-compatible wrapper that calls the remote endpoint."""
        self.logger.info(
            f"Initialized vLLM ASR inference at {self.remote_url} "
            f"(model={self.model_name})"
        )

        class WhisperRemoteWrapper:
            def __init__(self, endpoint_url, api_key, model_name, logger):
                self.endpoint_url = endpoint_url.rstrip("/")
                self.model_name = model_name
                self.logger = logger
                self.session = requests.Session()
                if api_key:
                    self.session.headers["Authorization"] = f"Bearer {api_key}"

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
            self.remote_url, self.api_key, self.model_name, self.logger
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


# Global model cache to avoid reloading.
# Thread lock prevents concurrent from_pretrained calls which cause
# meta tensor corruption in accelerate's dispatch hooks.
import threading

_model_cache: Dict[str, Tuple[Any, Any]] = {}
_model_lock = threading.Lock()
# One lock per cache key so cold loads don't serialize unrelated models.
_model_key_locks: Dict[str, threading.Lock] = {}


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
    if config.get("remote_inference_url"):
        cache_key = f"{model_name}@{config['remote_inference_url']}"

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
            return cached_model, cached_processor
        except (StopIteration, RuntimeError):
            if logger:
                logger.warning(f"Cached model {cache_key} invalid, reloading")
            del _model_cache[cache_key]
            return None

    # Two-level locking: the global lock only guards the cache dict and the
    # per-key lock registry; the (minutes-long) load itself runs under a
    # per-key lock so a cold load of one model no longer blocks cache hits
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
        loader = ModelLoaderFactory.create_loader(model_name, config, logger)
        model, processor = loader.load_model()

        with _model_lock:
            _model_cache[cache_key] = (model, processor)
        return model, processor


# Module-level GLiNER cache. Loaded once per (model_name) and reused across
# every agent instance — gateway, entity_extraction, routing all want the
# same 1.4GB model. Without this, the dispatcher's per-request agent
# instantiation would reload it every call, blowing memory through the
# suite (PyTorch's caching allocator never returns memory to the OS).
_gliner_cache: Dict[str, Any] = {}


class RemoteGlinerClient:
    """HTTP client wrapping the deploy/gliner sidecar.

    Exposes the same ``predict_entities(text, labels, threshold=...)``
    surface the in-process ``GLiNER`` class does so GatewayAgent can
    treat local + remote loaders interchangeably.
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._model_name = model_name
        self._logger = logger or logging.getLogger(__name__)
        self._session = requests.Session()

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
                # First request per (sidecar, model) cold-loads HF
                # weights; on CPU that takes ~30-60s for medium and
                # ~90s for large. Subsequent requests are sub-second.
                timeout=240,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            # A sidecar outage (down / 5xx / timeout / connection reset) is NOT
            # a genuine "no entities" result — swallowing it to [] made the
            # gateway's entity_extraction_failed degrade branch unreachable on
            # the remote path, so an outage read as a low-confidence route. Raise
            # so the caller can flag the routing decision as sidecar-degraded; a
            # real HTTP-200 with an empty entity list still returns [].
            self._logger.error(
                "Remote GLiNER prediction failed (url=%s): %s", self._url, exc
            )
            raise
        return list(data.get("entities", []) or [])


def get_or_load_gliner(
    model_name: str,
    logger: Optional[logging.Logger] = None,
    inference_url: Optional[str] = None,
    device: Optional[str] = None,
) -> Optional[Any]:
    """Return a cached GLiNER instance, loading once per model name.

    When ``inference_url`` is provided, return a ``RemoteGlinerClient``
    that POSTs to the deploy/gliner sidecar. Local mode loads via
    ``gliner.GLiNER.from_pretrained`` and requires the heavy torch
    stack the runtime image normally omits.

    ``device`` moves a locally-loaded model onto the given torch device
    (e.g. ``"cuda"``); ``"cpu"`` / None leaves it where from_pretrained put
    it. Ignored for the remote client (the sidecar owns its device).

    Returns None on load failure (callers must handle missing extractor).
    """
    cache_key = (model_name, inference_url or "_local_", device or "default")
    with _model_lock:
        cached = _gliner_cache.get(cache_key)
        if cached is not None:
            if logger:
                logger.info(
                    f"Using cached GLiNER model: {model_name} "
                    f"({'remote' if inference_url else 'local'})"
                )
            return cached
        if inference_url:
            instance = RemoteGlinerClient(inference_url, model_name, logger=logger)
            _gliner_cache[cache_key] = instance
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
            if device and device.lower() != "cpu":
                try:
                    instance = instance.to(device)
                except Exception as exc:
                    if logger:
                        logger.warning(
                            f"GLiNER device move to {device} failed, staying on "
                            f"default device: {exc}"
                        )
            _gliner_cache[cache_key] = instance
            if logger:
                logger.info(f"GLiNER loaded: {model_name}")
            return instance
        except Exception as exc:
            if logger:
                logger.error(f"GLiNER load failed for {model_name}: {exc}")
            return None
