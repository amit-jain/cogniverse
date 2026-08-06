#!/usr/bin/env python3
"""
Audio Embedding Generator

Generates acoustic and semantic embeddings for audio content:
- Acoustic embeddings (512-dim): CLAP model for audio features
- Semantic embeddings (768-dim): Sentence transformers for transcript semantics
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from cogniverse_core.common.models.semantic_embedder import get_semantic_embedder
from cogniverse_foundation.config.inference_auth import inference_headers

logger = logging.getLogger(__name__)


def _canonical_bearer_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    if not headers:
        return {}
    if set(headers) != {"Authorization"}:
        raise ValueError("clap_headers must contain only Authorization")
    authorization = headers["Authorization"]
    scheme, separator, token = authorization.partition(" ")
    if scheme != "Bearer" or not separator or not token or token != token.strip():
        raise ValueError("clap_headers Authorization must be a canonical bearer value")
    return {"Authorization": authorization}


class AudioEmbeddingGenerator:
    """Generate acoustic and semantic embeddings for audio"""

    def __init__(
        self,
        clap_model: str = "laion/clap-htsat-unfused",
        semantic_model: Optional[str] = None,
        clap_endpoint_url: Optional[str] = None,
        clap_headers: Optional[Dict[str, str]] = None,
        *,
        _resolved_headers: Optional[Mapping[str, str]] = None,
    ):
        """
        Initialize audio embedding generator

        Args:
            clap_model: CLAP model for acoustic embeddings (512-dim)
            semantic_model: Override name for the semantic embedder. When
                None, the shared ``get_semantic_embedder()`` factory picks
                a default — DenseOn served by the denseon sidecar when
                ``COGNIVERSE_SEMANTIC_EMBED_URL`` is set, otherwise an
                in-process ``all-mpnet-base-v2``.
            clap_endpoint_url: URL of the clap_embed sidecar. When set,
                acoustic embeddings route over HTTP instead of loading
                CLAP in-process — the deployed runtime image ships no
                torch, so in-cluster this is the only working path.
            clap_headers: Canonical bearer headers for a non-Modal endpoint.
                Modal endpoints use ``COGNIVERSE_INFERENCE_API_KEY`` and reject
                caller-supplied headers.
            _resolved_headers: Immutable credentials already resolved by an
                owning dependency object. Internal callers only.
        """
        self._clap_model_name = clap_model
        self._semantic_model_name = semantic_model
        self._clap_endpoint_url = (
            clap_endpoint_url.rstrip("/") if clap_endpoint_url else None
        )
        if _resolved_headers is not None:
            if clap_headers is not None:
                raise ValueError(
                    "clap_headers and _resolved_headers are mutually exclusive"
                )
            if self._clap_endpoint_url is None:
                raise ValueError("_resolved_headers requires clap_endpoint_url")
            self._clap_headers = _resolved_headers
        else:
            explicit_headers = _canonical_bearer_headers(clap_headers)
            if explicit_headers and self._clap_endpoint_url is None:
                raise ValueError("clap_headers requires clap_endpoint_url")
            configured_headers = (
                inference_headers(self._clap_endpoint_url)
                if self._clap_endpoint_url
                else {}
            )
            if configured_headers and clap_headers is not None:
                raise ValueError(
                    "clap_headers must not be supplied for a Modal endpoint"
                )
            self._clap_headers = configured_headers or explicit_headers

        # Lazy loading
        self._clap_model = None
        self._clap_processor = None
        self._semantic_model = None
        self._http_client = None
        self._clap_model_lock = threading.Lock()
        self._semantic_model_lock = threading.Lock()
        self._http_client_lock = threading.Lock()

        logger.info("AudioEmbeddingGenerator initialized")
        logger.info(f"  Acoustic model: {clap_model}")
        logger.info(
            "  Semantic model: %s",
            semantic_model if semantic_model else "(resolved lazily via env)",
        )

    def _get_http_client(self):
        """One pooled httpx.Client per generator — a bare httpx.post per
        segment re-handshakes TCP/TLS for every call in a batch. The
        generous timeout absorbs the sidecar's one-time model cold-load."""
        with self._http_client_lock:
            if self._http_client is None:
                import httpx

                self._http_client = httpx.Client(
                    timeout=600.0,
                    headers=self._clap_headers,
                )
            return self._http_client

    def close(self) -> None:
        """Close the pooled sidecar client; the next remote call rebuilds it."""
        with self._http_client_lock:
            if self._http_client is not None:
                self._http_client.close()
                self._http_client = None

    @property
    def clap_model(self):
        """Lazy load CLAP model"""
        if self._clap_model is None:
            with self._clap_model_lock:
                if self._clap_model is None:
                    logger.info(f"Loading CLAP model: {self._clap_model_name}")
                    try:
                        from transformers import ClapModel, ClapProcessor

                        model = ClapModel.from_pretrained(self._clap_model_name)
                        processor = ClapProcessor.from_pretrained(self._clap_model_name)
                        model.eval()
                    except Exception as e:
                        logger.error(f"Failed to load CLAP model: {e}")
                        raise
                    self._clap_model = model
                    self._clap_processor = processor
                    logger.info("✅ CLAP model loaded")
        return self._clap_model

    @property
    def clap_processor(self):
        """Get CLAP processor (triggers model load)"""
        _ = self.clap_model  # Trigger load
        return self._clap_processor

    @property
    def semantic_model(self):
        """Lazy-resolve the shared semantic embedder.

        Returns a cached embedder — remote LM provider or local
        SentenceTransformer — from the module-level factory. Every
        agent that calls this shares one backend instance instead of
        loading an independent ~400MB model per call site.
        """
        if self._semantic_model is None:
            with self._semantic_model_lock:
                if self._semantic_model is None:
                    try:
                        semantic_model = get_semantic_embedder(
                            model_name=self._semantic_model_name
                        )
                    except Exception as e:
                        logger.error(f"Failed to initialize semantic embedder: {e}")
                        raise
                    self._semantic_model = semantic_model
                    logger.info("✅ Semantic embedder ready")
        return self._semantic_model

    def generate_acoustic_embedding(
        self,
        audio_path: Optional[Path] = None,
        audio_array: Optional[np.ndarray] = None,
        sample_rate: int = 48000,
    ) -> np.ndarray:
        """
        Generate acoustic embedding using CLAP

        Args:
            audio_path: Path to audio file (if provided)
            audio_array: Audio array (if provided)
            sample_rate: Sample rate of audio

        Returns:
            512-dim acoustic embedding
        """
        if audio_path is None and audio_array is None:
            raise ValueError("Must provide either audio_path or audio_array")

        if self._clap_endpoint_url:
            return self._remote_acoustic_embedding(
                audio_path=audio_path,
                audio_array=audio_array,
                sample_rate=sample_rate,
            )

        try:
            # Load audio if path provided
            if audio_path is not None:
                import librosa

                audio_array, sample_rate = librosa.load(
                    str(audio_path), sr=sample_rate, mono=True
                )

            # Process with CLAP
            inputs = self.clap_processor(
                audios=audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )

            import torch

            with torch.no_grad():
                audio_embeds = self.clap_model.get_audio_features(**inputs)

            # Convert to numpy and flatten to 512 dims
            embedding = audio_embeds.squeeze().cpu().numpy()

            # Ensure exactly 512 dimensions
            if embedding.shape[0] != 512:
                logger.warning(
                    f"CLAP embedding has {embedding.shape[0]} dims, expected 512"
                )
                if embedding.shape[0] > 512:
                    embedding = embedding[:512]
                else:
                    # Pad with zeros
                    padding = np.zeros(512 - embedding.shape[0])
                    embedding = np.concatenate([embedding, padding])

            return embedding

        except Exception as e:
            # Don't return a zero vector — that silently indexes a meaningless
            # embedding. Raise so the per-segment ingestion handler skips and
            # records the failure instead.
            logger.error(f"Failed to generate acoustic embedding: {e}")
            raise

    def _remote_acoustic_embedding(
        self,
        audio_path: Optional[Path],
        audio_array: Optional[np.ndarray],
        sample_rate: int,
    ) -> np.ndarray:
        """POST the audio to the clap_embed sidecar and return its vector.

        An array input is serialised to WAV first; a path is sent as raw
        file bytes."""
        import base64
        import io

        if audio_path is not None:
            raw = Path(audio_path).read_bytes()
        else:
            import soundfile as sf

            buf = io.BytesIO()
            sf.write(buf, audio_array, sample_rate, format="WAV")
            raw = buf.getvalue()

        return self._post_remote(
            "/embed/audio",
            {"audio_b64": base64.b64encode(raw).decode()},
        )

    def _remote_acoustic_text_embedding(self, text: str) -> np.ndarray:
        return self._post_remote("/embed/text", {"text": text})

    def _post_remote(self, path: str, payload: Dict[str, str]) -> np.ndarray:
        import httpx

        url = f"{self._clap_endpoint_url}{path}"
        try:
            response = self._get_http_client().post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"CLAP request to {url} failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            vector = response.json()["vec"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"CLAP request to {url} returned an invalid embedding response"
            ) from exc
        return np.asarray(vector, dtype=np.float32)

    def generate_acoustic_text_embedding(self, text: str) -> np.ndarray:
        """Generate a 512-dim text embedding in CLAP's joint audio-text space.

        CLAP is contrastively trained so text and audio share one embedding
        space; encoding a query with get_text_features yields a vector directly
        comparable to the stored audio acoustic_embedding. A sentence-transformer
        embedding lives in a different space and cannot be compared to it.

        Returns:
            512-dim acoustic-space text embedding
        """
        if self._clap_endpoint_url:
            return self._remote_acoustic_text_embedding(text)

        try:
            inputs = self.clap_processor(
                text=[text],
                return_tensors="pt",
                padding=True,
            )

            import torch

            with torch.no_grad():
                text_embeds = self.clap_model.get_text_features(**inputs)

            embedding = text_embeds.squeeze().cpu().numpy()

            if embedding.shape[0] != 512:
                logger.warning(
                    f"CLAP text embedding has {embedding.shape[0]} dims, expected 512"
                )
                if embedding.shape[0] > 512:
                    embedding = embedding[:512]
                else:
                    padding = np.zeros(512 - embedding.shape[0])
                    embedding = np.concatenate([embedding, padding])

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate acoustic text embedding: {e}")
            raise

    def generate_semantic_embedding(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding from text using sentence transformers

        Args:
            text: Input text (transcript)

        Returns:
            768-dim semantic embedding
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for semantic embedding")
            return np.zeros(768, dtype=np.float32)

        try:
            # Generate embedding
            embedding = self.semantic_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Ensure exactly 768 dimensions
            if embedding.shape[0] != 768:
                logger.warning(
                    f"Semantic embedding has {embedding.shape[0]} dims, expected 768"
                )
                if embedding.shape[0] > 768:
                    embedding = embedding[:768]
                else:
                    # Pad with zeros
                    padding = np.zeros(768 - embedding.shape[0])
                    embedding = np.concatenate([embedding, padding])

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate semantic embedding: {e}")
            # Return zero vector on failure
            return np.zeros(768, dtype=np.float32)

    def generate_embeddings(
        self,
        audio_path: Optional[Path] = None,
        audio_array: Optional[np.ndarray] = None,
        transcript: Optional[str] = None,
        sample_rate: int = 48000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate both acoustic and semantic embeddings

        Args:
            audio_path: Path to audio file
            audio_array: Audio array
            transcript: Transcript text
            sample_rate: Audio sample rate

        Returns:
            Tuple of (acoustic_embedding [512], semantic_embedding [768])
        """
        # Generate acoustic embedding
        acoustic_embedding = self.generate_acoustic_embedding(
            audio_path=audio_path,
            audio_array=audio_array,
            sample_rate=sample_rate,
        )

        # Generate semantic embedding
        if transcript:
            semantic_embedding = self.generate_semantic_embedding(transcript)
        else:
            logger.warning("No transcript provided for semantic embedding")
            semantic_embedding = np.zeros(768, dtype=np.float32)

        return acoustic_embedding, semantic_embedding
