#!/usr/bin/env python3
"""
Audio Embedding Generator

Generates acoustic and semantic embeddings for audio content:
- Acoustic embeddings (512-dim): CLAP model for audio features
- Semantic embeddings (768-dim): Sentence transformers for transcript semantics
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class AudioEmbeddingGenerator:
    """Generate acoustic and semantic embeddings for audio"""

    def __init__(
        self,
        clap_model: str = "laion/clap-htsat-unfused",
        semantic_model: str = "sentence-transformers/all-mpnet-base-v2",
    ):
        """
        Initialize audio embedding generator

        Args:
            clap_model: CLAP model for acoustic embeddings (512-dim)
            semantic_model: Sentence transformer for semantic embeddings (768-dim)
        """
        self._clap_model_name = clap_model
        self._semantic_model_name = semantic_model

        # Lazy loading
        self._clap_model = None
        self._clap_processor = None
        self._semantic_model = None

        logger.info("AudioEmbeddingGenerator initialized")
        logger.info(f"  Acoustic model: {clap_model}")
        logger.info(f"  Semantic model: {semantic_model}")

    @property
    def clap_model(self):
        """Lazy load CLAP model"""
        if self._clap_model is None:
            logger.info(f"Loading CLAP model: {self._clap_model_name}")
            try:
                from transformers import ClapModel, ClapProcessor

                self._clap_model = ClapModel.from_pretrained(self._clap_model_name)
                self._clap_processor = ClapProcessor.from_pretrained(
                    self._clap_model_name
                )
                self._clap_model.eval()
                logger.info("✅ CLAP model loaded")
            except Exception as e:
                logger.error(f"Failed to load CLAP model: {e}")
                raise
        return self._clap_model

    @property
    def clap_processor(self):
        """Get CLAP processor (triggers model load)"""
        _ = self.clap_model  # Trigger load
        return self._clap_processor

    @property
    def semantic_model(self):
        """Lazy load semantic model"""
        if self._semantic_model is None:
            logger.info(f"Loading semantic model: {self._semantic_model_name}")
            try:
                self._semantic_model = SentenceTransformer(self._semantic_model_name)
                logger.info("✅ Semantic model loaded")
            except Exception as e:
                logger.error(f"Failed to load semantic model: {e}")
                raise
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

            # Generate embedding
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
            logger.error(f"Failed to generate acoustic embedding: {e}")
            # Return zero vector on failure
            return np.zeros(512, dtype=np.float32)

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
