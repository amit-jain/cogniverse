#!/usr/bin/env python3
"""
Generic Embedding Generator - Backend-agnostic implementation
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EmbeddingResult:
    video_id: str
    total_documents: int
    documents_processed: int
    documents_fed: int
    processing_time: float
    errors: list[str]
    metadata: dict[str, Any]


class BaseEmbeddingGenerator(ABC):
    def __init__(self, config: dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def generate_embeddings(
        self, video_data: dict[str, Any], output_dir: Path
    ) -> EmbeddingResult:
        pass
