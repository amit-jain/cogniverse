#!/usr/bin/env python3
"""
Vespa Embedding Processor - Handles Vespa-specific format conversion
"""

import logging
import struct
from binascii import hexlify
from typing import Any, Dict, Optional

import numpy as np


class VespaEmbeddingProcessor:
    """Processes embeddings for Vespa's specific format requirements"""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        model_name: str = None,
        schema_name: str = None,
    ):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.model_name = model_name or ""
        self.schema_name = schema_name or ""

    def process_embeddings(self, raw_embeddings: Any) -> Dict[str, Any]:
        """
        Process raw embeddings into Vespa format

        Args:
            raw_embeddings: Can be numpy array, dict of arrays, or already processed

        Returns:
            Dict with processed embeddings ready for Vespa
        """
        # Mem0's metadata-only updates flow through VespaBackend.update_document
        # with a Document whose ``embeddings`` dict is empty — raw_embeddings
        # arrives here as None. Returning None made the ingestion client do
        # ``"embedding" in None`` and raise a TypeError that bubbled up as
        # "argument of type 'NoneType' is not iterable". Return an empty dict
        # instead so downstream ``in`` checks are False and no embedding
        # fields are added to the Vespa put.
        if raw_embeddings is None:
            return {}

        if isinstance(raw_embeddings, np.ndarray):
            # Squeeze leading batch dimension: (1, N, D) → (N, D)
            if raw_embeddings.ndim == 3 and raw_embeddings.shape[0] == 1:
                raw_embeddings = raw_embeddings.squeeze(0)

            if raw_embeddings.ndim == 2 and raw_embeddings.shape[0] > 0:
                # Multi-vector or patch embeddings: (num_patches, embedding_dim)
                return {
                    "embedding": self._convert_to_float_dict(raw_embeddings),
                    "embedding_binary": self._convert_to_binary_dict(raw_embeddings),
                }
            elif raw_embeddings.ndim == 1:
                # Global embeddings: (embedding_dim,) → treat as single patch
                raw_embeddings = raw_embeddings.reshape(1, -1)
                return {
                    "embedding": self._convert_to_float_dict(raw_embeddings),
                    "embedding_binary": self._convert_to_binary_dict(raw_embeddings),
                }
            else:
                raise ValueError(
                    f"Unexpected embedding array shape: {raw_embeddings.shape}. "
                    f"Expected 1D (global), 2D (patches, dim), or 3D (1, patches, dim)."
                )
        elif isinstance(raw_embeddings, dict):
            # Multiple embeddings - process each
            processed = {}
            for key, value in raw_embeddings.items():
                if isinstance(value, np.ndarray):
                    if "binary" in key:
                        processed[key] = self._convert_to_binary_dict(value)
                    else:
                        processed[key] = self._convert_to_float_dict(value)
                else:
                    # Already processed
                    processed[key] = value
            return processed
        else:
            # Unknown format - pass through
            return raw_embeddings

    def _convert_to_float_dict(self, embeddings: np.ndarray) -> Any:
        """Convert numpy array to Vespa float format.

        Schema name is the authority: ``_sv_`` / ``lvt`` schemas use a raw float
        list; everything else (patch-based / multi-vector) uses hex-encoded
        bfloat16 in a mapped ``{patch_idx: hex}`` dict. A row-count heuristic
        wrongly fired single-vector format for a single-token multi-vector
        schema, producing the wrong tensor shape.
        """
        is_1d_input = embeddings.ndim == 1
        if is_1d_input:
            embeddings = embeddings.reshape(1, -1)

        is_single_vector = (
            is_1d_input  # 1D is a single global vector by data shape
            or "_sv_" in self.schema_name
            or "lvt" in self.schema_name.lower()
        )

        if is_single_vector:
            return embeddings[0].tolist()

        embedding_dict = {}
        for patch_idx in range(len(embeddings)):
            hex_string = self._numpy_to_hex_bfloat16(embeddings[patch_idx])
            embedding_dict[str(patch_idx)] = hex_string
        return embedding_dict

    def _convert_to_binary_dict(self, embeddings: np.ndarray) -> Any:
        """Convert numpy array to binary format

        For single-vector schemas, return hex-encoded binary string.
        For patch-based schemas, return dict of hex-encoded binary.
        """
        is_1d_input = embeddings.ndim == 1
        if is_1d_input:
            embeddings = embeddings.reshape(1, -1)

        # Binarize: positive values -> 1, negative/zero -> 0
        binarized = np.packbits(np.where(embeddings > 0, 1, 0), axis=1).astype(np.int8)

        is_single_vector = (
            is_1d_input
            or "_sv_" in self.schema_name
            or "lvt" in self.schema_name.lower()
        )

        if is_single_vector:
            # Single-vector schemas use hex string for binary embeddings
            return hexlify(binarized[0].tobytes()).decode("utf-8")

        # For patch-based schemas, use dict of hex-encoded values
        embedding_dict = {}
        for idx in range(len(binarized)):
            hex_string = hexlify(binarized[idx].tobytes()).decode("utf-8")
            embedding_dict[str(idx)] = hex_string

        return embedding_dict

    def _numpy_to_hex_bfloat16(self, array: np.ndarray) -> str:
        """Convert numpy array to hex-encoded bfloat16 format"""
        arr_f32 = np.asarray(array, dtype=np.float32).flatten()

        def float_to_bfloat16_hex(f: float) -> str:
            packed_float = struct.pack("=f", f)
            bfloat16_bits = struct.unpack("=H", packed_float[2:])[0]
            return format(bfloat16_bits, "04X")

        hex_list = [float_to_bfloat16_hex(float(val)) for val in arr_f32]
        return "".join(hex_list)
