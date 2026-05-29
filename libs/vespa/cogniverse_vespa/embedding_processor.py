#!/usr/bin/env python3
"""
Vespa Embedding Processor - Handles Vespa-specific format conversion
"""

import logging
import struct
from binascii import hexlify
from typing import Any, Dict, Optional

import numpy as np

# Single-vector schemas have a ``_sv_`` or ``_lvt_`` token in the name. The
# tokens are bracketed by underscores to avoid an unrelated schema whose
# name merely embeds the substring (e.g. ``audio_alvtree_index``); both
# halves of the check are lower-cased so an uppercase ``_SV_`` matches too.
_SINGLE_VECTOR_TOKENS = ("_sv_", "_lvt_")


def _is_single_vector_schema(schema_name: str) -> bool:
    name = (schema_name or "").lower()
    return any(token in name for token in _SINGLE_VECTOR_TOKENS)


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
                # Global embeddings: pass 1D array directly so _convert_to_float_dict
                # sees ndim==1, sets is_1d_input=True, and returns a plain float list
                # for single-vector schemas (e.g. agent_memories tensor<float>(d0[768])).
                # Pre-reshaping to (1, N) here loses that signal and causes hex encoding.
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
            # Pre-fix: silently returned ``raw_embeddings`` as-is. Downstream
            # ``"embedding" in raw_embeddings`` then raised ``TypeError:
            # argument of type 'int' is not iterable`` (or similar) far from
            # the source. Reject up-front with the actual type so the bad
            # caller is visible in the traceback.
            raise TypeError(
                f"Unsupported embedding payload type: {type(raw_embeddings).__name__}. "
                "Expected numpy.ndarray, dict of arrays, or None."
            )

    def _reject_multirow_for_single_vector(
        self, embeddings: np.ndarray, is_1d_input: bool
    ) -> None:
        """Fail loudly instead of silently keeping only ``embeddings[0]``.

        A single-vector schema can hold exactly one vector. A genuine 2D
        ``(N, dim)`` array with ``N > 1`` previously had rows 1..N-1 dropped
        with no warning — silent data loss. A reshaped 1D input (N==1) is fine.
        """
        if not is_1d_input and len(embeddings) > 1:
            raise ValueError(
                f"Single-vector schema '{self.schema_name}' received "
                f"{len(embeddings)} vectors but holds exactly one. Refusing to "
                "silently drop rows — pass a single vector or use a "
                "multi-vector schema."
            )

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

        if is_1d_input or _is_single_vector_schema(self.schema_name):
            self._reject_multirow_for_single_vector(embeddings, is_1d_input)
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

        # Reject NaN / Inf rather than silently binarizing them to 0. ``NaN >
        # 0`` is False in numpy, so a corrupted upstream embedding used to be
        # accepted into the index as an all-zero (or partly-zero) bitmap with
        # no signal that anything went wrong.
        if not np.all(np.isfinite(embeddings)):
            raise ValueError(
                "Embeddings contain non-finite values (NaN / Inf); "
                "refusing to binarize. Check the upstream encoder output."
            )

        # Binarize: positive values -> 1, negative/zero -> 0
        binarized = np.packbits(np.where(embeddings > 0, 1, 0), axis=1).astype(np.int8)

        if is_1d_input or _is_single_vector_schema(self.schema_name):
            # Single-vector schemas use hex string for binary embeddings
            self._reject_multirow_for_single_vector(embeddings, is_1d_input)
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
