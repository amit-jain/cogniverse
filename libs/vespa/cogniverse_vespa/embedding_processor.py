#!/usr/bin/env python3
"""
Vespa Embedding Processor - Handles Vespa-specific format conversion
"""

import logging
from binascii import hexlify
from typing import Any, Dict, Optional

import numpy as np

# Fallback name heuristic for callers that don't resolve the authoritative
# flag. Single-vector schemas carry a ``_sv_`` / ``_lvt_`` token (bracketed by
# underscores so an unrelated substring like ``audio_alvtree_index`` doesn't
# match); lower-cased so ``_SV_`` matches too. Prefer schema_is_single_vector.
_SINGLE_VECTOR_TOKENS = ("_sv_", "_lvt_")


def _is_single_vector_schema(schema_name: str) -> bool:
    name = (schema_name or "").lower()
    return any(token in name for token in _SINGLE_VECTOR_TOKENS)


def schema_is_single_vector(schema_def: Dict[str, Any]) -> Optional[bool]:
    """Whether a schema's embedding tensors are single-vector.

    Authoritative — inspects EVERY tensor field's type (the embedding field is
    named differently per schema: ``embedding``, ``colpali_embedding``,
    ``semantic_embedding`` …). A mapped dimension (``{}``) means
    multi-vector / patch (``tensor<bfloat16>(patch{}, v[128])``); a schema is
    single-vector only when none of its tensor fields has one. Returns ``None``
    when the schema declares no tensor field, so the caller falls back to the
    schema-name heuristic rather than guessing.
    """
    tensor_fields = [
        field
        for field in schema_def.get("document", {}).get("fields", [])
        if isinstance(field, dict) and "tensor" in str(field.get("type", ""))
    ]
    if not tensor_fields:
        return None
    return all("{" not in str(f.get("type", "")) for f in tensor_fields)


class VespaEmbeddingProcessor:
    """Processes embeddings for Vespa's specific format requirements"""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        model_name: str = None,
        schema_name: str = None,
        single_vector: Optional[bool] = None,
    ):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.model_name = model_name or ""
        self.schema_name = schema_name or ""
        # Authoritative single-vector vs multi-vector flag, resolved from the
        # schema's embedding tensor type by the caller (schema_is_single_vector).
        self._single_vector = single_vector

    def _resolve_single_vector(self) -> bool:
        """Single-vector format for the target schema.

        Prefers the authoritative flag the caller resolved from the schema's
        embedding tensor type (schema_is_single_vector); falls back to the
        schema-name heuristic when the caller didn't supply it.
        """
        if self._single_vector is not None:
            return self._single_vector
        return _is_single_vector_schema(self.schema_name)

    def process_embeddings(
        self,
        raw_embeddings: Any,
        needs_float: Optional[bool] = None,
        needs_binary: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Process raw embeddings into Vespa format

        Args:
            raw_embeddings: Can be numpy array, dict of arrays, or already processed
            needs_float: Whether the schema's strategies read the float form.
                ``None`` (unknown) produces it.
            needs_binary: Whether the binary form is read. ``None`` produces it.

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

            # A width-0 vector would encode to empty-string/empty-list
            # embedding fields and land malformed in Vespa.
            if raw_embeddings.ndim > 0 and raw_embeddings.shape[-1] == 0:
                raise ValueError(
                    f"zero-width embedding: shape {raw_embeddings.shape} has "
                    f"no embedding dimension"
                )

            # Only produce the formats the schema's ranking strategies read
            # (both when unspecified) — the discarded format previously cost
            # a full conversion pass per document.
            def _formats(arr: np.ndarray) -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                if needs_float is None or needs_float:
                    out["embedding"] = self._convert_to_float_dict(arr)
                if needs_binary is None or needs_binary:
                    out["embedding_binary"] = self._convert_to_binary_dict(arr)
                return out

            if raw_embeddings.ndim == 2 and raw_embeddings.shape[0] > 0:
                # Multi-vector or patch embeddings: (num_patches, embedding_dim)
                return _formats(raw_embeddings)
            elif raw_embeddings.ndim == 1:
                # Global embeddings: pass 1D array directly so _convert_to_float_dict
                # sees ndim==1, sets is_1d_input=True, and returns a plain float list
                # for single-vector schemas (e.g. agent_memories tensor<float>(d0[768])).
                # Pre-reshaping to (1, N) here loses that signal and causes hex encoding.
                return _formats(raw_embeddings)
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

        The schema's embedding tensor type is the authority: single-vector
        schemas use a raw float list; multi-vector / patch schemas use
        hex-encoded bfloat16 in a mapped ``{patch_idx: hex}`` dict.
        """
        is_1d_input = embeddings.ndim == 1
        if is_1d_input:
            embeddings = embeddings.reshape(1, -1)

        # Same guard as the binary path — on a float-only schema this is the
        # only format produced, and a NaN row would hex-encode straight into
        # the index with no signal.
        if not np.all(np.isfinite(embeddings)):
            raise ValueError(
                "Embeddings contain non-finite values (NaN / Inf); "
                "refusing to encode. Check the upstream encoder output."
            )

        if is_1d_input or self._resolve_single_vector():
            self._reject_multirow_for_single_vector(embeddings, is_1d_input)
            return embeddings[0].tolist()

        # One vectorized bfloat16 conversion for the whole matrix, then slice
        # the hex string per patch row.
        arr = np.ascontiguousarray(embeddings, dtype=np.float32)
        full_hex = (arr.view(np.uint32) >> 16).astype(">u2").tobytes().hex().upper()
        row_len = arr.shape[1] * 4
        return {
            str(patch_idx): full_hex[patch_idx * row_len : (patch_idx + 1) * row_len]
            for patch_idx in range(arr.shape[0])
        }

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

        if is_1d_input or self._resolve_single_vector():
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
        """Convert numpy array to hex-encoded bfloat16 format.

        Truncates each float32 to its high 16 bits (bfloat16) and emits the
        big-endian hex digits, vectorized — the per-float struct.pack loop
        cost ~50ms per 1024x128 document.
        """
        arr_f32 = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
        bf16 = (arr_f32.view(np.uint32) >> 16).astype(">u2")
        return bf16.tobytes().hex().upper()
