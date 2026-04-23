"""Correctness tests for VespaEmbeddingProcessor's sign-based binarization.

The ``lateon_mv`` schema (and the general ColBERT binary-quantization pattern)
stores per-token 128-dim embeddings both as bfloat16 (for MaxSim rerank) and
as 16-byte packed binary (for fast hamming-distance first-phase ranking).
These tests pin the packing rules so a refactor cannot silently change the
bit layout that the Vespa rank profiles depend on.
"""

from __future__ import annotations

from binascii import unhexlify

import numpy as np
import pytest

from cogniverse_vespa.embedding_processor import VespaEmbeddingProcessor


@pytest.fixture
def processor() -> VespaEmbeddingProcessor:
    return VespaEmbeddingProcessor()


def test_binary_conversion_packs_positives_as_one_negatives_as_zero(processor):
    """Sign → bit mapping: +ve ⇒ 1, <=0 ⇒ 0, packed MSB-first into bytes."""
    # Two tokens so we get the multi-token dict path; one row would be
    # treated as a single-vector and return a flat hex string instead.
    # Row 0 sign pattern: + + - - + + + - | - - - - + + + + → 0xCE 0x0F
    # Row 1 sign pattern: - + + + - + + + | + - + - + - + - → 0x77 0xAA
    vec = np.array(
        [
            [+0.1, +0.9, -0.1, -0.5, +0.2, +0.3, +0.4, -0.5,
             -0.1, -0.2, -0.3, -0.4, +0.1, +0.2, +0.3, +0.4],
            [-0.1, +0.2, +0.3, +0.4, -0.5, +0.6, +0.7, +0.8,
             +0.1, -0.2, +0.3, -0.4, +0.5, -0.6, +0.7, -0.8],
        ],
        dtype=np.float32,
    )
    out = processor._convert_to_binary_dict(vec)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"0", "1"}
    assert unhexlify(out["0"]) == b"\xce\x0f"
    assert unhexlify(out["1"]) == b"\x77\xaa"


def test_binary_conversion_dim128_gives_16_bytes_per_token(processor):
    """LateOn outputs (N, 128); packing yields 16 bytes (=128 bits) per token."""
    rng = np.random.default_rng(seed=42)
    tokens = rng.normal(size=(7, 128)).astype(np.float32)
    out = processor._convert_to_binary_dict(tokens)

    assert set(out.keys()) == {str(i) for i in range(7)}
    for token_idx, hex_str in out.items():
        assert len(hex_str) == 32, f"token {token_idx}: expected 32 hex chars (16 bytes), got {len(hex_str)}"
        assert len(unhexlify(hex_str)) == 16


def test_binary_conversion_matches_manual_packbits(processor):
    """Parity with numpy.packbits is the ground-truth bit layout Vespa expects."""
    rng = np.random.default_rng(seed=1)
    tokens = rng.normal(size=(3, 48)).astype(np.float32)  # LateOn-Code-edge dim
    out = processor._convert_to_binary_dict(tokens)

    expected = np.packbits(np.where(tokens > 0, 1, 0), axis=1).astype(np.int8)
    for token_idx in range(3):
        packed = unhexlify(out[str(token_idx)])
        # Compare as unsigned bytes — packbits returns int8 but the wire format is bytes
        assert list(packed) == list(expected[token_idx].astype(np.uint8))


def test_binary_conversion_single_vector_returns_hex_string(processor):
    """1D inputs (global embeddings) return a flat hex string, not a dict."""
    vec = np.array([+1.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0], dtype=np.float32)
    out = processor._convert_to_binary_dict(vec)

    assert isinstance(out, str)
    # Pattern 10101010 = 0xAA
    assert unhexlify(out) == b"\xaa"


def test_full_process_embeddings_emits_both_fields(processor):
    """process_embeddings() must return both ``embedding`` and ``embedding_binary``
    for multi-vector inputs — this is what the lateon_mv Vespa schema requires."""
    rng = np.random.default_rng(seed=7)
    tokens = rng.normal(size=(5, 128)).astype(np.float32)
    out = processor.process_embeddings(tokens)

    assert set(out.keys()) == {"embedding", "embedding_binary"}
    assert isinstance(out["embedding"], dict)
    assert isinstance(out["embedding_binary"], dict)
    assert len(out["embedding"]) == 5
    assert len(out["embedding_binary"]) == 5
    for i in range(5):
        assert len(unhexlify(out["embedding_binary"][str(i)])) == 16


def test_binary_conversion_zero_collapses_to_zero_bit(processor):
    """``np.where(embeddings > 0, 1, 0)`` treats exact zero as negative — the
    hamming-distance rank profile relies on this stable tie-break rule."""
    vec = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    out = processor._convert_to_binary_dict(vec)
    assert unhexlify(out["0"]) == b"\x00"
    assert unhexlify(out["1"]) == b"\xa0"
