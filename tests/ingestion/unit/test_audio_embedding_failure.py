"""generate_acoustic_embedding must raise on failure, not index zeros.

Returning np.zeros(512) on any CLAP/librosa error silently fed a meaningless
embedding into Vespa; the ingestion path's per-segment handler should instead
skip and record the failure.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)


def test_clap_failure_raises_instead_of_zero_vector():
    gen = object.__new__(AudioEmbeddingGenerator)
    # Pre-seed the lazy-loaded backing attrs so no real model loads; the
    # processor raises to simulate a CLAP/decode failure.
    gen._clap_model = object()
    gen._clap_processor = Mock(side_effect=RuntimeError("CLAP boom"))

    with pytest.raises(RuntimeError, match="CLAP boom"):
        gen.generate_acoustic_embedding(audio_array=np.zeros(48000, dtype=np.float32))
