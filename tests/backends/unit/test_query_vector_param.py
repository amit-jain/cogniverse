"""Query-tensor formatting: single-vector schemas need a dense list, multi-
vector schemas need a {token: vector} dict. A (1, dim) array is a single
vector and must flatten for single-vector schemas (it used to dict-ify and
trip a Vespa 400)."""

import numpy as np
import pytest

from cogniverse_vespa.search_backend import _format_query_vector_param

_SV = "video_videoprism_lvt_base_sv_chunk_6s"  # single-vector (_sv_/_lvt_)
_MV = "video_colpali_smol500_frame"  # multi-vector (no token)


def test_single_vector_schema_flattens_1xdim():
    arr = np.arange(4, dtype=np.float32).reshape(1, 4)
    assert _format_query_vector_param(arr, _SV) == [0.0, 1.0, 2.0, 3.0]


def test_single_vector_schema_passthrough_1d():
    arr = np.arange(4, dtype=np.float32)
    assert _format_query_vector_param(arr, _SV) == [0.0, 1.0, 2.0, 3.0]


def test_multivector_schema_dicts_multirow():
    arr = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert _format_query_vector_param(arr, _MV) == {
        "0": [0.0, 1.0],
        "1": [2.0, 3.0],
        "2": [4.0, 5.0],
    }


def test_multivector_schema_flat_1d():
    arr = np.arange(2, dtype=np.float32)
    assert _format_query_vector_param(arr, _MV) == [0.0, 1.0]


def test_single_vector_multirow_raises_not_silently_drops():
    """A genuine (N>1, dim) array on a single-vector schema must fail loud,
    mirroring the ingestion side — not silently keep only row 0."""
    arr = np.arange(8, dtype=np.float32).reshape(2, 4)
    with pytest.raises(ValueError, match="[Ss]ingle-vector"):
        _format_query_vector_param(arr, _SV)


def test_single_vector_empty_raises_not_indexerror():
    """An empty (0, dim) encoder result must raise a clear ValueError, not an
    IndexError from arr[0]."""
    arr = np.zeros((0, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="empty|no vectors"):
        _format_query_vector_param(arr, _SV)
