"""Query-tensor formatting: single-vector schemas need a dense list, multi-
vector schemas need a {token: vector} dict. A (1, dim) array is a single
vector and must flatten for single-vector schemas (it used to dict-ify and
trip a Vespa 400)."""

import numpy as np

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
