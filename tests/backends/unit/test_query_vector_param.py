"""Query-tensor formatting keys off the input's declared tensor TYPE: a dense
``tensor(v[dim])`` needs a flat list, a mapped ``tensor(querytoken{}, v[dim])``
needs a {token: vector} dict. A (1, dim) array is a single vector and must
flatten for a dense input (it used to dict-ify and trip a Vespa 400). Type — not
schema name — is authoritative, so a dense input whose schema name carries no
_sv_/_lvt_ token still encodes correctly, and a mixed schema encodes each input
by its own type."""

import numpy as np
import pytest

from cogniverse_vespa.search_backend import _format_query_vector_param

_SV = "tensor<float>(v[4])"  # dense single-vector — no mapped dimension
_MV = "tensor<float>(querytoken{}, v[2])"  # mapped multi-vector


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


def test_classifies_by_input_type_not_schema_name():
    """Per-input type is authoritative — a dense input flattens even though its
    type carries no _sv_/_lvt_ schema-name token (a name heuristic would have
    dict-ified it), while a mapped input on the SAME schema still dict-ifies.
    This is what lets a mixed schema (dense acoustic_query + mapped qt) encode
    each input correctly."""
    dense = "tensor<float>(v[512])"  # e.g. audio acoustic_query
    mapped = "tensor<float>(querytoken{}, v[128])"  # e.g. colpali qt

    one_by_dim = np.arange(3, dtype=np.float32).reshape(1, 3)
    assert _format_query_vector_param(one_by_dim, dense) == [0.0, 1.0, 2.0]

    multi = np.arange(4, dtype=np.float32).reshape(2, 2)
    assert _format_query_vector_param(multi, mapped) == {
        "0": [0.0, 1.0],
        "1": [2.0, 3.0],
    }
