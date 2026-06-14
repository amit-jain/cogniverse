"""Single-vector vs multi-vector classification from the schema's tensor type.

The embedding processor must format a single-vector schema's embedding as a
flat float list even when the schema name lacks the ``_sv_``/``_lvt_`` token
(e.g. ``agent_memories``). The authoritative ``single_vector`` flag — resolved
from the schema's embedding tensor type — overrides the name heuristic.
"""

import numpy as np

from cogniverse_vespa.embedding_processor import (
    VespaEmbeddingProcessor,
    schema_is_single_vector,
)


def _schema(embedding_type: str | None):
    fields = [{"name": "title", "type": "string"}]
    if embedding_type is not None:
        fields.append({"name": "embedding", "type": embedding_type})
    return {"document": {"fields": fields}}


def test_schema_is_single_vector_reads_tensor_type():
    assert schema_is_single_vector(_schema("tensor<float>(d0[768])")) is True
    assert (
        schema_is_single_vector(_schema("tensor<bfloat16>(patch{}, v[128])")) is False
    )
    assert (
        schema_is_single_vector(_schema("tensor<bfloat16>(token{}, v[128])")) is False
    )
    # No tensor field → undeterminable → None (caller falls back to heuristic).
    assert schema_is_single_vector(_schema(None)) is None


def test_classifies_by_any_tensor_field_not_named_embedding():
    # The embedding field is named differently per schema (colpali_embedding,
    # semantic_embedding, …) — classification must inspect ALL tensor fields,
    # not just one named "embedding".
    colpali = {
        "document": {
            "fields": [
                {"name": "title", "type": "string"},
                {
                    "name": "colpali_embedding",
                    "type": "tensor<bfloat16>(patch{}, v[128])",
                },
            ]
        }
    }
    assert schema_is_single_vector(colpali) is False


def test_mixed_single_and_multi_tensor_fields_is_multi():
    # audio_content holds a single-vector acoustic field AND a multi-vector
    # semantic field — any mapped dimension makes the schema multi-vector.
    audio = {
        "document": {
            "fields": [
                {"name": "acoustic_embedding", "type": "tensor<float>(v[512])"},
                {
                    "name": "semantic_embedding",
                    "type": "tensor<bfloat16>(token{}, v[128])",
                },
            ]
        }
    }
    assert schema_is_single_vector(audio) is False


def test_real_schemas_classified_correctly():
    import glob
    import json

    def _load(name):
        return json.load(open(glob.glob(f"configs/schemas/{name}*.json")[0]))

    assert schema_is_single_vector(_load("agent_memories")) is True
    assert schema_is_single_vector(_load("document_visual")) is False
    assert schema_is_single_vector(_load("audio_content")) is False
    assert schema_is_single_vector(_load("video_colpali_smol500_mv_frame")) is False


def test_authoritative_flag_formats_2d_single_vector_as_float_list():
    # agent_memories has NO _sv_/_lvt_ token, so the name heuristic would
    # misclassify a (1, 768) array as multi-vector (hex dict). The
    # authoritative flag produces the correct flat float list.
    proc = VespaEmbeddingProcessor(
        schema_name="agent_memories_acme_acme", single_vector=True
    )
    out = proc.process_embeddings(np.ones((1, 768), dtype=np.float32))

    assert isinstance(out["embedding"], list)
    assert len(out["embedding"]) == 768
    # Binary embedding is a single hex string, not a {patch: hex} dict.
    assert isinstance(out["embedding_binary"], str)


def test_authoritative_flag_formats_multi_vector_as_patch_dict():
    proc = VespaEmbeddingProcessor(schema_name="ambiguous_name", single_vector=False)
    out = proc.process_embeddings(np.ones((3, 128), dtype=np.float32))

    assert isinstance(out["embedding"], dict)
    assert set(out["embedding"].keys()) == {"0", "1", "2"}


def test_visual_schemas_are_320_dim():
    import glob

    for name in (
        "document_visual",
        "video_colqwen_omni_mv_chunk_30s",
        "video_colpali_smol500_mv_frame",
        "image_colpali_mv",
    ):
        path = glob.glob(f"configs/schemas/{name}*.json")[0]
        raw = open(path).read()
        assert "v[320]" in raw and "v[40]" in raw, f"{name} not widened"
        assert "v[128]" not in raw and "v[16]" not in raw, f"{name} still has 128/16"
