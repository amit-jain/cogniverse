"""Which encoder fills each query tensor input of a rank profile.

A rank profile's inputs are filled from the query text. An input scored
against a field that a profile embeds with a service of its own (named under
the field's name in ``inference_services``, e.g. ``"acoustic_embedding":
"clap_embed"``) takes that service's text encoder; every other input,
including one scored against a field the profile's ``embedding`` service
fills, takes the profile's query encoder, the binary inputs packing its float
output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class QueryInputEncoding:
    """The encoder of one query input and the shape it produces.

    ``service`` is None for the profile's query encoder. ``dim`` is the float
    width of each vector, None when the profile declares none.
    """

    service: Optional[str]
    multi_vector: bool
    dim: Optional[int]


@dataclass(frozen=True)
class TensorShape:
    mapped: bool
    cell: str
    dim: Optional[int]


_TENSOR_TYPE = re.compile(r"tensor(?:<(\w+)>)?\((.*)\)")
_INDEXED_DIM = re.compile(r"\w+\[(\d+)\]")


def tensor_shape(input_type: str) -> TensorShape:
    """Parse ``tensor<cell>(dims)`` into its mapped flag, cell type and the
    size of its indexed dimension."""
    match = _TENSOR_TYPE.fullmatch(input_type.replace(" ", ""))
    if match is None:
        raise ValueError(f"Not a tensor type: {input_type!r}")
    cell, dims = match.group(1) or "double", match.group(2)
    indexed = _INDEXED_DIM.findall(dims)
    return TensorShape(
        mapped="{}" in dims, cell=cell, dim=int(indexed[0]) if indexed else None
    )


def profile_encoder_encoding(profile_config: Mapping[str, Any]) -> QueryInputEncoding:
    """The shape of the query encoder the search backend builds for the
    profile: DenseOn for a dense profile, else the profile's model with the
    width its ``schema_config.embedding_dim`` declares."""
    from cogniverse_foundation.inference_specs import get_inference_service_spec

    embedding_type = str(profile_config.get("embedding_type") or "").lower()
    encoder_name = str(profile_config.get("encoder") or "").lower()
    if embedding_type == "dense" or encoder_name == "denseon":
        return QueryInputEncoding(
            service=None,
            multi_vector=False,
            dim=get_inference_service_spec("denseon").output_dimension,
        )
    schema_config = profile_config.get("schema_config") or {}
    return QueryInputEncoding(
        service=None,
        multi_vector=embedding_type == "multi_vector",
        dim=schema_config.get("embedding_dim"),
    )


def query_input_encodings(
    profile_config: Mapping[str, Any], rank_config: Mapping[str, Any]
) -> Dict[str, QueryInputEncoding]:
    """The encoder of every query input ``rank_config`` declares."""
    from cogniverse_core.query.encoders import SERVICE_TEXT_ENCODERS
    from cogniverse_foundation.inference_specs import get_inference_service_spec

    services = profile_config.get("inference_services") or {}
    input_fields = rank_config.get("input_fields") or {}
    profile_encoding = profile_encoder_encoding(profile_config)
    encodings = {}
    for name in rank_config.get("inputs") or {}:
        service = services.get(input_fields.get(name, ""))
        if service and service != services.get("embedding"):
            encoder_class = SERVICE_TEXT_ENCODERS.get(service)
            encodings[name] = QueryInputEncoding(
                service=service,
                multi_vector=bool(getattr(encoder_class, "multi_vector", False)),
                dim=get_inference_service_spec(service).output_dimension,
            )
        else:
            encodings[name] = profile_encoding
    return encodings


def encoding_fits(encoding: QueryInputEncoding, input_type: str) -> bool:
    """Whether the encoder's output binds to a tensor of ``input_type``:
    per-token vectors to a mapped dimension, one vector to none, and the
    vector width equal to the indexed dimension (an int8 input packs eight
    float dimensions per cell)."""
    shape = tensor_shape(input_type)
    if encoding.dim is None or shape.dim is None:
        return False
    if shape.mapped != encoding.multi_vector:
        return False
    width = encoding.dim // 8 if shape.cell == "int8" else encoding.dim
    return width == shape.dim
