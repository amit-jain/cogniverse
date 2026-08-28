"""Source-granularity source-key derivation stays schema-driven.

The backend must derive the collapse key from each schema's
``document_mapping.id`` so source-level search does not hardcode a video-only
field name. Schemas without an identity field must fail source collapse with a
schema-naming error.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cogniverse_sdk.document import DocumentFieldMapping
from cogniverse_vespa.search_backend import _schema_source_identity_field

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "configs" / "schemas"


@pytest.mark.unit
@pytest.mark.ci_fast
def test_schema_source_identity_field_matches_document_mapping():
    for schema_path in sorted(SCHEMAS_DIR.glob("*_schema.json")):
        schema_name = schema_path.stem.removesuffix("_schema")
        schema_json = json.loads(schema_path.read_text())
        mapping = DocumentFieldMapping.from_schema_json(
            schema_json, schema_name=schema_name, required=False
        )

        if mapping is None or not mapping.id:
            with pytest.raises(
                ValueError,
                match=rf"Schema '{re.escape(schema_name)}'.*source identity field",
            ):
                _schema_source_identity_field(
                    schema_json, schema_name=schema_name, required=True
                )
            continue

        assert (
            _schema_source_identity_field(schema_json, schema_name=schema_name)
            == mapping.id
        )
