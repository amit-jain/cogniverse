"""FilesystemSchemaLoader name derivation strips only the trailing suffix.

`f.stem.replace("_schema", "")` removed every occurrence, so a schema file
whose logical name itself contains `_schema` (e.g. `code_schema_index`) was
mis-listed and then failed `schema_exists` round-trip.
"""

from __future__ import annotations

import pytest

from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader


@pytest.mark.unit
@pytest.mark.ci_fast
def test_list_available_schemas_strips_only_trailing_suffix(tmp_path):
    (tmp_path / "video_schema.json").write_text("{}")
    (tmp_path / "code_schema_index_schema.json").write_text("{}")

    loader = FilesystemSchemaLoader(tmp_path)

    assert set(loader.list_available_schemas()) == {"video", "code_schema_index"}


@pytest.mark.unit
@pytest.mark.ci_fast
def test_listed_name_round_trips_through_schema_exists(tmp_path):
    (tmp_path / "code_schema_index_schema.json").write_text("{}")

    loader = FilesystemSchemaLoader(tmp_path)
    (name,) = loader.list_available_schemas()

    assert name == "code_schema_index"
    assert loader.schema_exists(name) is True
