"""Shipped profile selection follows capabilities and resolves its schema."""

import json

import pytest

from tests.utils import vespa_test_helpers as helpers


@pytest.mark.parametrize("change_cwd", [False, True])
def test_selects_shipped_video_by_capabilities(tmp_path, monkeypatch, change_cwd):
    if change_cwd:
        monkeypatch.chdir(tmp_path)
    profile = helpers.shipped_profile(
        profile_type="video", embedding_type="single_vector"
    )
    source = json.loads(helpers._PROFILES_PATH.read_text())["backend"]["profiles"]
    expected = {
        name: value
        for name, value in source.items()
        if value["type"] == "video" and value["embedding_type"] == "single_vector"
    }
    assert {profile.profile_name: profile.to_dict()} == expected
    assert (
        helpers.load_raw_schema_json(profile.schema_name)["name"] == profile.schema_name
    )


def test_selects_distinct_frame_and_chunk_profiles():
    frame = helpers.shipped_profile(
        profile_type="video", embedding_type="multi_vector", extract_keyframes=True
    )
    chunk = helpers.shipped_profile(
        profile_type="video", embedding_type="multi_vector", process_type="video_chunks"
    )
    assert frame.pipeline_config["extract_keyframes"] is True
    assert chunk.pipeline_config["extract_keyframes"] is False
    assert chunk.process_type == "video_chunks"
    source = json.loads(helpers._PROFILES_PATH.read_text())["backend"]["profiles"]
    assert frame.to_dict() == source[frame.profile_name]
    assert chunk.to_dict() == source[chunk.profile_name]


@pytest.fixture
def profile_catalog(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(helpers, "_PROFILES_PATH", path)
    monkeypatch.setattr(helpers, "_SCHEMAS_DIR", tmp_path)
    return path


def _write_profiles(path, names):
    profiles = {
        name: {
            "type": "video",
            "embedding_type": "single_vector",
            "schema_name": "fixture_current",
        }
        for name in names
    }
    path.write_text(json.dumps({"backend": {"profiles": profiles}}))


def test_profile_identity_and_schema_identity_can_differ(profile_catalog):
    _write_profiles(profile_catalog, ["fixture_profile"])
    (profile_catalog.parent / "fixture_current_schema.json").write_text(
        json.dumps({"name": "fixture_current"})
    )
    profile = helpers.shipped_profile(
        profile_type="video", embedding_type="single_vector"
    )
    assert (profile.profile_name, profile.schema_name) == (
        "fixture_profile",
        "fixture_current",
    )


def test_capability_selection_uses_profile_defaults(profile_catalog):
    profile_catalog.write_text(
        json.dumps(
            {
                "backend": {
                    "profiles": {
                        "fixture_unconfigured": {},
                        "fixture_profile": {
                            "embedding_type": "single_vector",
                            "schema_name": "fixture_current",
                        },
                    }
                }
            }
        )
    )
    (profile_catalog.parent / "fixture_current_schema.json").write_text(
        json.dumps({"name": "fixture_current"})
    )
    profile = helpers.shipped_profile(
        profile_type="video", embedding_type="single_vector"
    )
    assert (profile.profile_name, profile.type, profile.schema_name) == (
        "fixture_profile",
        "video",
        "fixture_current",
    )


@pytest.mark.parametrize("reverse_catalog", [False, True])
def test_selects_exact_capabilities_without_inference_services(
    profile_catalog, reverse_catalog
):
    selected = {
        "type": "video",
        "embedding_type": "single_vector",
        "process_type": "video_chunks",
        "pipeline_config": {"extract_keyframes": False},
        "schema_name": "fixture_current",
        "inference_services": {"embedding": "fixture_unconfigured"},
    }
    profiles = {
        "fixture_image": {**selected, "type": "image"},
        "fixture_multi_vector": {**selected, "embedding_type": "multi_vector"},
        "fixture_frame": {**selected, "process_type": "frames"},
        "fixture_keyframes": {
            **selected,
            "pipeline_config": {"extract_keyframes": True},
        },
        "fixture_selected": selected,
    }
    if reverse_catalog:
        profiles = dict(reversed(profiles.items()))
    profile_catalog.write_text(
        json.dumps({"backend": {"profiles": profiles}, "inference_service_urls": {}})
    )
    (profile_catalog.parent / "fixture_current_schema.json").write_text(
        json.dumps({"name": "fixture_current"})
    )
    profile = helpers.shipped_profile(
        profile_type="video",
        embedding_type="single_vector",
        process_type="video_chunks",
        extract_keyframes=False,
    )
    assert (
        profile.profile_name,
        profile.schema_name,
        profile.type,
        profile.embedding_type,
        profile.process_type,
        profile.pipeline_config,
    ) == (
        "fixture_selected",
        "fixture_current",
        "video",
        "single_vector",
        "video_chunks",
        {"extract_keyframes": False},
    )
    assert profile.extra_config == {
        "inference_services": {"embedding": "fixture_unconfigured"}
    }


@pytest.mark.parametrize("names", [[], ["fixture_b", "fixture_a"]])
def test_rejects_missing_or_ambiguous_capabilities(profile_catalog, names):
    _write_profiles(profile_catalog, names)
    with pytest.raises(ValueError) as error:
        helpers.shipped_profile(profile_type="video", embedding_type="single_vector")
    assert str(error.value) == (
        "Expected exactly one shipped profile for "
        "{'profile_type': 'video', 'embedding_type': 'single_vector', "
        "'process_type': None, 'extract_keyframes': None}; "
        f"matched {sorted(names)!r}"
    )


def test_rejects_a_missing_schema_with_its_identity(profile_catalog):
    _write_profiles(profile_catalog, ["fixture_profile"])
    with pytest.raises(FileNotFoundError) as error:
        helpers.shipped_profile(profile_type="video", embedding_type="single_vector")
    assert str(error.value) == (
        "No schema definition for base name 'fixture_current' at "
        f"{profile_catalog.parent / 'fixture_current_schema.json'}"
    )


def test_rejects_a_schema_declaring_a_different_identity(profile_catalog):
    _write_profiles(profile_catalog, ["fixture_profile"])
    (profile_catalog.parent / "fixture_current_schema.json").write_text(
        json.dumps({"name": "fixture_unrelated"})
    )
    with pytest.raises(ValueError) as error:
        helpers.shipped_profile(profile_type="video", embedding_type="single_vector")
    assert str(error.value) == (
        "Schema file for 'fixture_current' declares 'fixture_unrelated'"
    )
