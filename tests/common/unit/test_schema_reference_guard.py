"""Schema references resolve to shipped files or explicitly declared test data."""

from pathlib import Path

import pytest

from tests.utils.schema_references import missing_schema_references, schema_references

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


SCHEMA_REFERENCE_FIXTURES = {
    "libs/dashboard/cogniverse_dashboard/tabs/optimization.py": (
        "Pydantic training example type identities",
        "EntityExtractionExampleSchema ProfileSelectionExampleSchema QueryEnhancementExampleSchema RoutingExperienceSchema",
    ),
    "libs/foundation/cogniverse_foundation/telemetry/context.py": (
        "Telemetry label when schema context is unavailable",
        "unknown",
    ),
    "libs/sdk/cogniverse_sdk/interfaces/config_store.py": (
        "ConfigScope enum value for stored schema configuration",
        "schema",
    ),
    "libs/synthetic/cogniverse_synthetic/schemas.py": (
        "Pydantic training example type identity",
        "ProfileSelectionExampleSchema",
    ),
    "tests/admin/conftest.py": (
        "Placeholder filename in fixture isolation documentation",
        "X",
    ),
    "tests/admin/test_profile_api.py": (
        "Missing schema and immutable schema change rejection",
        "different_schema nonexistent_schema",
    ),
    "tests/admin/test_tenant_manager.py": (
        "Tenant schema prefix assertion",
        "video_colpali_smol500_mv_frame_",
    ),
    "tests/agents/integration/test_content_types_vespa.py": (
        "Constructed by VespaSchemaManager.upload_content_type_schemas",
        "image_content",
    ),
    "tests/agents/unit/test_graph_feed_retry.py": (
        "Tenant schema returned by the graph manager fixture",
        "knowledge_graph_acme",
    ),
    "tests/agents/unit/test_graph_manager_yql_escaping.py": (
        "Injected graph schema for YQL escaping",
        "kg_test",
    ),
    "tests/agents/unit/test_graph_search_contract.py": (
        "Injected graph schema for search response checks",
        "kg_test",
    ),
    "tests/agents/unit/test_profile_agnostic_search_service.py": (
        "Fabricated profile configuration and backend cache identities",
        "frame_based_colpali test_schema video_colqwen_omni video_xclip_base",
    ),
    "tests/agents/unit/test_wiki_manager.py": (
        "Invalid colon separator is rejected at construction",
        "wiki_pages_acme:production",
    ),
    "tests/backends/integration/test_feed_convergence.py": (
        "Intentionally absent document type tests convergence failure",
        "convergence_missing",
    ),
    "tests/backends/integration/test_tenant_schema_lifecycle.py": (
        "Intentionally absent schema and document type probes",
        "nonexistent_schema_xyz schema_that_was_never_deployed wiki",
    ),
    "tests/backends/unit/test_backend_bool_contracts.py": (
        "Missing schema errors and injected strategy metadata",
        "missing_schema nonexistent_schema video_probe_schema",
    ),
    "tests/backends/unit/test_backend_config.py": (
        "Configuration serialization and tenant merge fixtures",
        "base_schema new_schema schema1 schema2 schema_a schema_b system_schema tenant_custom_schema tenant_schema test test_schema video_profile",
    ),
    "tests/backends/unit/test_backend_registry_tenant.py": (
        "Backend double profile mutation fixtures",
        "document_chunk video_frame",
    ),
    "tests/backends/unit/test_build_query_inputs.py": (
        "Injected query strategy and tenant identities",
        "agent_memories_acme_acme video_colpali video_frame",
    ),
    "tests/backends/unit/test_delete_schema_suffix_guard.py": (
        "Synthetic registry rows and deletion failures",
        "video video_colpali video_x video_x_acme_acme",
    ),
    "tests/backends/unit/test_edge_inputs_vespa.py": (
        "Embedding token and namespace parser inputs",
        "agent_memories_acme_acme agent_memories_sv_768 agent_memories_sv_x archived_config_metadata knowledge_graph_acme_acme legacy_knowledge_graph_acme_acme tenant_metadata_acme_acme video_colpali_smol500_mv_frame_acme_acme video_test_mv_frame video_x_mv_frame wiki_agent_memories_index",
    ),
    "tests/backends/unit/test_embedding_schema_shape.py": (
        "Injected embedding shape and namespace metadata",
        "agent_memories_acme_acme ambiguous_name",
    ),
    "tests/backends/unit/test_filter_condition_quoting.py": (
        "Injected paginated query and export target",
        "video_frame",
    ),
    "tests/backends/unit/test_metadata_app_cache.py": (
        "Persistent session routing target",
        "s",
    ),
    "tests/backends/unit/test_partial_update.py": (
        "Persistent document operation target",
        "s",
    ),
    "tests/backends/unit/test_persistent_ops_failfast.py": (
        "Unavailable endpoint request target",
        "any_schema",
    ),
    "tests/backends/unit/test_persistent_vespa_ops.py": (
        "Persistent session routing target",
        "s",
    ),
    "tests/backends/unit/test_profile_change_listener_chain.py": (
        "Configuration listener event fixtures",
        "safe to_delete",
    ),
    "tests/backends/unit/test_query_metadata_status_check.py": (
        "Injected metadata query target",
        "agent_memories_acme x",
    ),
    "tests/backends/unit/test_ranking_strategy_extractor.py": (
        "Temporary ranking definitions and missing file cases",
        "ghost memo_bound memo_probe poison_probe s survivor video_colpali_sv_frame",
    ),
    "tests/backends/unit/test_schema_name_matching.py": (
        "Adversarial vector token parser inputs",
        "agent_memories_tenant_acme anything_with_lvt_token anything_with_sv_token audio_alvtree_index sv_prefix_only_no_underscore video_VIDEOPRISM_SV_global video_xclip_large_mv_chunk_30s video_xclip_lvt_base_sv_chunk_6s video_xclip_lvt_large_sv_chunk_6s",
    ),
    "tests/backends/unit/test_schema_registry.py": (
        "Synthetic loader responses, persisted registry rows, and plain tenant IDs",
        "existing_schema existing_schema_existing_tenant never_deployed new_schema nonexistent nonexistent_schema schema schema1 schema1_acme_acme schema2 schema2_startup_startup schema3 test_schema test_schema_acme test_schema_acme_acme acme startup tenant_a tenant_b",
    ),
    "tests/backends/unit/test_search_backend_dynamic_profiles.py": (
        "Injected profiles for cache and routing checks",
        "dyn fresh s",
    ),
    "tests/backends/unit/test_search_backend_schema_isolation.py": (
        "Injected tenant resolution and YQL targets",
        "another_schema global_wiki tenant_wiki tenant_wiki_acme_acme video_tenant_b",
    ),
    "tests/backends/unit/test_search_soft_timeout.py": (
        "Injected backend timeout target",
        "video_acme_acme",
    ),
    "tests/common/unit/test_profile_validator.py": (
        "Malformed and absent temporary schema files",
        "invalid_schema malformed nonexistent",
    ),
    "tests/common/unit/test_schema_reference_guard.py": (
        "Synthetic detector source and expected locations",
        "image_future_mv video_another_mv_frame video_future_sv_chunk_9s video_future_sv_chunk_9s_org_tenant video_future_sv_chunk_9s_unexplained video_typo_mv_frame available replacement",
    ),
    "tests/core/unit/test_schema_deployment_intents.py": (
        "Forged full name rejected because it does not match the intent owner",
        "wiki_pages_someone_else",
    ),
    "tests/core/unit/test_schema_registry_errors.py": (
        "Missing schema error propagation",
        "video_missing",
    ),
    "tests/dashboard/integration/test_approval_queue_regeneration_real.py": (
        "Pydantic training example type identity",
        "ProfileSelectionExampleSchema",
    ),
    "tests/dashboard/unit/test_chat_answer_format.py": (
        "Synthetic tenant prefix in captured search results",
        "video_colpali_smol500_mv_frame_flywheel_org_",
    ),
    "tests/dashboard/unit/test_optimization_forms.py": (
        "Pydantic training example type identities",
        "ProfileSelectionExampleSchema WorkflowExecutionSchema",
    ),
    "tests/evaluation/conftest.py": (
        "Injected evaluation strategy metadata",
        "video_frame",
    ),
    "tests/evaluation/integration/test_schema_driven_pipeline.py": (
        "Schema analyzer modality fixtures",
        "custom_data document_index image_collection video_frames",
    ),
    "tests/evaluation/unit/test_ground_truth.py": (
        "Injected backend schema and ranking metadata",
        "cached_schema test test_schema video_schema",
    ),
    "tests/evaluation/unit/test_schema_agnostic.py": (
        "Schema analyzer modality fixtures",
        "custom_data document_index documents image_collection images video_frames",
    ),
    "tests/foundation/unit/test_config_utils.py": (
        "Configuration cache mutation fixture",
        "s",
    ),
    "tests/ingestion/integration/test_chunk_only_pipeline_real.py": (
        "Local processing context for ffmpeg chunk extraction",
        "video_colqwen_chunks",
    ),
    "tests/ingestion/integration/test_pipeline_cache_live_path.py": (
        "Cache namespace for isolated pipeline artifacts",
        "testprof",
    ),
    "tests/ingestion/integration/test_real_ingestion_pipeline.py": (
        "Local Strategy fixture for processor resolution",
        "video_colpali",
    ),
    "tests/ingestion/integration/test_vllm_colpali_real_sidecar.py": (
        "Local embedding generator context without Vespa access",
        "video_colqwen",
    ),
    "tests/ingestion/unit/test_embedding_generator_impl.py": (
        "Injected embedding generator configuration",
        "s test_schema video_test",
    ),
    "tests/ingestion/unit/test_pipeline.py": (
        "Injected pipeline context and backend fixture",
        "test_schema",
    ),
    "tests/ingestion/unit/test_pipeline_stage_concurrency.py": (
        "Injected pipeline stage context",
        "video_colpali",
    ),
    "tests/routing/unit/synthetic/test_approval_system.py": (
        "Pydantic training example type identities",
        "EntityExtractionExampleSchema ProfileSelectionExampleSchema QueryEnhancementExampleSchema RoutingExperienceSchema WorkflowExecutionSchema",
    ),
    "tests/routing/unit/synthetic/test_backend_querier.py": (
        "Injected query profiles and modality routing fixtures",
        "alpha audio configured_profile document_segments image s text video_frame video_frames video_frames_mv video_segments",
    ),
    "tests/routing/unit/synthetic/test_generators_integration.py": (
        "Pydantic training example type identities",
        "WorkflowExecutionSchema",
    ),
    "tests/routing/unit/synthetic/test_schemas.py": (
        "Pydantic schema serialization and generator fixtures",
        "ProfileSelectionExampleSchema RoutingExperienceSchema source_schema",
    ),
    "tests/runtime/unit/test_admin_profiles_routes_http.py": (
        "Injected profiles for administrative route responses",
        "acme_video_colpali_sv video_colpali_sv video_new_sv video_prism_mv",
    ),
    "tests/runtime/unit/test_admin_reconcile_orphans.py": (
        "Injected deployed schema list for tenant recovery",
        "knowledge_graph_legit video_brand_new_sv_acme video_colpali_smol500_mv_frame_beta",
    ),
    "tests/runtime/unit/test_audio_profile_inference_services.py": (
        "Injected audio inference service configuration",
        "audio_clap_semantic",
    ),
    "tests/runtime/unit/test_backref_writes.py": (
        "Injected search segment result schema",
        "video_colpali_acme",
    ),
    "tests/runtime/unit/test_batch_optimization_modes.py": (
        "Missing profile and Pydantic example type fixtures",
        "ProfileSelectionExampleSchema RoutingExperienceSchema video_orphan",
    ),
    "tests/runtime/unit/test_dispatcher_answer_search_wiring.py": (
        "Injected active video profile",
        "video_custom_mv_frame",
    ),
    "tests/synthetic/unit/test_agent_inference.py": (
        "Injected modality classification input",
        "alpha opaque segments",
    ),
    "tests/synthetic/unit/test_confidence_extractor.py": (
        "Injected profiles and Pydantic example type identities",
        "EntityExtractionExampleSchema ProfileSelectionExampleSchema QueryEnhancementExampleSchema WorkflowExecutionSchema document_pages",
    ),
    "tests/synthetic/unit/test_profile_generator.py": (
        "Injected serving profiles and training corpus metadata",
        "audio_clap_semantic audio_segments document_pages document_text_semantic document_visual_colpali video_frames video_frames_mv wiki_semantic",
    ),
    "tests/synthetic/unit/test_profile_selector.py": (
        "Adversarial profile token parser input",
        "video_colpaliish_mv_frame",
    ),
    "tests/synthetic/unit/test_routing_generation.py": (
        "Injected content and Pydantic example type identities",
        "RoutingExperienceSchema document video_segments",
    ),
    "tests/synthetic/unit/test_topic_saliency_golden.py": (
        "Recorded video document fixture metadata",
        "video_frames",
    ),
    "tests/system/conftest.py": (
        "Documented tenant schema prefix",
        "video_colpali_smol500_mv_frame_",
    ),
    "tests/test_backend_registry.py": (
        "Backend double registration and mutation fixtures",
        "audio_segment changed_after_registration mock_full test_schema video_frame workflow_test",
    ),
    "tests/utils/test_vllm_sidecar.py": (
        "Injected deployment return values for fixture wiring checks",
        "video_colpali_smol500_mv_frame_bright_probe_test video_colpali_smol500_mv_frame_bright_probe_test_bright_probe_test video_colpali_smol500_mv_frame_test_tenant",
    ),
}


def test_detector_finds_unseen_names_in_schema_contexts():
    source = """\
SCHEMA = "fresh_assignment"
config = {"schema_name": "fresh_mapping"}
backend.search(schema="fresh_keyword")
def build(base_schema_name="fresh_default"):
    return base_schema_name
backend.schema_name = "fresh_attribute"
backend.load_schema("fresh_positional")
base_schema_names = ["fresh_list_a", "fresh_list_b"]
"""
    assert schema_references(source) == {
        (1, "fresh_assignment"),
        (2, "fresh_mapping"),
        (3, "fresh_keyword"),
        (4, "fresh_default"),
        (6, "fresh_attribute"),
        (7, "fresh_positional"),
        (8, "fresh_list_a"),
        (8, "fresh_list_b"),
    }


def test_detector_follows_aliases_without_crossing_function_scopes():
    source = """\
first = "fresh_alias"
second = first
third = second
backend.search(schema_name=third)
def unrelated():
    first = "ordinary_label"
    return first
"""
    assert schema_references(source) == {(1, "fresh_alias")}


def test_detector_finds_shape_only_references_and_preserves_complete_names():
    source = """\
profiles = ["video_future_sv_chunk_9s", "image_future_mv"]
# video_another_mv_frame
label = "unrelated_label"
"""
    assert schema_references(source) == {
        (1, "video_future_sv_chunk_9s"),
        (1, "image_future_mv"),
        (2, "video_another_mv_frame"),
    }


def test_detector_finds_schema_filenames_yql_and_document_urls():
    source = """\
path = "configs/schemas/fresh_path_schema.json"
yql = "select * from sources fresh_query where true"
url = "http://localhost:8080/document/v1/default/fresh_url/docid/doc"
"""
    assert schema_references(source) == {
        (1, "fresh_path"),
        (2, "fresh_query"),
        (3, "fresh_url"),
    }


def test_detector_reads_schema_parametrization_values():
    source = """\
@pytest.mark.parametrize("schema_name, expected", [
    ("fresh_parameter", True),
    ("another_parameter", False),
])
def test_example(schema_name, expected):
    pass
"""
    assert schema_references(source) == {
        (2, "fresh_parameter"),
        (3, "another_parameter"),
    }


def test_detector_ignores_json_schema_types_and_model_names():
    source = """\
config = {"embedding_model": "model_name", "example_schema": "ExampleSchema"}
request = {"input_schema": {"type": "object"}}
message = "unknown_schema is just prose"
"""
    assert schema_references(source) == set()


def test_detector_keeps_invalid_names_until_explicitly_declared():
    source = """\
backend.search(schema_name="deliberately_missing")
backend.search(schema_name="test_schema")
backend.search(schema_name="video_typo_mv_frame")
"""
    assert schema_references(source) == {
        (1, "deliberately_missing"),
        (2, "test_schema"),
        (3, "video_typo_mv_frame"),
    }


def test_schema_references_have_files():
    root = Path(__file__).resolve().parents[3]
    assert (
        missing_schema_references(
            root,
            synthetic_schemas={
                module: dict.fromkeys(names.split(), reason)
                for module, (reason, names) in SCHEMA_REFERENCE_FIXTURES.items()
            },
        )
        == []
    )


def test_detector_separates_filename_suffix_from_schema_identity():
    assert schema_references(
        'path = "configs/schemas/video_future_sv_chunk_9s_schema.json"'
    ) == {(1, "video_future_sv_chunk_9s")}


def test_detector_selects_schema_argument_after_tenant_argument():
    assert schema_references(
        'registry.schema_exists("org:tenant", "fresh_reference")'
    ) == {(1, "fresh_reference")}


@pytest.mark.parametrize(
    "source, expected",
    [
        ('loader.schema_exists("fresh_loader")', {(1, "fresh_loader")}),
        (
            'backend.schema_exists("fresh_backend", tenant_id="org:tenant")',
            {(1, "fresh_backend")},
        ),
        (
            "class Registry:\n"
            "    def schema_exists(self, tenant_id, schema_name):\n"
            "        return True\n"
            'loader.schema_exists("fresh_loader")\n',
            {(4, "fresh_loader")},
        ),
    ],
)
def test_detector_reads_one_schema_exists_argument_despite_other_signatures(
    source, expected
):
    assert schema_references(source) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            'backend.schema_exists("future_missing", "org:tenant")',
            {(1, "future_missing")},
        ),
        (
            'registry.schema_exists("org:tenant", "future_missing")',
            {(1, "future_missing")},
        ),
        (
            'receiver.schema_exists("future_missing", "plain_tenant")',
            {(1, "future_missing"), (1, "plain_tenant")},
        ),
        (
            'receiver.schema_exists("plain_tenant", "future_missing")',
            {(1, "future_missing"), (1, "plain_tenant")},
        ),
        (
            'tenant_id = "plain_tenant"\n'
            "first = tenant_id\n"
            "second = first\n"
            'receiver.schema_exists("future_missing", second)\n',
            {(4, "future_missing")},
        ),
        (
            'backend.search(schema_name="invalid:document")',
            {(1, "invalid:document")},
        ),
        (
            'receiver.schema_exists("invalid:document", "org:tenant")',
            {(1, "invalid:document"), (1, "org:tenant")},
        ),
        (
            'tenant_id = "future_missing"\n'
            'receiver.schema_exists("future_missing", "plain_tenant")\n',
            {(2, "future_missing"), (2, "plain_tenant")},
        ),
    ],
)
def test_detector_checks_ambiguous_schema_exists_positions(source, expected):
    assert schema_references(source) == expected


def test_detector_stops_aliases_at_shadowing_parameters():
    source = """\
name = "ordinary_label"
def lookup(name):
    backend.search(schema_name=name)
"""
    assert schema_references(source) == set()


def _catalog(root, *names):
    directory = root / "configs" / "schemas"
    directory.mkdir(parents=True)
    for name in names:
        (directory / f"{name}_schema.json").write_text("{}")
    (root / "tests").mkdir()
    (root / "libs").mkdir()


def test_guard_reports_future_names_in_both_source_roots(tmp_path):
    _catalog(tmp_path, "available")
    (tmp_path / "tests" / "example.py").write_text(
        'backend.search(schema_name="future_missing")\n'
    )
    (tmp_path / "libs" / "example.py").write_text(
        'backend.load_schema("another_missing")\n'
    )
    assert missing_schema_references(tmp_path) == [
        "libs/example.py:1: another_missing",
        "tests/example.py:1: future_missing",
    ]


def test_guard_accepts_only_tenant_suffixes_with_matching_tenant_evidence(tmp_path):
    _catalog(tmp_path, "video_future_sv_chunk_9s")
    (tmp_path / "tests" / "example.py").write_text(
        'tenant_id = "org:tenant"\n'
        'schema_name = "video_future_sv_chunk_9s_org_tenant"\n'
        'schema_name = "video_future_sv_chunk_9s_unexplained"\n'
        'base_schema_name = "video_future_sv_chunk_9s_org_tenant"\n'
    )
    assert missing_schema_references(tmp_path) == [
        "tests/example.py:3: video_future_sv_chunk_9s_unexplained",
        "tests/example.py:4: video_future_sv_chunk_9s_org_tenant",
    ]


def test_guard_recognizes_synthetic_schema_definitions_without_exempting_neighbors(
    tmp_path,
):
    _catalog(tmp_path, "available")
    (tmp_path / "tests" / "example.py").write_text(
        'fixture = {"name": "fabricated", "document": {"fields": []}}\n'
        'backend.search(schema_name="fabricated")\n'
        'backend.search(schema_name="unrelated_missing")\n'
    )
    assert missing_schema_references(tmp_path) == [
        "tests/example.py:3: unrelated_missing"
    ]


def test_guard_recognizes_written_schemas_without_exempting_shipped_paths(tmp_path):
    _catalog(tmp_path, "available")
    (tmp_path / "tests" / "example.py").write_text(
        '(tmp_path / "fabricated_schema.json").write_text("{}")\n'
        'loader.load_schema("fabricated")\n'
        'path = "configs/schemas/fabricated_schema.json"\n'
    )
    assert missing_schema_references(tmp_path) == ["tests/example.py:3: fabricated"]


def test_guard_limits_declared_fixtures_to_exact_module_and_name(tmp_path):
    _catalog(tmp_path, "available")
    (tmp_path / "tests" / "example.py").write_text(
        'schema_name = "fabricated"\nschema_name = "future_missing"\n'
    )
    (tmp_path / "libs" / "example.py").write_text('schema_name = "fabricated"\n')
    assert missing_schema_references(
        tmp_path,
        synthetic_schemas={"tests/example.py": {"fabricated": "Parser input"}},
    ) == [
        "libs/example.py:1: fabricated",
        "tests/example.py:2: future_missing",
    ]


def test_guard_rejects_declarations_for_missing_shipped_files(tmp_path):
    _catalog(tmp_path, "available")
    (tmp_path / "tests" / "example.py").write_text(
        'path = "configs/schemas/fabricated_schema.json"\n'
    )
    assert missing_schema_references(
        tmp_path,
        synthetic_schemas={"tests/example.py": {"fabricated": "Parser input"}},
    ) == ["tests/example.py:1: fabricated"]


def test_guard_requires_exact_declarations_with_a_reason(tmp_path):
    _catalog(tmp_path, "available")
    with pytest.raises(
        ValueError, match="Synthetic schema declarations need exact names and reasons"
    ):
        missing_schema_references(
            tmp_path,
            synthetic_schemas={"tests/example.py": {"video_*": "Parser input"}},
        )
    with pytest.raises(
        ValueError, match="Synthetic schema declarations need exact names and reasons"
    ):
        missing_schema_references(
            tmp_path,
            synthetic_schemas={"tests/example.py": {"fabricated": ""}},
        )


def test_guard_reads_names_in_comments_and_docstrings(tmp_path):
    _catalog(tmp_path, "available")
    (tmp_path / "libs" / "example.py").write_text(
        '"""Use video_future_sv_chunk_9s."""\n'
        "# configs/schemas/another_future_schema.json\n"
    )
    assert missing_schema_references(tmp_path) == [
        "libs/example.py:1: video_future_sv_chunk_9s",
        "libs/example.py:2: another_future",
    ]


def test_detector_ignores_embedded_source_without_hiding_adjacent_references():
    source = (
        "snippet = 'schema_name = \"fabricated\"'; "
        'profiles = ["video_future_sv_chunk_9s"]'
    )
    assert schema_references(source) == {(1, "video_future_sv_chunk_9s")}


def test_guard_detects_a_removed_catalog_entry(tmp_path):
    _catalog(tmp_path, "available")
    (tmp_path / "libs" / "example.py").write_text('SCHEMA = "available"\n')
    assert missing_schema_references(tmp_path) == []
    (tmp_path / "configs" / "schemas" / "available_schema.json").rename(
        tmp_path / "configs" / "schemas" / "replacement_schema.json"
    )
    assert missing_schema_references(tmp_path) == ["libs/example.py:1: available"]
