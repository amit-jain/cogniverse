"""Workflow agent ordering rules derived from configured agent roles."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from cogniverse_foundation.config.unified_config import AgentMappingRule
from cogniverse_synthetic.utils.agent_inference import AgentInferrer

pytestmark = pytest.mark.unit


def _mapping(modality: str, agent_name: str) -> AgentMappingRule:
    return AgentMappingRule(modality=modality, agent_name=agent_name)


def _video_agents(agent_name: str = "video_search") -> dict:
    return {
        agent_name: {
            "enabled": True,
            "modalities": ["VIDEO"],
            "capabilities": ["video_search"],
        }
    }


@pytest.fixture
def inferrer() -> AgentInferrer:
    return AgentInferrer(
        agents_config={
            "text_analysis_agent": {
                "enabled": True,
                "modalities": ["DOCUMENT"],
                "capabilities": ["text_analysis"],
            },
            "video_search": {
                "enabled": True,
                "modalities": ["VIDEO"],
                "capabilities": ["video_search"],
            },
            "document_agent": {
                "enabled": True,
                "modalities": ["DOCUMENT", "WIKI"],
                "capabilities": ["document_analysis"],
            },
            "coding_agent": {
                "enabled": True,
                "modalities": ["CODE"],
                "capabilities": ["coding"],
            },
            "image_search": {
                "enabled": True,
                "modalities": ["IMAGE"],
                "capabilities": ["image_search"],
            },
            "audio_analysis": {
                "enabled": True,
                "modalities": ["AUDIO"],
                "capabilities": ["audio_analysis"],
            },
            "summarizer": {
                "enabled": True,
                "modalities": [],
                "capabilities": ["summarize"],
                "roles": ["summarizer"],
            },
            "reporter": {
                "enabled": True,
                "modalities": [],
                "capabilities": ["report"],
                "roles": ["detailed_report"],
            },
        },
        agent_mappings=[
            _mapping("VIDEO", "video_search"),
            _mapping("DOCUMENT", "document_agent"),
            _mapping("IMAGE", "image_search"),
            _mapping("AUDIO", "audio_analysis"),
            _mapping("CODE", "coding_agent"),
            _mapping("WIKI", "document_agent"),
        ],
    )


def test_configured_modality_mappings_are_authoritative(
    inferrer: AgentInferrer,
) -> None:
    assert {
        modality: inferrer.infer_from_modality(modality)
        for modality in ("VIDEO", "DOCUMENT", "IMAGE", "AUDIO", "CODE", "WIKI")
    } == {
        "VIDEO": "video_search",
        "DOCUMENT": "document_agent",
        "IMAGE": "image_search",
        "AUDIO": "audio_analysis",
        "CODE": "coding_agent",
        "WIKI": "document_agent",
    }


@pytest.mark.parametrize(
    ("agent_mappings", "message"),
    [
        (None, "agent_mappings must be a non-empty list"),
        ([], "agent_mappings must be a non-empty list"),
        (
            [{"modality": "VIDEO", "agent_name": "video_search"}],
            r"agent_mappings\[0\] must be an AgentMappingRule",
        ),
        (
            [_mapping("video", "video_search")],
            "mapping modality must be one of: AUDIO, CODE, DOCUMENT, IMAGE, VIDEO, WIKI",
        ),
        (
            [_mapping("TEXT", "video_search")],
            "mapping modality must be one of: AUDIO, CODE, DOCUMENT, IMAGE, VIDEO, WIKI",
        ),
        (
            [_mapping("VIDEO", " ")],
            "mapping agent_name must be a non-empty string",
        ),
    ],
)
def test_mapping_shape_is_strict(agent_mappings, message) -> None:
    with pytest.raises(ValueError, match=message):
        AgentInferrer(
            agents_config=_video_agents(),
            agent_mappings=agent_mappings,
        )


def test_duplicate_and_conflicting_modality_mappings_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate agent mapping for modality 'VIDEO'",
    ):
        AgentInferrer(
            agents_config=_video_agents(),
            agent_mappings=[
                _mapping("VIDEO", "video_search"),
                _mapping("VIDEO", "video_search"),
            ],
        )

    agents = {
        **_video_agents("video_search"),
        **_video_agents("other_video_search"),
    }
    with pytest.raises(
        ValueError,
        match=(
            "conflicting agent mappings for modality 'VIDEO': "
            "'video_search' and 'other_video_search'"
        ),
    ):
        AgentInferrer(
            agents_config=agents,
            agent_mappings=[
                _mapping("VIDEO", "video_search"),
                _mapping("VIDEO", "other_video_search"),
            ],
        )


@pytest.mark.parametrize(
    ("agents_config", "message"),
    [
        (
            {
                "summarizer": {
                    "enabled": True,
                    "modalities": [],
                    "capabilities": ["summarization"],
                }
            },
            "mapping for modality 'VIDEO' targets unknown agent 'video_search'",
        ),
        (
            {
                "video_search": {
                    "enabled": False,
                    "modalities": ["VIDEO"],
                    "capabilities": ["video_search"],
                },
                "summarizer": {
                    "enabled": True,
                    "modalities": [],
                    "capabilities": ["summarization"],
                },
            },
            "mapping for modality 'VIDEO' targets disabled agent 'video_search'",
        ),
        (
            {
                "video_search": {
                    "enabled": True,
                    "modalities": ["DOCUMENT"],
                    "capabilities": ["video_search"],
                }
            },
            "agent 'video_search' does not declare mapped modality 'VIDEO'",
        ),
        (
            {
                "video_search": {
                    "enabled": True,
                    "modalities": ["VIDEO"],
                    "capabilities": ["search"],
                }
            },
            (
                "agent 'video_search' does not declare required capability "
                "'video_search' for modality 'VIDEO'"
            ),
        ),
    ],
)
def test_mapping_target_must_match_enabled_agent_contract(
    agents_config,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        AgentInferrer(
            agents_config=agents_config,
            agent_mappings=[_mapping("VIDEO", "video_search")],
        )


def test_concurrent_inferrers_do_not_bleed_configured_mappings() -> None:
    worker_count = 12
    start = threading.Barrier(worker_count)
    inferrers = {
        name: AgentInferrer(
            agents_config=_video_agents(name),
            agent_mappings=[_mapping("VIDEO", name)],
        )
        for name in ("tenant_a_video", "tenant_b_video")
    }
    expected = [
        "tenant_a_video" if index % 2 == 0 else "tenant_b_video"
        for index in range(worker_count)
    ]

    def resolve(agent_name: str) -> str:
        start.wait()
        return inferrers[agent_name].infer_from_modality("VIDEO")

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        actual = list(pool.map(resolve, expected))

    assert actual == expected


def test_code_and_wiki_characteristics_route_to_canonical_agents(
    inferrer: AgentInferrer,
) -> None:
    assert (
        inferrer.infer_from_characteristics(
            {"profile_type": "code", "modality": "CODE"}
        )
        == "coding_agent"
    )
    assert (
        inferrer.infer_from_characteristics(
            {"profile_type": "wiki", "modality": "WIKI"}
        )
        == "document_agent"
    )


@pytest.mark.parametrize(
    ("modality", "agent_name", "modalities", "capabilities", "message"),
    [
        (
            "CODE",
            "coding_agent",
            ["CODE"],
            ["code_search"],
            "agent 'coding_agent' does not declare required capability "
            "'coding' for modality 'CODE'",
        ),
        (
            "WIKI",
            "document_agent",
            ["DOCUMENT"],
            ["document_analysis"],
            "agent 'document_agent' does not declare mapped modality 'WIKI'",
        ),
    ],
)
def test_code_and_wiki_targets_require_exact_declared_contract(
    modality,
    agent_name,
    modalities,
    capabilities,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        AgentInferrer(
            agents_config={
                agent_name: {
                    "enabled": True,
                    "modalities": modalities,
                    "capabilities": capabilities,
                }
            },
            agent_mappings=[_mapping(modality, agent_name)],
        )


def test_search_agent_must_precede_secondary_agents(
    inferrer: AgentInferrer,
) -> None:
    assert inferrer.validate_agent_sequence(["video_search", "summarizer"]) is True
    assert inferrer.validate_agent_sequence(["summarizer", "video_search"]) is False


def test_secondary_only_sequence_is_invalid(inferrer: AgentInferrer) -> None:
    assert inferrer.validate_agent_sequence(["summarizer"]) is False


def test_search_only_sequence_is_valid(inferrer: AgentInferrer) -> None:
    assert inferrer.validate_agent_sequence(["video_search"]) is True


def test_unknown_agent_sequence_is_invalid(inferrer: AgentInferrer) -> None:
    assert inferrer.validate_agent_sequence(["video_search", "missing"]) is False


def test_empty_agent_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="agents configuration has no enabled agents"):
        AgentInferrer(
            agents_config={},
            agent_mappings=[_mapping("VIDEO", "video_search")],
        )


def test_agent_configuration_must_be_passed_explicitly() -> None:
    with pytest.raises(ValueError) as exc_info:
        AgentInferrer(
            agents_config=None,
            agent_mappings=[_mapping("VIDEO", "video_search")],
        )

    assert str(exc_info.value) == "agents_config is required"


def test_unconfigured_modality_is_rejected_without_named_fallback() -> None:
    inferrer = AgentInferrer(
        agents_config={
            "paper_finder": {
                "enabled": True,
                "modalities": ["DOCUMENT"],
                "capabilities": ["document_analysis"],
            }
        },
        agent_mappings=[_mapping("DOCUMENT", "paper_finder")],
    )

    with pytest.raises(
        ValueError,
        match="no configured agent mapping for modality 'VIDEO'",
    ):
        inferrer.infer_from_modality("VIDEO")


def test_roles_are_derived_from_capabilities_with_configured_names() -> None:
    inferrer = AgentInferrer(
        agents_config={
            "moving_picture_finder": {
                "enabled": True,
                "modalities": ["VIDEO"],
                "capabilities": ["video_search"],
            },
            "shortener": {
                "enabled": True,
                "modalities": [],
                "capabilities": ["summarization"],
            },
            "investigator": {
                "enabled": True,
                "modalities": [],
                "capabilities": ["detailed_report"],
            },
        },
        agent_mappings=[_mapping("VIDEO", "moving_picture_finder")],
    )

    assert inferrer.infer_workflow_sequence("moderate", "VIDEO", "summarize") == [
        "moving_picture_finder",
        "shortener",
    ]
    assert inferrer.get_agent_for_task("write a detailed report") == "investigator"


def test_duplicate_and_conflicting_explicit_role_mappings_are_rejected() -> None:
    base_video = _video_agents()
    with pytest.raises(
        ValueError,
        match="duplicate explicit agent role 'summarizer' for 'shortener'",
    ):
        AgentInferrer(
            agents_config={
                **base_video,
                "shortener": {
                    "enabled": True,
                    "modalities": [],
                    "capabilities": ["summarization"],
                    "roles": ["summarizer", "summarizer"],
                },
            },
            agent_mappings=[_mapping("VIDEO", "video_search")],
        )

    with pytest.raises(
        ValueError,
        match=(
            "conflicting explicit agent role 'summarizer': "
            "'shortener' and 'other_shortener'"
        ),
    ):
        AgentInferrer(
            agents_config={
                **base_video,
                "shortener": {
                    "enabled": True,
                    "modalities": [],
                    "capabilities": [],
                    "roles": ["summarizer"],
                },
                "other_shortener": {
                    "enabled": True,
                    "modalities": [],
                    "capabilities": [],
                    "roles": ["summarizer"],
                },
            },
            agent_mappings=[_mapping("VIDEO", "video_search")],
        )


@pytest.mark.parametrize("complexity", ["medium", " moderate", "COMPLEX", True])
def test_workflow_complexity_requires_exact_canonical_value(
    inferrer: AgentInferrer, complexity
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "query_complexity must be one of: complex, moderate, simple; "
            f"got {complexity!r}"
        ),
    ):
        inferrer.infer_workflow_sequence(complexity, "VIDEO", "summarize")


@pytest.mark.parametrize("task_type", ["summary", " summarize", "ANALYZE", True])
def test_workflow_task_type_requires_exact_canonical_value(
    inferrer: AgentInferrer, task_type
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "task_type must be one of: analyze, search, summarize, or None; "
            f"got {task_type!r}"
        ),
    ):
        inferrer.infer_workflow_sequence("moderate", "VIDEO", task_type)


def test_only_explicitly_enabled_agents_are_available() -> None:
    inferrer = AgentInferrer(
        agents_config={
            "paper_finder": {
                "enabled": True,
                "modalities": ["DOCUMENT"],
                "capabilities": ["document_analysis"],
            },
            "disabled_video_finder": {
                "enabled": False,
                "modalities": ["VIDEO"],
                "capabilities": ["search"],
            },
            "implicitly_enabled_audio_finder": {
                "modalities": ["AUDIO"],
                "capabilities": ["search"],
            },
        },
        agent_mappings=[_mapping("DOCUMENT", "paper_finder")],
    )

    assert inferrer.AGENT_CAPABILITIES == {
        "paper_finder": {
            "modalities": ["DOCUMENT"],
            "capabilities": ["document_analysis"],
        }
    }
    with pytest.raises(ValueError) as disabled_video:
        inferrer.infer_from_modality("VIDEO")
    assert str(disabled_video.value) == (
        "no configured agent mapping for modality 'VIDEO'"
    )
    with pytest.raises(ValueError) as missing_enabled_audio:
        inferrer.infer_from_modality("AUDIO")
    assert str(missing_enabled_audio.value) == (
        "no configured agent mapping for modality 'AUDIO'"
    )


def test_content_without_a_modality_is_rejected(inferrer: AgentInferrer) -> None:
    with pytest.raises(ValueError) as exc_info:
        inferrer.infer_from_characteristics(
            {"schema_name": "segments", "segment_description": "Marie Curie"}
        )

    assert str(exc_info.value) == "content requires profile_type and modality"


def test_content_with_inconsistent_profile_modality_is_rejected(
    inferrer: AgentInferrer,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        inferrer.infer_from_characteristics(
            {
                "schema_name": "opaque",
                "profile_type": "video",
                "modality": "AUDIO",
            }
        )

    assert str(exc_info.value) == (
        "content modality 'AUDIO' does not match profile_type 'video'"
    )


def test_keyword_free_content_routes_from_exact_profile_modality(
    inferrer: AgentInferrer,
) -> None:
    assert (
        inferrer.infer_from_characteristics(
            {
                "schema_name": "alpha",
                "embedding_type": "dense",
                "description": "opaque corpus item",
                "profile_type": "document",
                "modality": "DOCUMENT",
            }
        )
        == "document_agent"
    )


def test_search_task_without_a_modality_is_rejected(
    inferrer: AgentInferrer,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        inferrer.get_agent_for_task("find the latest material")

    assert str(exc_info.value) == "cannot infer modality from search task"


def test_search_task_with_multiple_modalities_is_rejected(
    inferrer: AgentInferrer,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        inferrer.get_agent_for_task("find the video and podcast")

    assert str(exc_info.value) == (
        "search task describes multiple modalities: AUDIO, VIDEO"
    )


def test_task_without_a_configured_role_or_modality_is_rejected(
    inferrer: AgentInferrer,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        inferrer.get_agent_for_task("process this material")

    assert str(exc_info.value) == "cannot infer enabled agent from task description"
