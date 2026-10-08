"""Goldens for the harness turn helpers.

Every dispatch consumer (the wiki auto-file hook, the harness transports)
reads one answer string. These pin that string for each shape a shipped agent
produces, built from the real output types, plus the per-conversation seed and
the OpenAI tool-call conversion.
"""

from __future__ import annotations

import dataclasses
import importlib
import json

import pytest

from cogniverse_runtime.config_loader import ConfigLoader
from cogniverse_runtime.harness_turn import (
    NoAnswerError,
    ToolCallShapeError,
    answer_fields_for,
    derive_request_seed,
    extract_answer_text,
    to_openai_tool_calls,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _declared_fields(output_type: type) -> list:
    model_fields = getattr(output_type, "model_fields", None)
    if model_fields:
        return list(model_fields)
    if dataclasses.is_dataclass(output_type):
        return [f.name for f in dataclasses.fields(output_type)]
    return []


def _shipped_output_types():
    """(agent, output-class, declared fields) for every agent the runtime can
    load, resolved by the same module convention the dispatcher uses."""
    for agent_name, class_path in sorted(ConfigLoader.AGENT_CLASSES.items()):
        module_path, _ = class_path.split(":")
        module = importlib.import_module(module_path)
        for attr in sorted(dir(module)):
            obj = getattr(module, attr)
            if not isinstance(obj, type):
                continue
            if not (attr.endswith("Output") or attr.endswith("Result")):
                continue
            if getattr(obj, "__module__", None) != module_path:
                continue
            fields = _declared_fields(obj)
            if not fields:
                continue
            yield agent_name, attr, fields


class TestShippedOutputCoverage:
    """The whole map from shipped output type to the fields the extractor
    renders. A new agent, a renamed answer field, or an output type the
    extractor has no rule for shows up as a diff here, naming the type."""

    def test_every_shipped_output_type_maps_to_its_answer_fields(self):
        resolved = {
            (agent, cls): answer_fields_for(fields)
            for agent, cls, fields in _shipped_output_types()
        }
        assert resolved == {
            ("audio_analysis_agent", "AudioResult"): (),
            ("audio_analysis_agent", "AudioSearchOutput"): (),
            ("audio_analysis_agent", "TranscriptionResult"): (),
            ("audit_explanation_agent", "AuditExplanationOutput"): ("explanation",),
            ("citation_tracing_agent", "CitationTracingOutput"): (),
            ("coding_agent", "CodingOutput"): ("summary",),
            ("coding_agent", "ExecutionResult"): (),
            (
                "contradiction_reconciliation_agent",
                "ContradictionReconciliationOutput",
            ): (),
            ("cross_tenant_comparison_agent", "CrossTenantComparisonOutput"): (),
            ("deep_research_agent", "DeepResearchOutput"): ("summary",),
            ("detailed_report_agent", "DetailedReportOutput"): (
                "executive_summary",
                "detailed_findings",
            ),
            ("detailed_report_agent", "ReportResult"): (
                "executive_summary",
                "detailed_findings",
            ),
            ("document_agent", "DocumentResult"): (),
            ("document_agent", "DocumentSearchOutput"): (),
            ("entity_extraction_agent", "EntityExtractionOutput"): (),
            ("federated_query_agent", "FederatedQueryOutput"): ("summary",),
            ("gateway_agent", "GatewayOutput"): (),
            ("image_search_agent", "ImageResult"): (),
            ("image_search_agent", "ImageSearchOutput"): (),
            ("kg_traversal_agent", "KGTraversalOutput"): ("summary",),
            ("knowledge_summarization_agent", "KnowledgeSummarizationOutput"): (
                "summary",
            ),
            ("multi_document_synthesis_agent", "MultiDocSynthesisOutput"): ("answer",),
            ("orchestrator_agent", "OrchestrationResult"): ("final_output",),
            ("orchestrator_agent", "OrchestratorOutput"): ("final_output",),
            ("profile_selection_agent", "ProfileSelectionOutput"): (),
            ("query_enhancement_agent", "QueryEnhancementOutput"): (),
            ("search_agent", "SearchOutput"): (),
            ("summarizer_agent", "SummarizerOutput"): ("summary",),
            ("summarizer_agent", "SummaryResult"): ("summary",),
            ("temporal_reasoning_agent", "TemporalReasoningOutput"): ("summary",),
        }

    def test_types_without_a_declared_answer_field_render_from_the_envelope(self):
        """The () entries above are not "unrenderable": their dispatch envelope
        carries the text (message + hits) or renders as structured JSON. Pin
        that for the search family, which is every () entry that ships hits."""
        from cogniverse_agents.image_search_agent import ImageResult, ImageSearchOutput

        hit = ImageResult(
            image_id="img_7",
            image_url="s3://bucket/img_7.jpg",
            title="Red bike on a wall",
            relevance_score=0.5,
        )
        output = ImageSearchOutput(results=[hit], count=1)
        envelope = {
            "status": "success",
            "agent": "image_search_agent",
            "message": "Found 1 images for 'red bike'",
            "results_count": output.count,
            "results": [r.model_dump() for r in output.results],
        }
        assert extract_answer_text(envelope) == (
            "Found 1 images for 'red bike'\n- Red bike on a wall"
        )


class TestAnswerGoldens:
    def test_nested_result_payload_coding_envelope(self):
        from cogniverse_agents.coding_agent import CodingOutput

        output = CodingOutput(
            plan="1. rename the field 2. rerun the suite",
            summary="Renamed the field and reran the suite.",
            iterations_used=1,
        )
        envelope = {
            "status": "success",
            "agent": "coding_agent",
            "message": "Coding task complete for 'rename the field'",
            "result": output.model_dump(),
        }
        assert extract_answer_text(envelope) == "Renamed the field and reran the suite."

    def test_flat_model_dump_with_a_declared_answer_field(self):
        from cogniverse_agents.multi_document_synthesis_agent import (
            MultiDocSynthesisOutput,
        )

        output = MultiDocSynthesisOutput(
            answer="Both filings name the same supplier.",
            citation_refs=[{"memory_id": "mem-1"}, {"memory_id": "mem-2"}],
        )
        envelope = {
            "status": "success",
            "agent": "multi_document_synthesis_agent",
            **output.model_dump(),
        }
        assert extract_answer_text(envelope) == "Both filings name the same supplier."

    def test_flat_model_dump_without_text_renders_the_payload(self):
        from cogniverse_agents.citation_tracing_agent import CitationTracingOutput

        output = CitationTracingOutput(root_memory_id="mem-1")
        envelope = {
            "status": "success",
            "agent": "citation_tracing_agent",
            **output.model_dump(),
        }
        assert extract_answer_text(envelope) == json.dumps(
            output.model_dump(), sort_keys=True, default=str
        )

    def test_entity_extraction_reads_as_a_sentence(self):
        """A live entity run: every entity by name and type, then each
        relationship, never the payload as JSON."""
        from cogniverse_agents.entity_extraction_agent import (
            Entity,
            EntityExtractionOutput,
            Relationship,
        )

        output = EntityExtractionOutput(
            query="Daenerys burned the castle at King's Landing, shown on HBO GO.",
            entities=[
                Entity(text="Daenerys", type="PERSON"),
                Entity(text="castle", type="CONCEPT"),
                Entity(text="King's Landing", type="PLACE"),
                Entity(text="HBO GO", type="ORGANIZATION"),
            ],
            relationships=[
                Relationship(
                    subject="Daenerys", relation="burn", object="castle", confidence=0.8
                ),
                Relationship(
                    subject="Daenerys",
                    relation="at",
                    object="King's Landing",
                    confidence=0.7,
                ),
            ],
            entity_count=4,
            has_entities=True,
            dominant_types=["PERSON", "CONCEPT", "PLACE"],
            path_used="dspy",
        )
        envelope = {
            "status": "success",
            "agent": "entity_extraction_agent",
            **output.model_dump(),
        }
        assert extract_answer_text(envelope) == (
            "Found 4 entities: Daenerys (person), castle (concept), "
            "King's Landing (place), HBO GO (organization). "
            "Relationships: Daenerys burn castle; Daenerys at King's Landing."
        )

    def test_no_entities_reads_as_a_sentence(self):
        from cogniverse_agents.entity_extraction_agent import EntityExtractionOutput

        output = EntityExtractionOutput(query="hello there", entity_count=0)
        envelope = {
            "status": "success",
            "agent": "entity_extraction_agent",
            **output.model_dump(),
        }
        assert extract_answer_text(envelope) == "Found no entities."

    @pytest.mark.parametrize(
        ("original", "enhanced", "reply"),
        [
            (
                "fire castle video",
                "burning castle video footage",
                'Enhanced "fire castle video" to "burning castle video footage".',
            ),
            (
                "fire castle video",
                "fire castle video",
                'Kept the query as asked: "fire castle video".',
            ),
        ],
    )
    def test_query_enhancement_reads_as_a_sentence(self, original, enhanced, reply):
        from cogniverse_agents.query_enhancement_agent import QueryEnhancementOutput

        output = QueryEnhancementOutput(
            original_query=original,
            enhanced_query=enhanced,
            expansion_terms=["footage"],
            query_variants=[enhanced],
            confidence=0.8,
            path_used="lm",
        )
        envelope = {
            "status": "success",
            "agent": "query_enhancement_agent",
            **output.model_dump(),
        }
        assert extract_answer_text(envelope) == reply

    def test_profile_selection_reads_as_a_sentence(self):
        from cogniverse_agents.profile_selection_agent import ProfileSelectionOutput

        output = ProfileSelectionOutput(
            query="Which profile for a video of a burning castle?",
            selected_profile="video_colpali_smol500_mv_frame",
            confidence=0.95,
            reasoning="Video search over frames.",
            query_intent="video_search",
            modality="video",
        )
        envelope = {
            "status": "success",
            "agent": "profile_selection_agent",
            **output.model_dump(),
        }
        assert extract_answer_text(envelope) == (
            "Selected profile video_colpali_smol500_mv_frame for a video search "
            "(confidence 0.95)."
        )

    def test_orchestrator_deep_synthesis_final_output(self):
        from cogniverse_agents.orchestrator_agent import OrchestratorOutput

        output = OrchestratorOutput(
            query="what changed in the filings",
            workflow_id="wf-17",
            final_output={
                "answer": "The supplier list grew by two names.",
                "iterations_used": 2,
                "subagent_calls_made": 3,
            },
        )
        envelope = {
            "status": "success",
            "agent": "orchestrator_agent",
            "message": "Orchestrated 'what changed in the filings' via A2A pipeline",
            "orchestration_result": output.model_dump(),
            "gateway_context": None,
        }
        assert extract_answer_text(envelope) == "The supplier list grew by two names."

    def test_orchestrator_fusion_aggregated_content(self):
        from cogniverse_agents.orchestrator_agent import OrchestratorOutput

        output = OrchestratorOutput(
            query="red bikes",
            workflow_id="wf-18",
            final_output={
                "query": "red bikes",
                "status": "success",
                "results": {"search_agent": {"results_count": 2}},
                "fusion_strategy": "simple",
                "fusion_quality": {"strategy": "simple", "modality_count": 1},
                "aggregated_content": "Two clips show a red bike on a wall.",
            },
        )
        envelope = {
            "status": "success",
            "agent": "orchestrator_agent",
            "message": "Orchestrated 'red bikes' via A2A pipeline",
            "orchestration_result": output.model_dump(),
            "gateway_context": None,
        }
        assert extract_answer_text(envelope) == "Two clips show a red bike on a wall."

    def test_detailed_report_joins_summary_and_findings(self):
        from cogniverse_agents.detailed_report_agent import DetailedReportOutput

        output = DetailedReportOutput(
            executive_summary="Throughput fell 12% after the June rollout.",
            detailed_findings=[
                {
                    "category": "Content Analysis",
                    "finding": "Latency p95 doubled on the ingest path",
                    "details": {"total_results": 4},
                    "significance": "high",
                },
                {
                    "category": "Patterns Identified",
                    "finding": "2 patterns detected",
                    "details": ["batching", "retry storm"],
                    "significance": "medium",
                },
            ],
            recommendations=["Roll back the batch size change."],
        )
        envelope = {
            "status": "success",
            "agent": "detailed_report_agent",
            "message": "Generated detailed report for 'throughput'",
            "result": output.model_dump(),
        }
        assert extract_answer_text(envelope) == (
            "Throughput fell 12% after the June rollout.\n\n"
            "- Content Analysis: Latency p95 doubled on the ingest path\n"
            "- Patterns Identified: 2 patterns detected"
        )

    def test_summarizer_result(self):
        from cogniverse_agents.summarizer_agent import SummaryResult, ThinkingPhase

        result = SummaryResult(
            summary="The clip shows a cyclist crossing a bridge.",
            key_points=["cyclist", "bridge"],
            visual_insights=["red jersey"],
            confidence_score=0.82,
            thinking_phase=ThinkingPhase(
                key_themes=["cycling"],
                content_categories=["video"],
                relevance_scores={"v_001": 0.9},
                visual_elements=["bridge"],
                reasoning="Two clips share the bridge setting.",
            ),
            metadata={"result_count": 2},
        )
        envelope = {
            "status": "success",
            "agent": "summarizer_agent",
            "message": "Generated summary for 'the clip'",
            "result": dataclasses.asdict(result),
        }
        assert (
            extract_answer_text(envelope)
            == "The clip shows a cyclist crossing a bridge."
        )

    def test_search_envelope_renders_the_hits(self):
        from cogniverse_agents.search_agent import SearchOutput

        output = SearchOutput(
            query="red bikes",
            enhanced_query="red bicycles outdoors",
            results=[
                {
                    "document_id": "v_001",
                    "score": 0.9123,
                    "temporal_info": {"start_time": 12.0, "end_time": 18.5},
                    "metadata": {"title": "Cyclist in a red jersey"},
                },
                {
                    "document_id": "v_002",
                    "score": 0.48,
                    "title": "Red bike leaning on a wall",
                },
            ],
            total_results=2,
        )
        envelope = {
            "status": "success",
            "agent": "search_agent",
            "message": f"Found {output.total_results} results for '{output.query}'",
            "results_count": output.total_results,
            "results": output.results,
            "profile": "video_colpali_smol500_mv_frame",
            "search_mode": "hybrid",
        }
        assert extract_answer_text(envelope) == (
            "Found 2 results for 'red bikes'\n"
            "- Cyclist in a red jersey (0:12–0:18)\n"
            "- Red bike leaning on a wall"
        )

    def test_video_hits_read_by_title_and_time_never_by_document_id(self):
        """Live search hits as the runtime serves them: each line names
        the video and the segment's time range, then the first line of the
        frame's description; no backend document id reaches the reply."""
        hits = [
            {
                "id": "57985f49_seg_3",
                "document_id": "id:content:video_colpali_smol500_mv_frame_t_main"
                "::57985f49_seg_3",
                "score": 14.826,
                "metadata": {
                    "video_id": "57985f49",
                    "video_title": "for_bigger_blazes.mp4",
                    "segment_description": "This is a still frame from a video "
                    "displayed on a television screen.\n\n**Objects and Scene "
                    "Setting:**\nA burning castle.",
                    "audio_transcript": "Dracarys.",
                },
                "temporal_info": {"start_time": 5.880875, "end_time": 6.880875},
            },
            {
                "id": "2bcd2065_seg_3",
                "document_id": "id:content:video_colpali_smol500_mv_frame_t_main"
                "::2bcd2065_seg_3",
                "score": 10.388,
                "metadata": {
                    "video_id": "2bcd2065",
                    "source_title": "v_-nl4G-00PtA.mp4",
                    "segment_description": "A medium close-up of a young man "
                    + "standing in a brightly lit kitchen " * 8,
                },
                "temporal_info": {"start_time": 66, "end_time": 67},
            },
            {
                "id": "dd95bb38_seg_0",
                "document_id": "id:content:video_colpali_smol500_mv_frame_t_main"
                "::dd95bb38_seg_0",
                "score": 9.129,
                "metadata": {"video_id": "dd95bb38"},
                "temporal_info": {"start_time": 0.0, "end_time": 1.0},
            },
        ]
        envelope = {
            "status": "success",
            "agent": "search_agent",
            "message": "Found 3 results for 'videos of a burning castle'",
            "results_count": 3,
            "results": hits,
        }
        assert extract_answer_text(envelope) == (
            "Found 3 results for 'videos of a burning castle'\n"
            "- for_bigger_blazes.mp4 (0:05–0:06): This is a still frame from a "
            "video displayed on a television screen.\n"
            "- v_-nl4G-00PtA.mp4 (1:06–1:07): A medium close-up of a young man "
            "standing in a brightly lit kitchen standing in a brightly lit "
            "kitchen standing in a brightly lit kitchen standing in a brightly "
            "lit kitchen standing in a brightly lit…\n"
            "- Result 3 (0:00–0:01)"
        )

    def test_gateway_wrapper_unwraps_the_downstream_answer(self):
        from cogniverse_agents.summarizer_agent import SummaryResult, ThinkingPhase

        downstream = {
            "status": "success",
            "agent": "summarizer_agent",
            "message": "Generated summary for 'the clip'",
            "result": dataclasses.asdict(
                SummaryResult(
                    summary="A cyclist crosses a bridge at dusk.",
                    key_points=[],
                    visual_insights=[],
                    confidence_score=0.5,
                    thinking_phase=ThinkingPhase(
                        key_themes=[],
                        content_categories=[],
                        relevance_scores={},
                        visual_elements=[],
                        reasoning="",
                    ),
                    metadata={},
                )
            ),
        }
        envelope = dict(downstream)
        envelope["agent"] = "gateway_agent"
        envelope["downstream_result"] = downstream
        envelope["gateway"] = {"complexity": "simple", "routed_to": "summarizer_agent"}
        assert extract_answer_text(envelope) == "A cyclist crosses a bridge at dusk."


class TestErrorEnvelopesAreNotAnswers:
    def test_a2a_error_envelope_raises_naming_the_agent_and_cause(self):
        envelope = {
            "status": "error",
            "error": "SearchBackendError: connection refused",
            "agent": "search_agent",
        }
        with pytest.raises(NoAnswerError) as excinfo:
            extract_answer_text(envelope)
        assert str(excinfo.value) == (
            "agent 'search_agent' reported status=error: "
            "SearchBackendError: connection refused"
        )
        assert excinfo.value.status == "error"

    def test_orchestrator_error_final_output_is_not_an_answer(self):
        from cogniverse_agents.orchestrator_agent import OrchestratorOutput

        output = OrchestratorOutput(
            query="",
            workflow_id="wf-19",
            final_output={"status": "error", "message": "Empty query"},
        )
        envelope = {
            "status": "success",
            "agent": "orchestrator_agent",
            "message": "Orchestrated '' via A2A pipeline",
            "orchestration_result": output.model_dump(),
        }
        with pytest.raises(NoAnswerError) as excinfo:
            extract_answer_text(envelope)
        assert str(excinfo.value) == (
            "'final_output' reported status=error: Empty query"
        )

    def test_empty_envelope_raises_naming_its_keys(self):
        with pytest.raises(NoAnswerError) as excinfo:
            extract_answer_text({"status": "success", "agent": "search_agent"})
        assert str(excinfo.value) == (
            "no answer text in result with keys ['agent', 'status']"
        )


class TestRequestSeed:
    SYSTEM = "You are Pi, a coding assistant. Be terse."

    def test_seed_anchors_on_the_first_user_message_not_the_system_prompt(self):
        """Two conversations from one system-prompting client must not share a
        bucket, and the seed of a system-prefixed conversation equals the seed
        of the same first user message with no history."""
        first = derive_request_seed(
            "follow up",
            [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": "where is the red bike"},
            ],
        )
        second = derive_request_seed(
            "follow up",
            [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": "summarize the audio"},
            ],
        )
        assert first == "bc261c7719f14682036d23122244d65a"
        assert second == "bccc69b171c72805b7d5cb6877141c23"
        assert first == derive_request_seed("where is the red bike", [])

    def test_seed_is_stable_across_the_turns_of_one_conversation(self):
        turn_one = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": "where is the red bike"},
        ]
        turn_three = turn_one + [
            {"role": "assistant", "content": "On the wall in clip v_002."},
            {"role": "user", "content": "and the blue one"},
            {"role": "assistant", "content": "Not in the corpus."},
            {"role": "user", "content": "try again"},
        ]
        seed = derive_request_seed("where is the red bike", turn_one)
        assert derive_request_seed("try again", turn_three) == seed
        assert seed == "bc261c7719f14682036d23122244d65a"

    def test_seed_ignores_per_turn_metadata_on_the_replayed_message(self):
        assert (
            derive_request_seed(
                "try again",
                [
                    {"role": "system", "content": self.SYSTEM},
                    {
                        "role": "user",
                        "content": "where is the red bike",
                        "name": "amit",
                        "id": "msg-88",
                    },
                ],
            )
            == "bc261c7719f14682036d23122244d65a"
        )

    def test_no_history_seeds_from_the_query(self):
        assert (
            derive_request_seed("where is the red bike", [])
            == "bc261c7719f14682036d23122244d65a"
        )
        assert (
            derive_request_seed("summarize the audio", [])
            == "bccc69b171c72805b7d5cb6877141c23"
        )


class TestOpenAIToolCalls:
    def test_conversion_golden(self):
        assert to_openai_tool_calls(
            [
                {
                    "id": "call_1",
                    "name": "read_file",
                    "arguments": {"path": "libs/a.py"},
                },
                {"id": "call_2", "name": "list_dir"},
            ]
        ) == [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "libs/a.py"}',
                },
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "list_dir", "arguments": "{}"},
            },
        ]

    def test_missing_key_names_the_index_and_the_key(self):
        with pytest.raises(ToolCallShapeError) as excinfo:
            to_openai_tool_calls(
                [
                    {"id": "call_1", "name": "read_file"},
                    {"id": "call_2", "arguments": {}},
                ]
            )
        assert str(excinfo.value) == (
            "pending_tool_calls[1] is missing 'name'; has ['arguments', 'id']"
        )

    def test_non_mapping_call_names_the_index_and_the_type(self):
        with pytest.raises(ToolCallShapeError) as excinfo:
            to_openai_tool_calls([{"id": "call_1", "name": "read_file"}, "call_2"])
        assert str(excinfo.value) == (
            "pending_tool_calls[1] is str, expected a mapping with 'id' and 'name'"
        )


@pytest.mark.parametrize("status", ["error", "failed"])
def test_orchestration_failed_status_cannot_render_answer(status):
    payload = {
        "agent": "orchestrator_agent",
        "final_output": {
            "status": status,
            "message": "No orchestration step completed successfully",
            "aggregated_content": "untrusted child error text",
        },
    }
    with pytest.raises(NoAnswerError) as raised:
        extract_answer_text(payload)
    assert raised.value.status == status
    assert str(raised.value) == (
        f"'final_output' reported status={status}: "
        "No orchestration step completed successfully"
    )


def test_orchestration_partial_status_preserves_valid_answer():
    assert (
        extract_answer_text(
            {
                "agent": "orchestrator_agent",
                "status": "partial",
                "orchestration_result": {
                    "final_output": {"status": "partial", "aggregated_content": "42"}
                },
            }
        )
        == "42"
    )
