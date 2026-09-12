"""Goldens for the harness turn helpers.

Every dispatch consumer (the wiki auto-file hook, the harness transports)
reads one answer string. These pin that string for each shape a shipped agent
produces, built from the real output types, plus the per-conversation seed and
the OpenAI tool-call conversion.
"""

from __future__ import annotations

import dataclasses
import importlib

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
            "Found 1 images for 'red bike'\n- img_7 · score 0.500: Red bike on a wall"
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
        from cogniverse_agents.entity_extraction_agent import (
            Entity,
            EntityExtractionOutput,
        )

        output = EntityExtractionOutput(
            query="who founded vespa",
            entities=[Entity(text="Vespa", type="TECHNOLOGY", confidence=0.91)],
            relationships=[],
            entity_count=1,
            has_entities=True,
            dominant_types=["TECHNOLOGY"],
            path_used="gliner",
        )
        envelope = {
            "status": "success",
            "agent": "entity_extraction_agent",
            **output.model_dump(),
        }
        assert extract_answer_text(envelope) == (
            '{"dominant_types": ["TECHNOLOGY"], '
            '"entities": [{"confidence": 0.91, "context": "", "text": "Vespa", '
            '"type": "TECHNOLOGY"}], '
            '"entity_count": 1, "has_entities": true, "path_used": "gliner", '
            '"query": "who founded vespa", "relationships": []}'
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
            "message": f"Found {output.total_results} results for "
            f"'{output.enhanced_query}'",
            "results_count": output.total_results,
            "results": output.results,
            "profile": "video_colpali_smol500_mv_frame",
            "search_mode": "hybrid",
        }
        assert extract_answer_text(envelope) == (
            "Found 2 results for 'red bicycles outdoors'\n"
            "- v_001 · score 0.912 · 12.0s-18.5s: Cyclist in a red jersey\n"
            "- v_002 · score 0.480: Red bike leaning on a wall"
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
