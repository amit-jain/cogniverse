"""Tests for RLM wiring in DetailedReportAgent.

Verifies that DetailedReportAgent inherits RLMAwareMixin and its input/output
schemas expose the rlm, rlm_synthesis, and rlm_telemetry fields.
"""

from cogniverse_agents.detailed_report_agent import (
    DetailedReportAgent,
    DetailedReportInput,
    DetailedReportOutput,
)
from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin
from cogniverse_core.agents.rlm_options import RLMOptions


class TestDetailedReportAgentRLMMixin:
    """DetailedReportAgent must inherit RLMAwareMixin."""

    def test_is_subclass_of_rlm_aware_mixin(self):
        assert issubclass(DetailedReportAgent, RLMAwareMixin)

    def test_rlm_mixin_precedes_a2a_agent_in_mro(self):
        mro_names = [cls.__name__ for cls in DetailedReportAgent.__mro__]
        rlm_idx = mro_names.index("RLMAwareMixin")
        a2a_idx = mro_names.index("A2AAgent")
        assert rlm_idx < a2a_idx, (
            "RLMAwareMixin must appear before A2AAgent in MRO "
            f"(RLMAwareMixin={rlm_idx}, A2AAgent={a2a_idx})"
        )


class TestDetailedReportInputRLMField:
    """DetailedReportInput must expose an optional rlm field."""

    def test_rlm_field_exists(self):
        fields = DetailedReportInput.model_fields
        assert "rlm" in fields, "DetailedReportInput must have 'rlm' field"

    def test_rlm_field_is_optional(self):
        fields = DetailedReportInput.model_fields
        field = fields["rlm"]
        assert not field.is_required(), "rlm field must be optional (not required)"

    def test_rlm_field_defaults_to_none(self):
        inp = DetailedReportInput(query="test", search_results=[])
        assert inp.rlm is None

    def test_rlm_field_accepts_rlm_options(self):
        opts = RLMOptions(enabled=True)
        inp = DetailedReportInput(query="test", search_results=[], rlm=opts)
        assert inp.rlm is opts
        assert inp.rlm.enabled is True

    def test_rlm_field_type_annotation(self):
        import typing

        fields = DetailedReportInput.model_fields
        annotation = fields["rlm"].annotation
        # Should be Optional[RLMOptions] — i.e. RLMOptions | None
        args = typing.get_args(annotation)
        assert RLMOptions in args, (
            f"rlm field annotation should include RLMOptions, got {annotation}"
        )


class TestDetailedReportOutputRLMFields:
    """DetailedReportOutput must expose rlm_synthesis and rlm_telemetry fields."""

    def test_rlm_synthesis_field_exists(self):
        fields = DetailedReportOutput.model_fields
        assert "rlm_synthesis" in fields, (
            "DetailedReportOutput must have 'rlm_synthesis' field"
        )

    def test_rlm_telemetry_field_exists(self):
        fields = DetailedReportOutput.model_fields
        assert "rlm_telemetry" in fields, (
            "DetailedReportOutput must have 'rlm_telemetry' field"
        )

    def test_rlm_synthesis_defaults_to_none(self):
        out = DetailedReportOutput(executive_summary="summary")
        assert out.rlm_synthesis is None

    def test_rlm_telemetry_defaults_to_none(self):
        out = DetailedReportOutput(executive_summary="summary")
        assert out.rlm_telemetry is None

    def test_rlm_synthesis_accepts_string(self):
        out = DetailedReportOutput(
            executive_summary="summary",
            rlm_synthesis="RLM synthesized answer",
        )
        assert out.rlm_synthesis == "RLM synthesized answer"

    def test_rlm_telemetry_accepts_dict(self):
        telemetry = {"rlm_enabled": True, "depth_reached": 2, "total_calls": 5}
        out = DetailedReportOutput(
            executive_summary="summary",
            rlm_telemetry=telemetry,
        )
        assert out.rlm_telemetry == telemetry
