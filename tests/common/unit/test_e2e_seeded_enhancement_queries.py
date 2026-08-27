"""The simba e2e contract asserts every query the module seeded appears as a
served span. Under replay the module seeds the sampled recording, not the
live ENHANCEMENT_QUERIES loop, so the seeded set must derive from the sample."""

from tests.e2e import span_capture
from tests.e2e import test_batch_optimization_e2e as e2e


def _sampled_qe_inputs() -> set[str]:
    from cogniverse_foundation.telemetry.config import SPAN_NAME_QUERY_ENHANCEMENT

    records = span_capture.load_capture_json(e2e.OPTIMIZER_SPAN_CAPTURE_PATH)
    sampled = span_capture.sample_capture_by_name(
        records, e2e._optimizer_capture_sample_caps()
    )
    return {
        str(record["attributes"].get("input.value") or "").strip()
        for record in sampled
        if record["name"] == SPAN_NAME_QUERY_ENHANCEMENT
    }


def test_replay_seeded_queries_are_exactly_the_sampled_recording(monkeypatch):
    monkeypatch.delenv(e2e.OPTIMIZER_SPAN_CAPTURE_MODE_ENV, raising=False)
    seeded = e2e._seeded_enhancement_queries()
    assert seeded == _sampled_qe_inputs()
    assert len(seeded) == 91
    assert "a fire starter" not in seeded


def test_record_seeded_queries_follow_the_live_seeding_loop(monkeypatch):
    monkeypatch.setenv(e2e.OPTIMIZER_SPAN_CAPTURE_MODE_ENV, "record")
    monkeypatch.setenv("BATCH_SPAN_COUNT", "3")
    seeded = e2e._seeded_enhancement_queries()
    assert seeded == set(e2e.ENHANCEMENT_QUERIES[:3]) | {
        q for q, _, _ in e2e.GROUNDED_ENHANCEMENT_QUERIES
    }
