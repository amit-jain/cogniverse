"""Round-trip coverage for the XGBoost ProfilePerformanceOptimizer.

The 390-LOC profile recommender (consumed by the dashboard optimization tab) had
zero tests: a feature-array shape change or a save->load breakage would ship
undetected. This trains on real extracted features, predicts, saves, reloads,
and asserts the reloaded model reproduces the prediction.
"""

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from cogniverse_agents.routing.profile_performance_optimizer import (
    ProfilePerformanceOptimizer,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

# Three profiles, 5 queries each. Three classes exercises XGBoost's multiclass
# path (the mlogloss eval_metric the trainer configures) and gives a stratified
# train/test split enough samples per class.
_QUERIES = [
    "show me the diagram on page 3",
    "what does the chart illustrate",
    "find the picture of the bridge",
    "render the figure from the report",
    "the visual layout of the poster",
    "summarize the quarterly financials",
    "explain the methodology section",
    "list the key findings in the text",
    "define the term in the glossary",
    "quote the abstract paragraph",
    "play the podcast about robots",
    "transcribe the keynote audio",
    "the acoustic signature of the whale",
    "listen to the recorded interview",
    "the spoken narration track",
]
_PROFILES = ["visual_profile"] * 5 + ["text_profile"] * 5 + ["audio_profile"] * 5


def _trained_optimizer(model_dir):
    opt = ProfilePerformanceOptimizer(model_dir=model_dir)
    X = np.array(
        [opt.extract_query_features(q).to_array() for q in _QUERIES], dtype=float
    )
    opt.label_encoder = LabelEncoder()
    y = opt.label_encoder.fit_transform(_PROFILES)
    opt.train(X, y)
    return opt


def test_train_predict_reports_a_known_profile(tmp_path):
    opt = _trained_optimizer(tmp_path)
    assert opt.is_trained

    profile, confidence = opt.predict_best_profile("show me the diagram on page 1")
    assert profile in {"visual_profile", "text_profile", "audio_profile"}
    assert 0.0 <= confidence <= 1.0


def test_predict_before_train_raises(tmp_path):
    opt = ProfilePerformanceOptimizer(model_dir=tmp_path)
    with pytest.raises(RuntimeError, match="not trained"):
        opt.predict_best_profile("anything")


def test_save_load_round_trip_reproduces_prediction(tmp_path):
    opt = _trained_optimizer(tmp_path)
    query = "find the picture of the tower"
    before = opt.predict_best_profile(query)

    opt.save()

    reloaded = ProfilePerformanceOptimizer(model_dir=tmp_path)
    assert reloaded.load() is True
    assert reloaded.is_trained

    after = reloaded.predict_best_profile(query)
    # Same profile and (bit-for-bit) same confidence after a save->load.
    assert after[0] == before[0]
    assert after[1] == pytest.approx(before[1])


def test_load_returns_false_when_no_model_saved(tmp_path):
    opt = ProfilePerformanceOptimizer(model_dir=tmp_path)
    assert opt.load() is False


def test_two_profile_training_uses_binary_metric(tmp_path):
    """A tenant with exactly two profiles must train — XGBoost uses a binary
    objective there, so the multiclass mlogloss eval_metric crashed it."""
    opt = ProfilePerformanceOptimizer(model_dir=tmp_path)
    queries = _QUERIES[:5] + _QUERIES[5:10]  # visual + text queries
    profiles = ["visual_profile"] * 5 + ["text_profile"] * 5
    X = np.array(
        [opt.extract_query_features(q).to_array() for q in queries], dtype=float
    )
    opt.label_encoder = LabelEncoder()
    y = opt.label_encoder.fit_transform(profiles)

    metrics = opt.train(X, y)  # must not raise

    assert metrics["n_profiles"] == 2
    profile, _ = opt.predict_best_profile("find the picture of the tower")
    assert profile in {"visual_profile", "text_profile"}


@pytest.mark.asyncio
async def test_extract_bounds_span_pull_and_warns_on_truncation(
    monkeypatch, caplog, tmp_path
):
    """The evaluation-span pull must set an explicit limit (get_spans defaults to
    only 1000, silently capping the training window) and warn when the window
    fills that cap so a truncated training sample is visible, not silent."""
    from types import SimpleNamespace

    import pandas as pd

    from cogniverse_agents.routing import profile_performance_optimizer as ppo

    monkeypatch.setattr(ppo, "SPAN_QUERY_LIMIT", 3)

    calls = {}

    async def _get_spans(**kwargs):
        calls.update(kwargs)
        # A full-cap frame (== limit rows) whose names don't match search|eval,
        # so the client filter empties it and the method raises after warning.
        return pd.DataFrame({"name": ["misc", "misc", "misc"]})

    provider = SimpleNamespace(traces=SimpleNamespace(get_spans=_get_spans))
    mgr = SimpleNamespace(get_provider=lambda tenant_id: provider)
    monkeypatch.setattr(ppo, "get_telemetry_manager", lambda: mgr)

    opt = ProfilePerformanceOptimizer(model_dir=tmp_path)
    with caplog.at_level("WARNING"):
        with pytest.raises(ValueError):
            await opt.extract_training_data_from_phoenix(
                tenant_id="acme:acme",
                project_name="cogniverse-acme:acme",
            )

    assert calls["limit"] == 3, "span pull must pass an explicit bounded limit"
    assert any("cap" in r.getMessage().lower() for r in caplog.records), (
        "a window that fills the cap must surface a truncation warning"
    )
