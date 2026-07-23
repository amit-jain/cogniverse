"""Experiment registry round-trip through a real Phoenix dataset store.

create_experiment registers a durable dataset record; log_evaluation appends
evaluation rows to it. These pin that the records are readable back from
Phoenix with exact values, and that logging into a never-created experiment
raises instead of silently dropping the evaluation.
"""

import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

TENANT = "acme:exp-roundtrip"


@pytest.fixture
def provider(search_evaluator_provider, phoenix_container):
    """Provider resolved the way the dashboard does it, against the managed
    Phoenix container (search_evaluator_provider boots the telemetry
    manager singleton for this endpoint)."""
    from cogniverse_evaluation.providers import get_evaluation_provider
    from cogniverse_evaluation.providers.registry import get_evaluation_registry

    try:
        yield get_evaluation_provider(
            name="phoenix",
            tenant_id=TENANT,
            config={
                "http_endpoint": phoenix_container["http_endpoint"],
                "grpc_endpoint": phoenix_container["grpc_endpoint"],
            },
        )
    finally:
        get_evaluation_registry().clear_cache()


class TestExperimentRoundTrip:
    def test_create_and_log_round_trip(self, provider, phoenix_container):
        name = f"exp-{uuid.uuid4().hex[:8]}"

        result = provider.create_experiment(
            name, description="round-trip experiment", metadata={"owner": "qa"}
        )

        assert result["id"] == f"experiment-{name}"
        assert result["name"] == name

        provider.log_evaluation(
            experiment_id=result["id"],
            evaluation_name="relevance",
            score=0.85,
            label="good",
            explanation="matched golden set",
        )

        from phoenix.client import Client

        raw = Client(base_url=phoenix_container["http_endpoint"]).datasets.get_dataset(
            dataset=result["id"]
        )
        examples = list(raw.examples)
        assert len(examples) == 2

        by_event = {ex["input"]["event"]: ex for ex in examples}
        assert set(by_event) == {"experiment_created", "evaluation"}
        created = by_event["experiment_created"]
        assert created["input"]["experiment"] == name
        assert created["output"]["description"] == "round-trip experiment"
        evaluation = by_event["evaluation"]
        assert evaluation["input"]["evaluation_name"] == "relevance"
        assert float(evaluation["output"]["score"]) == 0.85
        assert evaluation["output"]["label"] == "good"

    def test_log_evaluation_unknown_experiment_raises(self, provider):
        with pytest.raises(ValueError):
            provider.log_evaluation(
                experiment_id=f"experiment-ghost-{uuid.uuid4().hex[:8]}",
                evaluation_name="relevance",
                score=0.5,
            )

    def test_experiment_url_points_at_provider_endpoint(
        self, provider, phoenix_container
    ):
        url = provider.get_experiment_url("experiment-x")

        assert url == f"{phoenix_container['http_endpoint']}/projects/experiment-x"
