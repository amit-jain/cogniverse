"""The naming contract of a tenant's optimization artifact datasets.

``ArtifactManager`` names every dataset it writes with
``artifact_dataset_name``; ``is_artifact_dataset`` reads that name back for
the one tenant, so listings of evaluation datasets can leave artifacts out.
"""

from __future__ import annotations

import pytest

from cogniverse_foundation.telemetry.providers.base import (
    artifact_dataset_name,
    is_artifact_dataset,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TENANT = "acme:prod"


def test_each_artifact_name_is_built_from_kind_tenant_and_key():
    assert [
        artifact_dataset_name("config", TENANT, "golden_set_ground_truth--r1"),
        artifact_dataset_name("prompts", TENANT, "query_enhancement-v3"),
        artifact_dataset_name("model", TENANT, "profile_selection--r0"),
    ] == [
        "dspy-config-acme:prod-golden_set_ground_truth--r1",
        "dspy-prompts-acme:prod-query_enhancement-v3",
        "dspy-model-acme:prod-profile_selection--r0",
    ]


@pytest.mark.parametrize("kind", ["", "golden-set"])
def test_a_kind_that_is_not_one_word_is_refused(kind):
    with pytest.raises(ValueError) as refused:
        artifact_dataset_name(kind, TENANT, "key")
    assert str(refused.value) == (
        f"artifact kind must be one word without '-', got {kind!r}"
    )


@pytest.mark.parametrize(
    "name, artifact",
    [
        (artifact_dataset_name("config", TENANT, "gateway_thresholds--r2"), True),
        (artifact_dataset_name("experiments", TENANT, "routing"), True),
        (artifact_dataset_name("workflow", TENANT, "templates"), True),
        # Another tenant's artifact, including one whose id extends this one's.
        (artifact_dataset_name("config", "acme:production", "x--r1"), False),
        (artifact_dataset_name("config", "acme:prod2", "x--r1"), False),
        # Evaluation datasets, one of them named like the prefix.
        ("golden-acme-prod", False),
        ("dspy-eval-set", False),
        (f"dspy-{TENANT}-x", False),
        ("dspy-config", False),
    ],
)
def test_only_the_tenants_own_artifact_names_are_artifacts(name, artifact):
    assert is_artifact_dataset(name, TENANT) is artifact
