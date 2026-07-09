"""AdapterMetadata timestamps must be timezone-aware.

created_at/updated_at defaulted to naive datetime.utcnow while from_dict parsed
stored ISO values that carry a +00:00 offset (tz-aware). Sorting or comparing a
freshly-created record against a loaded one then raised
"can't compare offset-naive and offset-aware datetimes".
"""

from __future__ import annotations

from datetime import timezone

from cogniverse_finetuning.registry.models import AdapterMetadata


def test_default_created_at_is_tz_aware():
    m = AdapterMetadata(
        adapter_id="a1",
        tenant_id="acme",
        name="n",
        version="1",
        base_model="b",
        model_type="llm",
        agent_type=None,
        training_method="sft",
        adapter_path="/tmp/a",
    )
    assert m.created_at.tzinfo is not None
    assert m.created_at.utcoffset() == timezone.utc.utcoffset(m.created_at)


def test_fresh_and_loaded_records_are_comparable():
    fresh = AdapterMetadata(
        adapter_id="a1",
        tenant_id="acme",
        name="n",
        version="1",
        base_model="b",
        model_type="llm",
        agent_type=None,
        training_method="sft",
        adapter_path="/tmp/a",
    )
    loaded = AdapterMetadata.from_vespa_doc(
        {
            "adapter_id": "a2",
            "tenant_id": "acme",
            "name": "n2",
            "version": "2",
            "base_model": "b",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    )
    # Would have raised TypeError before the fix.
    ordered = sorted([fresh, loaded], key=lambda m: m.created_at)
    assert ordered[0].adapter_id == "a2"
