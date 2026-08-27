"""In-pod scripts are f-string templates rendered on the host; a brace the
template forgets to double surfaces as a render-time ValueError only once
the e2e reaches that helper. Render and parse them here instead."""

import ast

from tests.e2e import test_batch_optimization_e2e as e2e

_ROW = {
    "example_id": "truth:0",
    "decision": "promote",
    "scored": True,
    "score": 0.9,
    "base_score": 0.8,
    "candidate_score": 0.9,
    "created_at": "2026-08-01T00:00:00+00:00",
    "content": "{}",
}


def test_backdated_training_selection_script_renders_and_parses():
    script = e2e._backdated_training_selection_script(
        "flywheel_org:production", "entity_extraction", [_ROW]
    )
    ast.parse(script)
    assert '[{"content": row["content"], "ledger": json.dumps(ledger)}]' in script
    assert "am = ArtifactManager(tp, 'flywheel_org:production')" in script
    assert (
        "name=am._versioned_dataset_name(\"model\", 'entity_extraction', version)"
        in script
    )
