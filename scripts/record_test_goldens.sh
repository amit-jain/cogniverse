#!/usr/bin/env bash
# Re-record every integration-test golden against the live k3d cluster.
#
# Goldens are not checked into git — they're regenerated on demand by
# running pytest with RECORD_GOLDEN=1 against the live LM + Vespa +
# ColBERT + Phoenix stack. This script is the one-command wrapper.
#
# Prereqs (run once before the goldens can be recorded):
#   ./scripts/setup_local_tests.sh        # kubectl port-forwards
#   ./scripts/seed_bright_corpus.py       # BRIGHT corpus into Vespa
#
# Usage:
#   ./scripts/record_test_goldens.sh                # record everything
#   ./scripts/record_test_goldens.sh joint_trace    # one block only
#   ./scripts/record_test_goldens.sh per_segment claim cross_modal
#
# After recording, run pytest without RECORD_GOLDEN to verify byte-equal
# replay:
#   uv run pytest tests/agents/integration/ tests/core/integration/

set -euo pipefail

unset COGNIVERSE_CONFIG TEST_LLM_API_BASE TEST_LLM_MODEL OPENAI_API_KEY

export COLBERT_PYLATE_URL="${COLBERT_PYLATE_URL:-http://localhost:29002}"
export PHOENIX_HTTP_ENDPOINT="${PHOENIX_HTTP_ENDPOINT:-http://localhost:26006}"
export PHOENIX_GRPC_ENDPOINT="${PHOENIX_GRPC_ENDPOINT:-http://localhost:4317}"
export RECORD_GOLDEN=1

declare -A TARGETS=(
    [per_segment]="tests/agents/integration/test_per_segment_kg_provenance.py"
    [claim]="tests/agents/integration/test_claim_extractor_dspy.py"
    [cross_modal]="tests/agents/integration/test_cross_modal_linking.py"
    [content_backrefs]="tests/agents/integration/test_content_schema_backrefs.py"
    [iterative_loop]="tests/agents/integration/test_iterative_retrieval_loop.py"
    [kg_consumers]="tests/agents/integration/test_kg_consumer_agents_segment_provenance.py"
    [joint_trace]="tests/core/integration/test_joint_trace_encoding.py"
    [bright]="tests/agents/integration/test_bright_video_probes.py"
)

if [ $# -eq 0 ]; then
    # Record everything. Order matters slightly — joint_trace + per_segment
    # are quick, bright is the slowest (~15 min for the 30-query sweep).
    BLOCKS=(joint_trace per_segment claim cross_modal content_backrefs
            iterative_loop kg_consumers bright)
else
    BLOCKS=("$@")
fi

for block in "${BLOCKS[@]}"; do
    path="${TARGETS[$block]:-}"
    if [ -z "$path" ]; then
        echo "Unknown block: $block. Known: ${!TARGETS[*]}" >&2
        exit 2
    fi
    echo "================================================================"
    echo "Recording goldens: $block ($path)"
    echo "================================================================"
    uv run pytest "$path" -v --tb=short --log-cli-level=ERROR
done

echo
echo "All requested blocks recorded. Verify byte-equal replay with:"
echo "  uv run pytest ${BLOCKS[@]/#/tests/}"
