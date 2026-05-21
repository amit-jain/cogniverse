#!/usr/bin/env bash
# Probe every endpoint the live integration tests depend on.
#
# Every endpoint is exposed by the k3d serverlb (NodePort), so no
# kubectl port-forwards are required. If you see DOWN below, check
# that the chart was deployed with the vllm-llm-student / -teacher
# services set to NodePort (default since the per-segment KG work).
#
# Endpoints (NodePort on the host):
#   localhost:8080  -> Vespa app                  (charts default)
#   localhost:19071 -> Vespa config               (charts default)
#   localhost:26006 -> Phoenix HTTP               (NodePort 26006)
#   localhost:4317  -> Phoenix OTLP gRPC          (NodePort 4317)
#   localhost:29002 -> ColBERT pylate sidecar     (NodePort 29002)
#   localhost:29010 -> vllm-llm-student (gemma)   (NodePort 29010)
#   localhost:29011 -> vllm-llm-teacher           (NodePort 29011)
#
# Usage:
#     ./scripts/setup_local_tests.sh         # probe every endpoint
#     ./scripts/setup_local_tests.sh check   # alias
#
# Once everything reports ok, drive the tests:
#     uv run pytest tests/agents/integration/test_claim_extractor_dspy.py
#     RECORD_GOLDEN=1 uv run pytest tests/agents/integration/test_bright_video_probes.py
#     ./scripts/seed_bright_corpus.py   # one-shot BRIGHT corpus seed

set -euo pipefail

check_endpoints() {
    echo "--- Endpoint health ---"
    _probe "Vespa app    (8080)"  "http://localhost:8080/state/v1/health"
    _probe "Vespa config (19071)" "http://localhost:19071/ApplicationStatus"
    _probe "Phoenix HTTP (26006)" "http://localhost:26006"
    _probe "ColBERT pyl. (29002)" "http://localhost:29002/health"
    _probe "LM student   (29010)" "http://localhost:29010/v1/models"
    _probe "LM teacher   (29011)" "http://localhost:29011/v1/models"
}

_probe() {
    local label="$1" url="$2"
    if curl -sf -m 3 "$url" >/dev/null 2>&1; then
        echo "  ok   $label"
    else
        echo "  DOWN $label  ($url)"
    fi
}

case "${1:-check}" in
    check|start) check_endpoints ;;
    *)
        echo "usage: $0 [check]" >&2
        exit 2
        ;;
esac
