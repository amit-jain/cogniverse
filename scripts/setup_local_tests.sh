#!/usr/bin/env bash
# Set up port-forwards + env so the live integration tests can run.
#
# The H / B / A / C / D / E / G integration suites all hit the live k3d
# cogniverse cluster. They need three local listeners:
#
#   localhost:8101 -> svc/cogniverse-vllm-llm-student:8000  (gemma-4-e4b-it)
#   localhost:8102 -> svc/cogniverse-vllm-llm-teacher:8000  (teacher model)
#   localhost:26006 -> svc/cogniverse-phoenix:6006          (Phoenix UI / OTel HTTP)
#   localhost:4317  -> svc/cogniverse-phoenix:4317          (Phoenix OTel gRPC)
#
# k3d's serverlb already forwards Vespa (:8080, :19071) and colbert_pylate
# (:29002) by default, so those don't need explicit kubectl port-forwards.
#
# Usage:
#     ./scripts/setup_local_tests.sh         # start forwards in the background
#     ./scripts/setup_local_tests.sh stop    # kill all forwards
#     ./scripts/setup_local_tests.sh check   # probe each endpoint, report status
#
# Once running you can drive the tests:
#     uv run pytest tests/agents/integration/test_claim_extractor_dspy.py
#     RECORD_GOLDEN=1 uv run pytest tests/agents/integration/test_bright_video_probes.py
#     ./scripts/seed_bright_corpus.py   # one-shot BRIGHT corpus seed

set -euo pipefail

NAMESPACE="${COGNIVERSE_NS:-cogniverse}"
LOG_DIR="${TMPDIR:-/tmp}/cogniverse-portforwards"
mkdir -p "$LOG_DIR"

declare -A FORWARDS=(
    [llm-student]="svc/cogniverse-vllm-llm-student 8101:8000"
    [llm-teacher]="svc/cogniverse-vllm-llm-teacher 8102:8000"
    [phoenix-http]="svc/cogniverse-phoenix 26006:6006"
    [phoenix-grpc]="svc/cogniverse-phoenix 4317:4317"
)

start_forwards() {
    for name in "${!FORWARDS[@]}"; do
        spec="${FORWARDS[$name]}"
        target=$(echo "$spec" | awk '{print $1}')
        ports=$(echo "$spec" | awk '{print $2}')
        local_port=$(echo "$ports" | cut -d: -f1)

        if ss -tln 2>/dev/null | grep -q ":${local_port} "; then
            echo "[skip] $name — localhost:$local_port already in use"
            continue
        fi

        log="$LOG_DIR/$name.log"
        nohup kubectl port-forward -n "$NAMESPACE" $target $ports >"$log" 2>&1 &
        disown
        echo "[up]   $name -> $target $ports  (log: $log)"
    done
    sleep 3
    check_forwards
}

stop_forwards() {
    pkill -f 'kubectl port-forward -n '"$NAMESPACE"'.*(vllm-llm|phoenix)' || true
    echo "[stop] killed all matching kubectl port-forwards"
}

check_forwards() {
    echo
    echo "--- Endpoint health ---"
    _probe "LM student   (8101)"  "http://localhost:8101/v1/models"
    _probe "LM teacher   (8102)"  "http://localhost:8102/v1/models"
    _probe "Phoenix HTTP (26006)" "http://localhost:26006"
    _probe "Vespa app    (8080)"  "http://localhost:8080/state/v1/health"
    _probe "Vespa config (19071)" "http://localhost:19071/ApplicationStatus"
    _probe "ColBERT pyl. (29002)" "http://localhost:29002/health"
}

_probe() {
    local label="$1" url="$2"
    if curl -sf -m 3 "$url" >/dev/null 2>&1; then
        echo "  ok   $label"
    else
        echo "  DOWN $label  ($url)"
    fi
}

case "${1:-start}" in
    start) start_forwards ;;
    stop)  stop_forwards ;;
    check) check_forwards ;;
    *)
        echo "usage: $0 [start|stop|check]" >&2
        exit 2
        ;;
esac
