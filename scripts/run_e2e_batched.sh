#!/usr/bin/env bash
# Run the e2e suite in two batches with a runtime restart between them.
#
# Why batched: the runtime pod (currently 40Gi memory limit on k3s) leaks
# ~0.4 Gi/min under sustained ingestion + orchestration load. A single
# run of the whole suite hits the cap around the 90-minute mark and the
# pod OOMKills mid-run, cascading every subsequent test to 5xx/connection
# errors. Each batch finishes inside the budget; the restart between
# resets per-request state (Mem0 caches, torch CPU allocator pool, DSPy
# module caches) so batch 2 starts from ~2 Gi baseline again.
#
# Prereqs: `cogniverse up` must already be running (k3d cluster healthy,
# runtime + llm + vespa + phoenix + vllm-embed all Ready).
#
# Usage:
#   bash scripts/run_e2e_batched.sh              # both batches
#   bash scripts/run_e2e_batched.sh batch1       # gateway/search/CRUD only
#   bash scripts/run_e2e_batched.sh batch2       # heavy ingestion + rest
#
# Expected wall-clock on a dev k3d: ~50 min batch 1, ~40 min batch 2.

set -euo pipefail

NS=cogniverse
LOG_DIR=${LOG_DIR:-/tmp/cogniverse_e2e_runs}
mkdir -p "$LOG_DIR"

# Load repo-level .env so tests like the Telegram flow pick up
# TELEGRAM_BOT_TOKEN / TELEGRAM_TEST_CHAT_ID without relying on the caller's
# shell having exported them. Without this the tests skipif-out.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

# Batch 1: light/medium tests that don't exercise the heavy ingestion
# pipeline. Gateway classification, orchestration (LLM-bound but no
# ColPali frame-encoding), search, CRUD, registry, multi-turn, synthetic
# data, tenant ops, stats. Memory stays under ~15 Gi for this set.
BATCH1=(
  tests/e2e/test_a2a_gateway_e2e.py
  tests/e2e/test_a2a_multiturn_e2e.py
  tests/e2e/test_api_e2e.py
)

# Batch 2: everything else, including the ingestion tests
# (video/image/audio/pdf/document/batch) that each load ColPali,
# run per-frame encoding, and accumulate torch allocator state.
# Also includes coding, deep research, graph, messaging, multi-profile,
# tenant extensibility, wiki, batch optimization. Needs a fresh pod so
# it starts at baseline memory.
BATCH2=(
  tests/e2e/test_coding_cli_e2e.py
  tests/e2e/test_deep_research_and_annotation_queue_e2e.py
  tests/e2e/test_graph_cli_e2e.py
  tests/e2e/test_messaging_e2e.py
  tests/e2e/test_multiprofile_and_isolation_e2e.py
  tests/e2e/test_tenant_extensibility_e2e.py
  tests/e2e/test_wiki_e2e.py
  tests/e2e/test_batch_optimization_e2e.py
)

# Intentionally excluded (not part of the regular suite):
#   test_quality_monitor_e2e.py  — needs the quality-monitor sidecar running
#   test_dashboard_e2e.py        — needs Playwright + a browser

wait_runtime_ready() {
  echo "waiting for runtime 2/2 Running..."
  until kubectl get pods -n "$NS" -l app.kubernetes.io/component=runtime --no-headers 2>/dev/null \
      | awk '$2=="2/2" && $3=="Running"' | grep -q .; do
    sleep 5
  done
  # First /health/live right after restart can take a few seconds while
  # the uvicorn worker finishes startup; warm the endpoint once.
  for _ in 1 2 3; do
    curl -fsS --max-time 10 http://localhost:28000/health/live >/dev/null 2>&1 && break
    sleep 5
  done
  echo "runtime ready"
}

restart_runtime() {
  echo "restarting runtime pod to reset memory baseline..."
  kubectl delete pod -n "$NS" -l app.kubernetes.io/component=runtime --wait=false >/dev/null
  wait_runtime_ready
}

run_batch() {
  local label=$1
  shift
  local log="$LOG_DIR/$label.log"
  echo "=== $label — $(date) ==="
  echo "log: $log"
  uv run pytest "$@" --tb=short -v 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  local passed failed skipped
  passed=$(grep -c PASSED "$log" || true)
  failed=$(grep -c FAILED "$log" || true)
  skipped=$(grep -c SKIPPED "$log" || true)
  echo "=== $label done rc=$rc passed=$passed failed=$failed skipped=$skipped ==="
  return $rc
}

target=${1:-all}

case "$target" in
  batch1)
    wait_runtime_ready
    run_batch batch1 "${BATCH1[@]}"
    ;;
  batch2)
    restart_runtime
    run_batch batch2 "${BATCH2[@]}"
    ;;
  all)
    wait_runtime_ready
    run_batch batch1 "${BATCH1[@]}" || true
    restart_runtime
    run_batch batch2 "${BATCH2[@]}"
    ;;
  *)
    echo "usage: $0 [batch1|batch2|all]" >&2
    exit 2
    ;;
esac
