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

# Load repo-level secrets so tests like the Telegram flow pick up
# TELEGRAM_BOT_TOKEN / TELEGRAM_TEST_CHAT_ID without relying on the caller's
# shell having exported them. Without this the tests deselect out.
# `.env` is either a single file of KEY=VALUE lines or a directory of
# per-key `KEY.env` files holding a bare value; cogniverse_cli.secrets
# resolves both forms and this must match it.
# >>> e2e-env-loader
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
elif [[ -d "$REPO_ROOT/.env" ]]; then
  for _secret_file in "$REPO_ROOT"/.env/*.env; do
    [[ -e "$_secret_file" ]] || continue
    _secret_name="$(basename "$_secret_file" .env)"
    _secret_value="$(grep -vE '^[[:space:]]*(#|$)' "$_secret_file" | head -1 || true)"
    [[ -n "$_secret_value" ]] || continue
    export "$_secret_name=${_secret_value#"$_secret_name"=}"
  done
  unset _secret_file _secret_name _secret_value
fi
# <<< e2e-env-loader

# Only one e2e run may touch the cluster at a time. Concurrent runs multiply
# concurrent LM/ingestion load on a serving stack whose memory scales with it;
# on this unified-memory host the GPU pool is pinned and unswappable, so the
# kernel reclaims from everything else until it OOMs the desktop.
# >>> e2e-run-lock
E2E_LOCK_FILE="${E2E_LOCK_FILE:-/tmp/cogniverse_e2e_run.lock}"
E2E_LOCK_SCAN_PATTERN="${E2E_LOCK_SCAN_PATTERN:-pytest.*tests/e2e}"

_e2e_lock_live_holder() {
  local pid
  pid="$(head -1 "$E2E_LOCK_FILE" 2>/dev/null | tr -dc '0-9' || true)"
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" 2>/dev/null && printf '%s' "$pid"
  return 0
}

# Skips our own process group: a scan by command line otherwise matches the
# very shell running the scan, because that shell's argv contains the pattern.
# A second run launched from this same group is caught by the lock file above.
_e2e_foreign_run() {
  local mypgid
  mypgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ')"
  ps -eo pid=,pgid=,args= 2>/dev/null | awk -v skip="$mypgid" -v pat="$E2E_LOCK_SCAN_PATTERN" '
    $2 == skip { next }
    $0 ~ pat { print $1; exit }
  '
  return 0
}

_e2e_release_lock() {
  local owner
  owner="$(head -1 "$E2E_LOCK_FILE" 2>/dev/null | tr -dc '0-9' || true)"
  [[ "$owner" == "$$" ]] && rm -f "$E2E_LOCK_FILE"
  return 0
}

_e2e_acquire_lock() {
  local holder foreign
  holder="$(_e2e_lock_live_holder)"
  if [[ -n "$holder" ]]; then
    echo "REFUSING: an e2e run already holds the lock (pid $holder)." >&2
    echo "  lock: $E2E_LOCK_FILE" >&2
    echo "  cmd : $(ps -o args= -p "$holder" 2>/dev/null | head -1)" >&2
    echo "  Concurrent e2e runs have OOMed this host. Wait for it, or kill it." >&2
    exit 3
  fi
  foreign="$(_e2e_foreign_run)"
  if [[ -n "$foreign" ]]; then
    echo "REFUSING: another e2e run is already in flight (pid $foreign)." >&2
    echo "  cmd : $(ps -o args= -p "$foreign" 2>/dev/null | head -1)" >&2
    echo "  Concurrent e2e runs have OOMed this host. Wait for it, or kill it." >&2
    exit 3
  fi
  [[ -e "$E2E_LOCK_FILE" ]] && echo "Taking over a stale e2e lock at $E2E_LOCK_FILE." >&2
  printf '%s\n' "$$" > "$E2E_LOCK_FILE"
  trap _e2e_release_lock EXIT
  return 0
}

_e2e_acquire_lock
# <<< e2e-run-lock

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
  # Knowledge-System Improvements e2e suite (Phases 0-11). Each per-test
  # tenant deploys its own Vespa schema, and the RLM-bound tests run
  # under the in-cluster vLLM — keeping these in batch 2 (after the
  # runtime restart) avoids contention with batch 1's lighter
  # gateway/CRUD set.
  tests/e2e/test_conftest_helpers_e2e.py
  tests/e2e/test_knowledge_schema_e2e.py
  tests/e2e/test_provenance_e2e.py
  tests/e2e/test_contradiction_detection_e2e.py
  tests/e2e/test_trust_ranking_e2e.py
  tests/e2e/test_federation_e2e.py
  tests/e2e/test_pinning_quotas_e2e.py
  tests/e2e/test_rlm_telemetry_e2e.py
  tests/e2e/test_rlm_ab_harness_e2e.py
  tests/e2e/test_signature_variants_e2e.py
  tests/e2e/test_canary_state_machine_e2e.py
  tests/e2e/test_artifact_rollback_cli_e2e.py
  tests/e2e/test_sandbox_policy_boot_e2e.py
  tests/e2e/test_gateway_health_probe_e2e.py
  tests/e2e/test_citation_and_audit_agents_e2e.py
  tests/e2e/test_kg_traversal_agent_e2e.py
  tests/e2e/test_knowledge_summarization_agent_e2e.py
  tests/e2e/test_cross_tenant_comparison_agent_e2e.py
  tests/e2e/test_federated_query_agent_e2e.py
  tests/e2e/test_contradiction_reconciliation_agent_e2e.py
  tests/e2e/test_multi_document_synthesis_agent_e2e.py
  tests/e2e/test_temporal_reasoning_agent_e2e.py
  tests/e2e/test_deep_synthesis_workflow_e2e.py
)

# Explicit exclusions for files that stay outside the batch sweep because a
# different selection mechanism owns them.
# Format: "tests/e2e/test_*.py|one-line reason"
# >>> e2e-batch-exclusions
E2E_BATCH_EXCLUSIONS=(
  "tests/e2e/test_dashboard_e2e.py|browser lane via pytest.mark.browser"
  "tests/e2e/test_modal_inference_e2e.py|requires_modal_inference collection gate"
  "tests/e2e/test_cronworkflow_execution_heavy_e2e.py|e2e_heavy marker"
)
# <<< e2e-batch-exclusions

# Known uncovered files; this list is a ratchet and may only shrink.
# Format: "tests/e2e/test_*.py|one-line surface name"
# >>> e2e-batch-uncovered
E2E_BATCH_UNCOVERED=(
  "tests/e2e/test_annotation_feedback_e2e.py|annotation feedback workflow"
  "tests/e2e/test_asr_sidecar_e2e.py|ASR sidecar"
  "tests/e2e/test_cron_guard.py|CronWorkflow guard contract"
  "tests/e2e/test_cronworkflow_execution_e2e.py|light CronWorkflow execution"
  "tests/e2e/test_inbound_dspy_span_e2e.py|inbound DSPy span contract"
  "tests/e2e/test_inbound_lm_output_approximations.py|inbound LM-output approximations"
  "tests/e2e/test_inbound_redis_replay_e2e.py|inbound Redis replay durability"
  "tests/e2e/test_ingestion_pipeline_telemetry.py|ingestion-pipeline telemetry"
  "tests/e2e/test_ingestion_upload_e2e.py|ingestion upload"
  "tests/e2e/test_inpod_telemetry_prelude_guard.py|in-pod telemetry prelude guard"
  "tests/e2e/test_kubectl_context_contract_e2e.py|kubectl context contract"
  "tests/e2e/test_manual_optimization_e2e.py|manual optimization workflow"
  "tests/e2e/test_messaging_gateway_e2e.py|messaging gateway"
  "tests/e2e/test_multimodal_report_e2e.py|multimodal report"
  "tests/e2e/test_optimizer_persistence_e2e.py|optimizer persistence"
  "tests/e2e/test_orchestrator_inbound_e2e.py|orchestrator inbound"
  "tests/e2e/test_quality_monitor_e2e.py|quality-monitor sidecar"
  "tests/e2e/test_run_lock.py|run-lock and GPU residency contract"
)
# <<< e2e-batch-uncovered

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
  # --tb=long + -rA: the end-of-run FAILURES section carries every
  # full traceback so post-mortem analysis can pinpoint the actual
  # assertion that failed. Prior sweeps used --tb=short, so killing
  # them mid-run left only "FAILED" markers with zero diagnostic.
  uv run pytest "$@" --tb=long -rA -v 2>&1 | tee "$log"
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
