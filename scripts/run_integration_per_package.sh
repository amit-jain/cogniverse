#!/usr/bin/env bash
# Run integration tests one package at a time, each in its own pytest
# (so the Python process exits between packages and the OS reclaims
# every byte — shared_vespa stays bounded, ColPali/pylate/Mem0 model
# weights don't accumulate across packages, K3s gateway and Ollama
# clients don't leak singletons across the session).
#
# This is the matched-scope equivalent of CI's per-workflow split, run
# locally. Use this instead of `pytest tests/` for the full integration
# sweep — the latter loads every package's state into one process and
# peak memory grows linearly with packages traversed.

set -u

# Order bottom-up by the UV-workspace dependency tiers documented in
# docs/architecture/sdk-architecture.md
# (Foundation → Core → Implementation → Application). A break in a
# lower tier is the likely root cause for any cascade above it, so
# running lowest-first makes the summary read as a bisection: the
# earliest failed package is where to start fixing.
#
#   common      — Foundation: config persistence, shared utilities
#   backends    — Core/Implementation: Vespa adapter (used by all data layers)
#   evaluation  — Core: evaluation framework
#   memory      — Core+backends: Mem0 / knowledge layer
#   routing     — Implementation: DSPy router
#   agents      — Implementation: agent implementations
#   ingestion   — Implementation/Application: ingestion pipeline (vllm-heavy)
#   finetuning  — Application: training
#   runtime     — Application: FastAPI server
#   admin       — Application: admin API on top of runtime
#   system      — Top: full-system e2e (orchestrates everything above)
DEFAULT_PACKAGES=(
    common
    backends
    evaluation
    memory
    routing
    agents
    ingestion
    finetuning
    runtime
    admin
    system
)

usage() {
    cat <<EOF
Usage: $0 [--log-dir DIR] [--pytest-args "ARGS"] [PACKAGE...]

  --log-dir DIR      Directory for per-package logs (default: /tmp/cogniverse-it)
  --pytest-args STR  Extra args passed to every pytest invocation
  PACKAGE...         Restrict to these packages (default: all 11)

Examples:
  $0                          # run all 11 sequentially
  $0 memory agents            # only memory + agents
  $0 --pytest-args "-x"       # stop within a package at first failure
EOF
    exit 1
}

LOG_DIR="/tmp/cogniverse-it"
EXTRA_ARGS=""
PACKAGES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-dir)   LOG_DIR="$2"; shift 2 ;;
        --pytest-args) EXTRA_ARGS="$2"; shift 2 ;;
        -h|--help)   usage ;;
        --*)         echo "unknown flag: $1" >&2; usage ;;
        *)           PACKAGES+=("$1"); shift ;;
    esac
done

if [[ ${#PACKAGES[@]} -eq 0 ]]; then
    PACKAGES=("${DEFAULT_PACKAGES[@]}")
fi

mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"
: > "$SUMMARY"

START_TS=$(date +%s)
FAILED_PACKAGES=()

for pkg in "${PACKAGES[@]}"; do
    target="tests/$pkg"
    if [[ ! -d "$target" ]]; then
        echo "[$pkg] SKIP — $target not found" | tee -a "$SUMMARY"
        continue
    fi

    log="$LOG_DIR/${pkg}.log"
    echo ""
    echo "=================================================================="
    echo "[$pkg] starting (log: $log)"
    echo "=================================================================="

    pkg_start=$(date +%s)
    # shellcheck disable=SC2086
    uv run pytest "$target" --tb=long -v $EXTRA_ARGS > "$log" 2>&1
    rc=$?
    pkg_elapsed=$(( $(date +%s) - pkg_start ))

    passed=$(grep -cE " PASSED " "$log" || true)
    failed=$(grep -cE " FAILED " "$log" || true)
    errored=$(grep -cE " ERROR " "$log" || true)
    last=$(tail -1 "$log")

    line="[$pkg] rc=$rc  passed=$passed  failed=$failed  errored=$errored  elapsed=${pkg_elapsed}s  ($last)"
    echo "$line" | tee -a "$SUMMARY"

    if [[ $rc -ne 0 ]]; then
        FAILED_PACKAGES+=("$pkg")
    fi
done

TOTAL_ELAPSED=$(( $(date +%s) - START_TS ))
echo ""
echo "=================================================================="
echo "SUMMARY  (total ${TOTAL_ELAPSED}s)"
echo "=================================================================="
cat "$SUMMARY"

if [[ ${#FAILED_PACKAGES[@]} -gt 0 ]]; then
    echo ""
    echo "Failed packages: ${FAILED_PACKAGES[*]}"
    echo "Logs under: $LOG_DIR"
    exit 1
fi

echo ""
echo "All ${#PACKAGES[@]} packages passed."
