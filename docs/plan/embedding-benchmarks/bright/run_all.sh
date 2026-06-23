#!/usr/bin/env bash
# Full BRIGHT benchmark orchestration: TEI vs vLLM (Phase 1) + vLLM model sweep (Phase 2).
#
# Prerequisites:
#   export HF_TOKEN=<your_token>
#   pip install faiss-cpu datasets numpy
#
# Runs everything in BRIGHT_DIR (default: directory of this script).
# Edit CATEGORIES to add/remove subsets. Default is a manageable 3-category run.
#
# Usage:
#   ./run_all.sh
#   CATEGORIES="biology economics stackoverflow leetcode" ./run_all.sh

set -euo pipefail
cd "$(dirname "$0")"

CATEGORIES="${CATEGORIES:-biology economics stackoverflow}"
DATA_DIR="bright_data"
EMB_DIR="embeddings"
RES_DIR="results"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

TEI_IMAGE="ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.2"
VLLM_IMAGE="vllm/vllm-openai-cpu:latest-x86_64"
TEI_PORT=8081
VLLM_PORT=8082

# ── helpers ────────────────────────────────────────────────────────────────────

wait_healthy() {
    local url="$1" timeout="${2:-180}" name="${3:-service}"
    echo "  Waiting for $name at $url ..."
    local i=0
    until curl -sf "$url" >/dev/null 2>&1; do
        sleep 5; i=$((i+5))
        if [ $i -ge $timeout ]; then echo "ERROR: $name did not become healthy after ${timeout}s" >&2; exit 1; fi
    done
    echo "  $name is healthy (${i}s)"
}

stop_container() { docker rm -f "$1" 2>/dev/null || true; }

start_tei() {
    local dtype="${1:-float32}"
    stop_container bench-tei
    docker run -d --name bench-tei \
        --platform linux/amd64 \
        -p ${TEI_PORT}:80 \
        -v "${HF_CACHE}:/data" \
        -e HF_HOME=/data \
        -e HF_HUB_OFFLINE=1 \
        -e HUGGINGFACE_HUB_CACHE=/data/hub \
        "${TEI_IMAGE}" \
        --model-id google/embeddinggemma-300m \
        --dtype "${dtype}" \
        --max-batch-tokens 80000 \
        --max-client-batch-size 64 \
        >/dev/null
    wait_healthy "http://localhost:${TEI_PORT}/health" 300 "TEI fp32"
}

start_vllm() {
    local model="$1" dtype="$2" name="${3:-bench-vllm}"
    stop_container "${name}"
    docker run -d --name "${name}" \
        -p ${VLLM_PORT}:8000 \
        --shm-size=2g \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        "${VLLM_IMAGE}" \
        "${model}" --convert embed --dtype "${dtype}" \
        --gpu-memory-utilization 0.2 \
        --disable-log-stats \
        >/dev/null
    wait_healthy "http://localhost:${VLLM_PORT}/health" 300 "vLLM ${model} ${dtype}"
}

encode_all() {
    local out_prefix="$1" url="$2" proto="$3" model="$4"
    local bs="${5:-32}" colbert_flag="${6:-}"
    for cat in $CATEGORIES; do
        echo "  Encoding corpus  [$cat] ..."
        python3 encode.py \
            --url "${url}" --proto "${proto}" --model "${model}" \
            --input "${DATA_DIR}/${cat}/corpus.jsonl" \
            --output-prefix "${out_prefix}/${cat}/corpus" \
            --batch-size "${bs}" ${colbert_flag}
        echo "  Encoding queries [$cat] ..."
        python3 encode.py \
            --url "${url}" --proto "${proto}" --model "${model}" \
            --input "${DATA_DIR}/${cat}/queries.jsonl" \
            --output-prefix "${out_prefix}/${cat}/queries" \
            --batch-size 8 ${colbert_flag}
    done
}

bench_latency() {
    local out_dir="$1" url="$2" proto="$3" model="$4"
    mkdir -p "${out_dir}"
    python3 bench_latency.py \
        --url "${url}" --proto "${proto}" --model "${model}" \
        --batch-sizes 1,8,32 --n-single 10 --n-batch 5 \
        --output "${out_dir}/latency.json"
}

bench_quality() {
    local label="$1" colbert_flag="${2:-}"
    local cats_csv="${CATEGORIES// /,}"
    python3 bench_quality.py \
        --data-dir "${DATA_DIR}" \
        --corpus-prefix "${EMB_DIR}/${label}/{cat}/corpus" \
        --query-prefix  "${EMB_DIR}/${label}/{cat}/queries" \
        --categories "${cats_csv}" \
        --output "${RES_DIR}/${label}/quality.json" \
        ${colbert_flag}
}

# ── Step 0: generate subsets ───────────────────────────────────────────────────

echo "══════════════════════════════════════════════════════════════"
echo "Step 0: generate BRIGHT subsets (fraction=0.2)"
echo "══════════════════════════════════════════════════════════════"
if [ -d "${DATA_DIR}" ]; then
    echo "  ${DATA_DIR}/ exists — skipping (delete to regenerate)"
else
    cats_arg="${CATEGORIES// /,}"
    python3 gen_bright_subset.py --output "${DATA_DIR}" --categories "${cats_arg}"
fi

# ── Phase 1: TEI fp32 ──────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 1a: TEI-CPU fp32 — google/embeddinggemma-300m"
echo "══════════════════════════════════════════════════════════════"
LABEL="tei-gemma"
mkdir -p "${RES_DIR}/${LABEL}"
start_tei float32
encode_all "${EMB_DIR}/${LABEL}" "http://localhost:${TEI_PORT}/embed" tei "" 64
bench_latency "${RES_DIR}/${LABEL}" "http://localhost:${TEI_PORT}/embed" tei ""
bench_quality "${LABEL}"
stop_container bench-tei

# ── Phase 1: vLLM fp32 ─────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 1b: vLLM-CPU fp32 — google/embeddinggemma-300m"
echo "══════════════════════════════════════════════════════════════"
LABEL="vllm-gemma-fp32"
mkdir -p "${RES_DIR}/${LABEL}"
start_vllm "google/embeddinggemma-300m" float32
encode_all "${EMB_DIR}/${LABEL}" "http://localhost:${VLLM_PORT}/v1/embeddings" openai "google/embeddinggemma-300m" 32
bench_latency "${RES_DIR}/${LABEL}" "http://localhost:${VLLM_PORT}/v1/embeddings" openai "google/embeddinggemma-300m"
bench_quality "${LABEL}"
stop_container bench-vllm

# ── Phase 1: vLLM bf16 (latency only) ─────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 1c: vLLM-CPU bf16 — latency only (reuse fp32 quality)"
echo "══════════════════════════════════════════════════════════════"
LABEL="vllm-gemma-bf16"
mkdir -p "${RES_DIR}/${LABEL}"
start_vllm "google/embeddinggemma-300m" bfloat16
bench_latency "${RES_DIR}/${LABEL}" "http://localhost:${VLLM_PORT}/v1/embeddings" openai "google/embeddinggemma-300m"
stop_container bench-vllm

# pick best dtype by comparing p50 single latency
BEST_DTYPE=$(python3 - <<'EOF'
import json
fp32 = json.load(open("results/vllm-gemma-fp32/latency.json"))["modes"]["single"]["p50"]
bf16 = json.load(open("results/vllm-gemma-bf16/latency.json"))["modes"]["single"]["p50"]
print("float32" if fp32 <= bf16 else "bfloat16")
EOF
)
echo ""
echo "Best dtype for this machine: ${BEST_DTYPE}  (fp32=$(python3 -c "import json; print(json.load(open('results/vllm-gemma-fp32/latency.json'))['modes']['single']['p50'])")s  bf16=$(python3 -c "import json; print(json.load(open('results/vllm-gemma-bf16/latency.json'))['modes']['single']['p50'])")s)"

# ── Phase 2: DenseOn ───────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 2a: vLLM-CPU ${BEST_DTYPE} — lightonai/DenseOn"
echo "══════════════════════════════════════════════════════════════"
LABEL="vllm-denseon"
mkdir -p "${RES_DIR}/${LABEL}"
start_vllm "lightonai/DenseOn" "${BEST_DTYPE}"
encode_all "${EMB_DIR}/${LABEL}" "http://localhost:${VLLM_PORT}/v1/embeddings" openai "lightonai/DenseOn" 32
bench_latency "${RES_DIR}/${LABEL}" "http://localhost:${VLLM_PORT}/v1/embeddings" openai "lightonai/DenseOn"
bench_quality "${LABEL}"
stop_container bench-vllm

# ── Phase 2: LateOn (ColBERT) ──────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Phase 2b: vLLM-CPU ${BEST_DTYPE} — lightonai/LateOn (ColBERT)"
echo "══════════════════════════════════════════════════════════════"
LABEL="vllm-lateon"
mkdir -p "${RES_DIR}/${LABEL}"
start_vllm "lightonai/LateOn" "${BEST_DTYPE}"
encode_all "${EMB_DIR}/${LABEL}" "http://localhost:${VLLM_PORT}/v1/embeddings" openai "lightonai/LateOn" 32 "--colbert"
bench_latency "${RES_DIR}/${LABEL}" "http://localhost:${VLLM_PORT}/v1/embeddings" openai "lightonai/LateOn"
bench_quality "${LABEL}" "--colbert"
stop_container bench-vllm

# ── Summary ────────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "══════════════════════════════════════════════════════════════"
python3 - <<'PYEOF'
import json, os, glob

RESULTS = [
    ("tei-gemma",       "TEI-CPU fp32",          "dense"),
    ("vllm-gemma-fp32", "vLLM-CPU fp32 (gemma)", "dense"),
    ("vllm-gemma-bf16", "vLLM-CPU bf16 (gemma)", "dense"),
    ("vllm-denseon",    "vLLM-CPU DenseOn",      "dense"),
    ("vllm-lateon",     "vLLM-CPU LateOn",       "colbert"),
]

print(f"\n{'Model':<28}  {'type':<8}  {'p50(s)':<8}  {'p50 b32(s)':<12}  {'nDCG@10'}")
print("-" * 75)
for label, name, mtype in RESULTS:
    lat_f  = f"results/{label}/latency.json"
    qual_f = f"results/{label}/quality.json"
    lat = json.load(open(lat_f)) if os.path.exists(lat_f) else {}
    qual = json.load(open(qual_f)) if os.path.exists(qual_f) else {}
    p50_single = lat.get("modes",{}).get("single",{}).get("p50","—")
    p50_b32    = lat.get("modes",{}).get("batch-32",{}).get("p50","—")
    ndcg       = qual.get("macro_avg_ndcg", "—")
    print(f"{name:<28}  {mtype:<8}  {str(p50_single):<8}  {str(p50_b32):<12}  {ndcg}")
PYEOF

echo ""
echo "Done. All results in ${RES_DIR}/"
