#!/usr/bin/env bash
# Sample /admin/debug/memsnap at fixed intervals.
#
# Usage:
#   bash scripts/memsnap_sampler.sh                # 15-min interval, session long
#   INTERVAL=300 ITERS=20 bash scripts/memsnap_sampler.sh
#
# Output: one JSON per sample in $OUT_DIR (default /tmp/cogniverse_memsnap/).
# Filename pattern: snap_<minute>.json where <minute> is offset from start.
# Run after the first snapshot so subsequent samples carry real growth diffs
# via tracemalloc's compare_to(prev).
#
# Prereq: runtime chart deployed with COGNIVERSE_DEBUG_MEM=1 (already set
# in values.k3s.yaml).

set -euo pipefail

URL=${URL:-http://localhost:28000/admin/debug/memsnap}
INTERVAL=${INTERVAL:-900}   # seconds between snapshots — default 15 min
ITERS=${ITERS:-12}          # total snapshots — default 12 × 15 min = 3 h
OUT_DIR=${OUT_DIR:-/tmp/cogniverse_memsnap}
TOP_N=${TOP_N:-40}

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/snap_*.json

start=$(date +%s)
for i in $(seq 0 $((ITERS - 1))); do
  now=$(date +%s)
  minute=$(( (now - start) / 60 ))
  out="$OUT_DIR/snap_${minute}min.json"
  echo "[$(date '+%H:%M:%S')] sample $i → $out"
  curl -sS -X POST "${URL}?top_n=${TOP_N}" --max-time 60 > "$out" || {
    echo "  ! curl failed (runtime may be wedged); continuing"
    echo "{\"error\": \"curl_failed\", \"minute\": $minute}" > "$out"
  }
  # Print running total
  python3 -c "
import json
with open('$out') as f:
    d = json.load(f)
print(f'  total_mb={d.get(\"total_mb\", \"?\")} top_growth_count={len(d.get(\"top_growth\", []))}')
" 2>/dev/null || true

  [ $i -lt $((ITERS - 1)) ] && sleep "$INTERVAL"
done

echo "done. snapshots in $OUT_DIR"
ls -la "$OUT_DIR"
