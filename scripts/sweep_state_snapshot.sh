#!/usr/bin/env bash
# Snapshots cluster + Vespa + Mem0 registry state every INTERVAL seconds.
# Designed to run in parallel with run_e2e_batched.sh so a post-mortem
# can correlate the moment K-System tests start failing with what
# changed in the underlying state (schema count, registry size, pod
# restart, etc.).
#
# Output: /tmp/cogniverse_e2e_runs/sweep_state.jsonl — one JSON record
# per tick. Inspect with `jq` after the sweep finishes.
set -u

OUT="${OUT:-/tmp/cogniverse_e2e_runs/sweep_state.jsonl}"
INTERVAL="${INTERVAL:-60}"
VESPA_CFG="${VESPA_CFG:-http://localhost:19071}"
RUNTIME="${RUNTIME:-http://localhost:28000}"
BATCH_LOG="${BATCH_LOG:-/tmp/cogniverse_e2e_runs/batch2.log}"

mkdir -p "$(dirname "$OUT")"
echo "snapshot → $OUT (interval=${INTERVAL}s)" >&2

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  # Vespa schemas
  schema_count="$(curl -sS -m 10 "$VESPA_CFG/application/v2/tenant/default/application/default/environment/prod/region/default/instance/default/content/schemas/" 2>/dev/null \
    | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); items=d if isinstance(d,list) else d.get('children',[])
    print(len(items))
except Exception:
    print(-1)" 2>/dev/null || echo -1)"

  # Runtime pod name + age + restart count
  pod_json="$(kubectl -n cogniverse get pods -o json 2>/dev/null \
    | python3 -c "import json,sys
try:
    d=json.load(sys.stdin)
    runtime=[p for p in d['items'] if any('runtime' in (c.get('name') or '') for c in p['spec']['containers']) and p['metadata']['name'].startswith('cogniverse-runtime-')]
    if not runtime:
        print('{}'); raise SystemExit
    p = runtime[0]
    name = p['metadata']['name']
    start = p['status'].get('startTime', '')
    restarts = sum(c.get('restartCount', 0) for c in p['status'].get('containerStatuses', []))
    ready = sum(1 for c in p['status'].get('containerStatuses', []) if c.get('ready'))
    total = len(p['status'].get('containerStatuses', []))
    print(json.dumps({'name': name, 'start': start, 'restarts': restarts, 'ready': f'{ready}/{total}'}))
except Exception:
    print('{}')" 2>/dev/null || echo '{}')"

  # Tenant_metadata doc count
  tenant_count="$(curl -sS -m 10 "http://localhost:8080/search/?yql=select+*+from+sources+tenant_metadata+where+true&hits=0" 2>/dev/null \
    | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print(d.get('root',{}).get('fields',{}).get('totalCount',-1))
except Exception:
    print(-1)" 2>/dev/null || echo -1)"

  # config_metadata doc count (proxy for registry size)
  registry_count="$(curl -sS -m 10 "http://localhost:8080/search/?yql=select+*+from+sources+config_metadata+where+true&hits=0" 2>/dev/null \
    | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); print(d.get('root',{}).get('fields',{}).get('totalCount',-1))
except Exception:
    print(-1)" 2>/dev/null || echo -1)"

  # Last test reported in the sweep log (so we can align snapshots to test boundaries)
  last_test="$(tail -2 "$BATCH_LOG" 2>/dev/null | grep -oE 'tests/[^ ]+' | tail -1 || echo)"

  printf '{"ts":"%s","schemas":%s,"tenant_metadata_docs":%s,"config_metadata_docs":%s,"pod":%s,"last_test":"%s"}\n' \
    "$ts" "$schema_count" "$tenant_count" "$registry_count" "$pod_json" "$last_test" >> "$OUT"

  sleep "$INTERVAL"
done
