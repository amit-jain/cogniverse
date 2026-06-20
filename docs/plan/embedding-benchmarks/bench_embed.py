import sys, json, time, statistics, urllib.request
url, proto, model, n_single, n_batch = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
texts = json.load(open("/tmp/inp32_distinct.json"))
def body(items):
    return {"inputs": items[0] if len(items)==1 else items} if proto=="tei" \
        else {"model": model, "input": items}
def post(items, to=900):
    d = json.dumps(body(items)).encode()
    r = urllib.request.Request(url, data=d, headers={"Content-Type":"application/json"})
    t=time.time()
    try:
        x=urllib.request.urlopen(r, timeout=to); x.read(); return time.time()-t, x.status
    except Exception as e:
        return time.time()-t, f"ERR {str(e)[:90]}"
# warmup
post([texts[0]]); post([texts[0]])
# single: n_single iterations, each a DIFFERENT text
s_times=[]; s_stat=None
for i in range(n_single):
    dt,st=post([texts[i % len(texts)]]); s_times.append(dt); s_stat=st
# batch-32: n_batch iterations (all 32 distinct)
b_times=[]; b_stat=None
for i in range(n_batch):
    dt,st=post(texts[:32]); b_times.append(dt); b_stat=st
def stats(xs): return f"med={statistics.median(xs):.3f}s min={min(xs):.3f}s max={max(xs):.3f}s"
thr = 32/statistics.median(b_times)
print(f"  single (n={n_single}, distinct): {stats(s_times)} http={s_stat}")
print(f"  batch32 (n={n_batch}):           {stats(b_times)} http={b_stat}  -> {thr:.1f} texts/s")
