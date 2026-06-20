import sys, json, time, statistics, urllib.request
mode, url, model, n_single = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
LABELS=["PERSON","ORGANIZATION","LOCATION","EVENT","PRODUCT","TECHNOLOGY","CONCEPT"]
Q=["Obama speaking at MIT about climate change","Python tutorial using TensorFlow and PyTorch",
"cooking videos from northern Italy","quantum computing lecture by Richard Feynman",
"stock market analysis for Tesla in 2024","ancient Roman architecture documentary",
"jazz concert recorded in New Orleans","vaccine research conducted at Pfizer",
"SpaceX Falcon 9 rocket launch coverage","Renaissance paintings in the Louvre museum"]
def body(q):
    if mode=="sie": return {"items":[{"text":q}],"params":{"labels":LABELS}}
    return {"text":q,"labels":LABELS,"threshold":0.4}  # own sidecar
def post(q,to=120):
    d=json.dumps(body(q)).encode()
    r=urllib.request.Request(url,data=d,headers={"Content-Type":"application/json","Accept":"application/json"})
    t=time.time()
    try:
        x=urllib.request.urlopen(r,timeout=to); x.read(); return time.time()-t, x.status
    except Exception as e: return time.time()-t, f"ERR {str(e)[:70]}"
post(Q[0]); post(Q[0])  # warmup
ts=[]; st=None
for i in range(n_single):
    dt,s=post(Q[i%len(Q)]); ts.append(dt); st=s
print(f"  single (n={n_single}, distinct queries): med={statistics.median(ts)*1000:.0f}ms min={min(ts)*1000:.0f}ms max={max(ts)*1000:.0f}ms http={st}")
