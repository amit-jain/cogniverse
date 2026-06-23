#!/usr/bin/env python3
"""Latency and throughput benchmark for TEI and vLLM embedding endpoints.

Measures single-text and multi-text batch latency at configurable batch sizes,
reporting p50/p95/p99/min/max and throughput. Saves results as JSON.

Usage:
  python bench_latency.py --url http://localhost:8081/embed --proto tei
  python bench_latency.py --url http://localhost:8082/v1/embeddings --proto openai \\
      --model google/embeddinggemma-300m --batch-sizes 1,8,32
"""
import argparse, json, math, os, statistics, sys, time, urllib.request
from pathlib import Path

def post(url, proto, model, texts, timeout=900):
    if proto == "tei":
        body = {"inputs": texts[0] if len(texts) == 1 else texts}
    else:
        body = {"model": model, "input": texts}
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return time.perf_counter() - t0, None
    except Exception as e:
        return time.perf_counter() - t0, str(e)[:120]

def percentile(xs, p):
    xs = sorted(xs)
    idx = (len(xs) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)

def stats(times):
    return {
        "p50":  round(percentile(times, 50),  3),
        "p95":  round(percentile(times, 95),  3),
        "p99":  round(percentile(times, 99),  3),
        "min":  round(min(times),             3),
        "max":  round(max(times),             3),
        "mean": round(statistics.mean(times), 3),
    }

def run_mode(url, proto, model, texts, batch_size, n_iters, n_warmup, label):
    batch = texts[:batch_size]
    # warmup
    for _ in range(n_warmup):
        post(url, proto, model, batch)

    times, errors = [], []
    for i in range(n_iters):
        # rotate through distinct texts to avoid caching
        start = (i * batch_size) % len(texts)
        b = (texts + texts)[start:start + batch_size]
        dt, err = post(url, proto, model, b)
        times.append(dt)
        if err:
            errors.append(err)

    s = stats(times)
    throughput = round(batch_size / s["p50"], 2)
    err_rate   = len(errors) / n_iters

    print(f"  {label:<20s}  p50={s['p50']:.3f}s  p95={s['p95']:.3f}s  p99={s['p99']:.3f}s  "
          f"min={s['min']:.3f}s  max={s['max']:.3f}s  tput={throughput:.1f}t/s"
          + (f"  errors={len(errors)}/{n_iters}" if errors else ""))
    return {**s, "throughput_texts_per_sec": throughput, "batch_size": batch_size,
            "n_iters": n_iters, "error_rate": err_rate}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",          required=True)
    ap.add_argument("--proto",        choices=["tei", "openai"], required=True)
    ap.add_argument("--model",        default="")
    ap.add_argument("--input",        default=None, help="path to inputs JSON (list of strings)")
    ap.add_argument("--n-single",     type=int, default=10)
    ap.add_argument("--n-batch",      type=int, default=5)
    ap.add_argument("--n-warmup",     type=int, default=2)
    ap.add_argument("--batch-sizes",  default="1,8,32")
    ap.add_argument("--output",       default="latency_results.json")
    args = ap.parse_args()

    # load inputs
    if args.input:
        input_path = args.input
    else:
        # look for inputs file relative to this script
        here = Path(__file__).parent
        input_path = here.parent / "inputs_2048tok_32distinct.json"
    with open(input_path) as f:
        texts = json.load(f)
    if not isinstance(texts, list):
        texts = list(texts.values())
    print(f"Loaded {len(texts)} input texts from {input_path}")

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    results = {"url": args.url, "proto": args.proto, "model": args.model, "modes": {}}

    print(f"\nEndpoint: {args.url}  proto={args.proto}  model={args.model or '—'}")
    print("-" * 90)

    # single-text mode
    key = "single"
    r = run_mode(args.url, args.proto, args.model, texts, 1, args.n_single, args.n_warmup, key)
    results["modes"][key] = r

    # batch modes
    for bs in batch_sizes:
        if bs == 1:
            continue  # already covered above
        key = f"batch-{bs}"
        r = run_mode(args.url, args.proto, args.model, texts, bs, args.n_batch, args.n_warmup, key)
        results["modes"][key] = r

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output}")

if __name__ == "__main__":
    main()
