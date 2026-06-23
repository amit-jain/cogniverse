#!/usr/bin/env python3
"""Encode a corpus.jsonl or queries.jsonl file via a running TEI or vLLM endpoint.

Saves:
  {output_prefix}_vecs.npy   — float32 [N, dim] dense OR object array for ColBERT
  {output_prefix}_ids.txt    — one id per line, same order as rows in _vecs.npy

TEI proto  : POST /embed          body {"inputs": [...]}
OpenAI proto: POST /v1/embeddings  body {"model": ..., "input": [...]}

ColBERT detection: if the first returned embedding is itself a list of lists
(i.e. embedding[0] is a list), treat as multi-vector and save as object array.

Usage:
  python encode.py --url http://localhost:8081/embed --proto tei \\
      --input bright_data/biology/corpus.jsonl --output-prefix embeddings/tei-bio/corpus

  python encode.py --url http://localhost:8082/v1/embeddings --proto openai \\
      --model lightonai/LateOn --input bright_data/biology/corpus.jsonl \\
      --output-prefix embeddings/vllm-lateon-bio/corpus --colbert
"""
import argparse, json, math, os, sys, time, urllib.request
import numpy as np

def post_batch(url, proto, model, texts, timeout=300):
    if proto == "tei":
        body = {"inputs": texts}
    else:
        body = {"model": model, "input": texts}
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    if proto == "tei":
        return resp  # list of embeddings
    return [item["embedding"] for item in resp["data"]]

def is_colbert_response(emb):
    return isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list)

def load_rows(path, key):
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            yield row["id"], row[key]

def encode_file(url, proto, model, input_path, output_prefix, batch_size, force_colbert):
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    # count rows for progress reporting
    with open(input_path) as f:
        n_rows = sum(1 for _ in f)

    key = "query" if "queries" in os.path.basename(input_path) else "text"

    ids_path  = output_prefix + "_ids.txt"
    vecs_path = output_prefix + "_vecs.npy"

    all_ids  = []
    all_vecs = []  # list of arrays; flushed to disk at end
    colbert_mode = force_colbert
    n_done = 0
    batch_ids, batch_texts = [], []

    def flush(ids, texts):
        nonlocal colbert_mode
        vecs = post_batch(url, proto, model, texts)
        if not all_vecs and is_colbert_response(vecs[0]):
            colbert_mode = True
        if colbert_mode:
            for v in vecs:
                all_vecs.append(np.array(v, dtype=np.float32))
        else:
            all_vecs.append(np.array(vecs, dtype=np.float32))
        all_ids.extend(ids)

    t0 = time.time()
    for doc_id, text in load_rows(input_path, key):
        batch_ids.append(doc_id)
        batch_texts.append(text)
        if len(batch_ids) == batch_size:
            flush(batch_ids, batch_texts)
            n_done += len(batch_ids)
            batch_ids, batch_texts = [], []
            if (n_done // batch_size) % 100 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta  = (n_rows - n_done) / rate if rate > 0 else 0
                print(f"  {n_done:>7,}/{n_rows:,}  {rate:.1f} docs/s  ETA {eta/60:.1f}m", flush=True)

    if batch_ids:
        flush(batch_ids, batch_texts)
        n_done += len(batch_ids)

    # save
    with open(ids_path, "w") as f:
        f.write("\n".join(all_ids) + "\n")

    if colbert_mode:
        # ragged object array
        arr = np.empty(len(all_vecs), dtype=object)
        for i, v in enumerate(all_vecs):
            arr[i] = v
        np.save(vecs_path, arr, allow_pickle=True)
    else:
        mat = np.concatenate(all_vecs, axis=0)
        np.save(vecs_path, mat)

    elapsed = time.time() - t0
    mode_tag = "colbert" if colbert_mode else "dense"
    print(f"  done: {n_done:,} docs in {elapsed:.1f}s ({n_done/elapsed:.1f}/s)  mode={mode_tag}")
    print(f"  saved: {vecs_path}  {ids_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",           required=True)
    ap.add_argument("--proto",         choices=["tei", "openai"], required=True)
    ap.add_argument("--model",         default="")
    ap.add_argument("--input",         required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--batch-size",    type=int, default=32)
    ap.add_argument("--colbert",       action="store_true")
    args = ap.parse_args()

    print(f"Encoding {args.input} → {args.output_prefix}_vecs.npy")
    encode_file(args.url, args.proto, args.model, args.input,
                args.output_prefix, args.batch_size, args.colbert)

if __name__ == "__main__":
    main()
