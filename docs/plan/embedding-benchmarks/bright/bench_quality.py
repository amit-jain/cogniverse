#!/usr/bin/env python3
"""Compute nDCG@10 on BRIGHT subsets using pre-encoded embeddings.

Dense mode  (default): FAISS IndexFlatIP over L2-normalised vectors.
ColBERT mode (--colbert): MaxSim scoring with FAISS candidate pre-filtering.

Usage:
  # dense
  python bench_quality.py --data-dir bright_data \\
      --corpus-prefix embeddings/vllm-gemma/corpus \\
      --query-prefix  embeddings/vllm-gemma/queries \\
      --categories biology,economics \\
      --output results/vllm-gemma/quality.json

  # colbert
  python bench_quality.py --data-dir bright_data \\
      --corpus-prefix embeddings/vllm-lateon/corpus \\
      --query-prefix  embeddings/vllm-lateon/queries \\
      --categories biology --colbert \\
      --output results/vllm-lateon/quality.json
"""
import argparse, json, math, os, sys
import numpy as np

def ndcg_at_k(ranked_ids, relevant_ids, k=10):
    gains = [1.0 if doc_id in relevant_ids else 0.0 for doc_id in ranked_ids[:k]]
    ideal = sorted(gains, reverse=True)
    def dcg(gs):
        return sum(g / math.log2(i + 2) for i, g in enumerate(gs))
    return dcg(gains) / (dcg(ideal) + 1e-10)

def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            qrels.setdefault(r["query_id"], set()).add(r["doc_id"])
    return qrels

def l2_normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms

# ── dense mode ────────────────────────────────────────────────────────────────

def eval_dense(corpus_vecs, corpus_ids, query_vecs, query_ids, qrels, k=10):
    import faiss
    dim = corpus_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    cv = l2_normalize(corpus_vecs)
    index.add(cv)

    qv = l2_normalize(query_vecs)
    scores_mat, idx_mat = index.search(qv, k)

    ndcgs = []
    for qi, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        ranked = [corpus_ids[i] for i in idx_mat[qi]]
        ndcgs.append(ndcg_at_k(ranked, qrels[qid], k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0, len(ndcgs)

# ── colbert mode ──────────────────────────────────────────────────────────────

def maxsim_score(q_vecs, d_vecs):
    # q_vecs: [q_len, dim], d_vecs: [d_len, dim] — already L2 normalised
    # score = sum_i max_j dot(q_i, d_j)
    sims = q_vecs @ d_vecs.T  # [q_len, d_len]
    return float(sims.max(axis=1).sum())

def eval_colbert(corpus_vecs_obj, corpus_ids, query_vecs_obj, query_ids, qrels,
                 k=10, candidates=200):
    import faiss

    # flatten all corpus token vectors into one big matrix for candidate retrieval
    # track which doc each token belongs to
    print("  Building flattened FAISS index for ColBERT candidate retrieval...")
    token_to_doc = []
    token_rows   = []
    for doc_idx, doc_vecs in enumerate(corpus_vecs_obj):
        dv = np.array(doc_vecs, dtype=np.float32)
        dv = l2_normalize(dv)
        corpus_vecs_obj[doc_idx] = dv  # normalise in-place
        for _ in dv:
            token_to_doc.append(doc_idx)
        token_rows.append(dv)

    flat_mat = np.concatenate(token_rows, axis=0).astype(np.float32)
    dim = flat_mat.shape[1]
    token_to_doc = np.array(token_to_doc, dtype=np.int32)

    flat_index = faiss.IndexFlatIP(dim)
    flat_index.add(flat_mat)
    print(f"  Flat index: {flat_mat.shape[0]:,} token vectors, dim={dim}")

    ndcgs = []
    for qi, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        qv = np.array(query_vecs_obj[qi], dtype=np.float32)
        qv = l2_normalize(qv)  # [q_len, dim]

        # retrieve candidate docs: for each query token, get top-k token hits
        _, token_hits = flat_index.search(qv, candidates)
        cand_doc_ids  = set(token_to_doc[token_hits.ravel()].tolist())

        # exact MaxSim on candidates
        candidate_scores = {}
        for doc_idx in cand_doc_ids:
            dv = corpus_vecs_obj[doc_idx]
            candidate_scores[doc_idx] = maxsim_score(qv, dv)

        ranked_idxs = sorted(candidate_scores, key=candidate_scores.get, reverse=True)[:k]
        ranked_ids  = [corpus_ids[i] for i in ranked_idxs]
        ndcgs.append(ndcg_at_k(ranked_ids, qrels[qid], k))

        if (qi + 1) % 20 == 0:
            print(f"  query {qi+1}/{len(query_ids)}  running nDCG@10={np.mean(ndcgs):.4f}")

    return float(np.mean(ndcgs)) if ndcgs else 0.0, len(ndcgs)

# ── main ──────────────────────────────────────────────────────────────────────

def eval_category(category, data_dir, corpus_prefix_tpl, query_prefix_tpl, colbert, k):
    qrels_path   = os.path.join(data_dir, category, "qrels.jsonl")
    corpus_vecs_path = corpus_prefix_tpl.format(cat=category) + "_vecs.npy"
    corpus_ids_path  = corpus_prefix_tpl.format(cat=category) + "_ids.txt"
    query_vecs_path  = query_prefix_tpl.format(cat=category)  + "_vecs.npy"
    query_ids_path   = query_prefix_tpl.format(cat=category)  + "_ids.txt"

    for p in [qrels_path, corpus_vecs_path, corpus_ids_path, query_vecs_path, query_ids_path]:
        if not os.path.exists(p):
            print(f"  SKIP {category}: missing {p}")
            return None

    qrels = load_qrels(qrels_path)

    allow_pickle = colbert
    corpus_vecs = np.load(corpus_vecs_path, allow_pickle=allow_pickle)
    query_vecs  = np.load(query_vecs_path,  allow_pickle=allow_pickle)

    with open(corpus_ids_path) as f:
        corpus_ids = [l.strip() for l in f if l.strip()]
    with open(query_ids_path) as f:
        query_ids  = [l.strip() for l in f if l.strip()]

    print(f"\n  [{category}]  corpus={len(corpus_ids):,}  queries={len(query_ids):,}  "
          f"qrels_queries={len(qrels):,}")

    if colbert:
        score, n = eval_colbert(corpus_vecs, corpus_ids, query_vecs, query_ids, qrels, k)
    else:
        score, n = eval_dense(corpus_vecs, corpus_ids, query_vecs, query_ids, qrels, k)

    print(f"  nDCG@{k} = {score:.4f}  (over {n} queries)")
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",       default="bright_data")
    ap.add_argument("--corpus-prefix",  required=True,
                    help="prefix template, use {cat} as placeholder, e.g. embeddings/tei/{cat}/corpus")
    ap.add_argument("--query-prefix",   required=True,
                    help="prefix template with {cat}")
    ap.add_argument("--categories",     default="biology,economics,stackoverflow")
    ap.add_argument("--output",         default="quality_results.json")
    ap.add_argument("--colbert",        action="store_true")
    ap.add_argument("--k",              type=int, default=10)
    args = ap.parse_args()

    categories = args.categories.split(",")
    results    = {"colbert": args.colbert, "k": args.k, "categories": {}}

    for cat in categories:
        score = eval_category(cat, args.data_dir, args.corpus_prefix,
                              args.query_prefix, args.colbert, args.k)
        if score is not None:
            results["categories"][cat] = round(score, 4)

    valid = [v for v in results["categories"].values()]
    if valid:
        macro = round(sum(valid) / len(valid), 4)
        results["macro_avg_ndcg"] = macro
        print(f"\n{'─'*50}")
        print(f"  Macro-avg nDCG@{args.k} over {len(valid)} categories: {macro:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {args.output}")

if __name__ == "__main__":
    main()
