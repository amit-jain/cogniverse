#!/usr/bin/env python3
"""Generate proportional 1/5 subsets of BRIGHT per category.

Relevant docs (gold_ids) are always included; remaining slots filled with
random negatives so every query retains its full set of positive docs.

Usage:
  python gen_bright_subset.py --output bright_data/
  python gen_bright_subset.py --categories biology,economics --fraction 0.2
"""
import argparse, json, math, os, random, sys

ALL_CATEGORIES = [
    "biology", "earth_science", "economics", "psychology", "robotics",
    "stackoverflow", "sustainable_living", "leetcode", "pony",
    "aops", "theoremqa_questions", "theoremqa_theorems",
]

def load_bright_category(category):
    from datasets import load_dataset
    corpus_ds  = load_dataset("xlangai/BRIGHT", f"{category}_corpus",  split="test", trust_remote_code=True)
    queries_ds = load_dataset("xlangai/BRIGHT", category,              split="test", trust_remote_code=True)
    return corpus_ds, queries_ds

def process_category(category, fraction, output_dir, seed):
    corpus_ds, queries_ds = load_bright_category(category)

    corpus_by_id = {row["id"]: row for row in corpus_ds}
    all_ids      = list(corpus_by_id.keys())
    n_total      = len(all_ids)

    # collect all gold doc ids referenced by any query
    must_include = set()
    for row in queries_ds:
        for gid in row.get("gold_ids", []):
            if gid in corpus_by_id:
                must_include.add(gid)

    target = math.ceil(n_total * fraction)
    negatives = [d for d in all_ids if d not in must_include]
    random.seed(seed)
    n_fill = max(0, target - len(must_include))
    sampled_neg = set(random.sample(negatives, min(n_fill, len(negatives))))
    keep_ids    = must_include | sampled_neg

    cat_dir = os.path.join(output_dir, category)
    os.makedirs(cat_dir, exist_ok=True)

    # corpus subset
    with open(os.path.join(cat_dir, "corpus.jsonl"), "w") as f:
        for doc_id in keep_ids:
            row = corpus_by_id[doc_id]
            f.write(json.dumps({"id": row["id"], "title": row.get("title",""), "text": row["text"]}) + "\n")

    # all queries
    with open(os.path.join(cat_dir, "queries.jsonl"), "w") as f:
        for row in queries_ds:
            f.write(json.dumps({"id": row["id"], "query": row["query"]}) + "\n")

    # qrels (only gold_ids present in subset — all of them by construction)
    n_qrels = 0
    with open(os.path.join(cat_dir, "qrels.jsonl"), "w") as f:
        for row in queries_ds:
            for gid in row.get("gold_ids", []):
                if gid in keep_ids:
                    f.write(json.dumps({"query_id": row["id"], "doc_id": gid, "relevance": 1}) + "\n")
                    n_qrels += 1

    n_queries = len(queries_ds)
    print(f"  {category:30s}  corpus {n_total:>7,} → {len(keep_ids):>6,}  "
          f"must-include {len(must_include):>4,}  queries {n_queries:>4,}  qrels {n_qrels:>5,}")
    return len(keep_ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",     default="bright_data")
    ap.add_argument("--fraction",   type=float, default=0.2)
    ap.add_argument("--categories", default="all")
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    cats = ALL_CATEGORIES if args.categories == "all" else args.categories.split(",")
    print(f"Generating {args.fraction:.0%} subsets → {args.output}/\n")
    print(f"  {'category':30s}  {'orig':>7}   {'subset':>6}  {'must':>8}  {'queries':>7}  {'qrels':>6}")
    print("  " + "-"*75)
    for cat in cats:
        try:
            process_category(cat, args.fraction, args.output, args.seed)
        except Exception as e:
            print(f"  {cat}: ERROR — {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
