"""Generate 32 distinct ~2040-token inputs for the embedding benchmarks.

Each text is a different topic, padded with repeated topical sentences until it
reaches >=2048 tokens under embeddinggemma's tokenizer, then truncated to 2040
tokens (just under the model's 2048 limit). Output: inputs_2048tok_32distinct.json
(a JSON list of 32 unique strings). Used for both single (one text per iteration)
and batch-32 (all 32 at once) timing in bench_embed.py.
"""

import json

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")

topics = [
    "neural retrieval", "climate policy", "ancient rome", "quantum computing",
    "marine biology", "supply chain logistics", "renaissance art",
    "genome sequencing", "monetary policy", "wildfire ecology",
    "semiconductor fabrication", "jazz harmony", "urban planning",
    "vaccine immunology", "orbital mechanics", "contract law",
    "glacier dynamics", "protein folding", "distributed databases",
    "medieval trade", "solar cell physics", "linguistic typology",
    "coral reef restoration", "cryptography", "volcanology",
    "behavioral economics", "deep sea exploration", "cellular respiration",
    "graph theory", "desert agriculture", "radio astronomy", "forensic science",
]

texts = []
for i, topic in enumerate(topics):
    base = (
        f"In the study of {topic}, researchers in document {i} examine how "
        f"systems encode and retrieve information about {topic} across many "
        f"varied and detailed real world scenarios and corpora. "
    )
    s = base
    while len(tok(s)["input_ids"]) < 2048:
        s += base
    ids = tok(s)["input_ids"][:2040]
    texts.append(tok.decode(ids, skip_special_tokens=True))

ntoks = [len(tok(t)["input_ids"]) for t in texts]
json.dump(texts, open("inputs_2048tok_32distinct.json", "w"))
print(
    f"generated {len(texts)} distinct texts, token range {min(ntoks)}-{max(ntoks)}, "
    f"all unique: {len(set(texts)) == len(texts)}"
)
