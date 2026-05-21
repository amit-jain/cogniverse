# TODO: Cross-Modal Linker Precision (C5/C7 gap)

## Problem

`CrossModalLinker` (`libs/agents/cogniverse_agents/graph/cross_modal_linker.py`) uses MaxSim cosine over `lightonai/LateOn` (ColBERT-family) multi-vector embeddings to score whether two cross-modal `Mention` pairs refer to the same real-world entity. On the Marie Curie test fixture this fails to discriminate:

| Pair (transcript ↔ vlm/ocr) | Should match? | Observed cosine |
|---|---|---|
| "Marie Curie" ↔ "woman in lab coat" | YES (semantic) | **0.96** |
| "Marie Curie" ↔ "Curie 1903" | YES (lexical) | **0.97** |
| "Marie Curie" ↔ "yellow flowers in glass vase" | NO (unrelated) | **0.95** |
| "woman in lab coat" ↔ "Curie 1903" | NO (transitive only) | **0.95** |

ColBERT-LateOn is a lexical/keyword retriever, not a cross-modal semantic model. The true-positive and true-negative cosines overlap in a ~0.01 band — no scalar threshold separates them.

## Tests this blocks

- `tests/agents/integration/test_cross_modal_linking.py::test_c5_flowers_at_14_emits_no_same_as_edge` — false positive (linker emits a `same_as` edge between "Marie Curie" and "yellow flowers")
- `tests/agents/integration/test_cross_modal_linking.py::test_c7_triangle_emits_two_same_as_edges_byte_equal` — emits 3 pairwise edges instead of 2, because the transitive "woman in lab coat" ↔ "Curie 1903" pair also crosses the threshold

C1, C2, C3, C4, C6, C8 all pass with goldens locked (6/8). The two failing tests surface a real product limitation that no threshold tweak fixes.

## Fix options (in priority order)

### 1. CLIP-class cross-modal encoder (proper fix)
Add a per-modality encoder that maps Person/Object names from transcript text into the same semantic space as VLM-described visual entities. Candidates:
- **OpenAI CLIP** (`openai/clip-vit-base-patch32` or `clip-vit-large-patch14`) — text and image encoders sharing one space, but VLM descriptions are text not images, so we'd be doing text-to-text comparison through a model trained for text-to-image
- **SentenceTransformer all-MiniLM-L12-v2** or **bge-large** — proper text-to-text semantic similarity, trained on contrastive pairs that should pull "lab coat scientist" close to "Marie Curie" while keeping "yellow flowers" far
- **Sentence-T5 / E5** — same idea, different training data

Required work:
- Stand up a sidecar similar to `cogniverse-colbert-pylate` but serving the chosen model
- Update `CrossModalLinker.__init__` to take an optional alternative encoder URL
- Add a config flag to choose ColBERT vs the new model per tenant
- Re-record C5/C7 goldens against the new encoder

### 2. Type-gated linking (cheaper, partial)
Promote the GLiNER `label` (Person, Location, Substance, ...) onto each `Node` and refuse `same_as` between mentions of incompatible types. This kills the "Marie Curie (Person) ↔ yellow flowers (Concept)" false positive but doesn't help the "lab coat (Concept) ↔ Curie 1903 (Concept)" transitive case in C7.

Required work:
- Add `Node.label: str` carrying the GLiNER tag
- Update `DocExtractor` to set it on every Node it produces
- Update `CrossModalLinker.link()` to filter pairs where `label_a != label_b` AND neither side is "Person" (Person can match a visual subject of any non-Person label)

### 3. Higher threshold (band-aid, will not work)
The 0.01 gap between true positives and true negatives means no threshold separates them. Already tested: 0.96 cutoff loses C3 (legitimate positive), 0.97 loses both legitimate positives.

## Recommendation
Ship option 2 first (type-gated) — small, mechanical, fixes C5 without product re-architecting. Then queue option 1 for the broader cross-modal initiative when an embedding-model sidecar is being deployed for unrelated reasons (e.g., reranking).

## Tracking
- Test goldens: `tests/agents/integration/goldens/cross_modal_*.json`
- Failing tests: C5 + C7 above
- Linker code: `libs/agents/cogniverse_agents/graph/cross_modal_linker.py`
- Reference: plan v3 at `/home/amitjain/.claude/plans/yes-first-create-a-toasty-catmull.md` — Change 5
