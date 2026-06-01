# Audit Cycle 6 — Cluster: LearnedReranker rejection tests were hollow

| Site | Class | Problem | Fix |
|------|-------|---------|-----|
| `tests/routing/unit/test_learned_reranker.py` `test_reranker_raises_on_heuristic_model`, `test_reranker_raises_on_unknown_model` | HOLLOW-TEST | both called `LearnedReranker()` with no `config_manager`, so they raised — and asserted — `"config_manager is required"`, never exercising the model-rejection logic they claim to test | pass `config_manager=mock_config_manager` so the real validation runs; assert the actual messages (`"requires a learned model"`, `"not found in supported_models"`) |

Test-only change. Verified non-hollow by mutation: neutering the production
`heuristic` raise makes the heuristic test fail (it passed vacuously before).
