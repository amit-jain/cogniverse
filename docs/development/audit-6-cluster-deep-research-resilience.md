# Audit Cycle 6 — Cluster: DeepResearchAgent evidence gathering

Review summary for two Class-D findings in `deep_research_agent.py`.

## Findings & fixes

| # | Site | Class | Failure on happy path | Fix |
|---|------|-------|-----------------------|-----|
| 1 | `_search_parallel` | D | `asyncio.gather(*tasks, return_exceptions=False)` — a single sub-question whose backend search raised aborted the **entire** research run | each `search_one` catches its own exception and returns empty evidence (`results=[]` + `error`), so the gather never propagates and the remaining sub-questions still complete |
| 2 | `_evaluate_evidence` | D | the LLM-facing evidence summary used `len(e.get("results", []))`; for a non-list payload (string/dict from some `search_fn`) this reported a character/key count, not a result count | added `_result_count()` (list → length; any other non-empty payload → 1; empty/None → 0), consistent with `_extract_citations`'s list handling |

## Tests (`tests/agents/unit/test_deep_research_agent.py`, fail on pre-fix)

| Finding | Test | Assertion |
|---------|------|-----------|
| 1 | `test_one_failed_subsearch_does_not_abort_the_rest` | `search_fn` raises for `"boom"`; the `"ok"` question still returns its results, `"boom"` returns `[]` + an `error` (pre-fix: gather raises, whole run aborts) |
| 2 | `test_result_count_handles_non_list_shapes` | list→len, `[]`→0, string→1, dict→1, None→0 |
