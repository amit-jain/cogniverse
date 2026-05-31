# Audit Cycle 6 — Cluster: LM/backend numeric-field coercion

Review summary for the Class-C "raw `float()` / direct comparison on an
LM-or-backend-produced numeric field" cluster. DSPy modules and search
backends return confidence/relevance/score as floats **or** label strings
(`"high"`), percent strings (`"85%"`), or other loose shapes. A bare
`float(x)` or `x > 0.7` then raises `ValueError`/`TypeError` on the happy
path the moment a real model returns a non-numeric value.

Canonical coercion helper: `cogniverse_foundation.confidence.parse_confidence`
(maps every shape to `[0.0, 1.0]`, falls back to `default`). Relocated to
foundation in commit `8374a104` so every layer can share one implementation;
`cogniverse_agents._confidence` re-exports it.

All fixes are local on `main` (unpushed). Each ships a regression test that
fails on the pre-fix code and passes after.

## Findings & fixes

| # | Site | Class | Failure on happy path | Fix |
|---|------|-------|-----------------------|-----|
| 1 | `core/common/vlm_interface.py:87` | C | `float(result.relevance_score)` on a DSPy field — `"high"` → ValueError crashes `analyze_visual_content` | `parse_confidence(...)` (commit `8374a104`) |
| 2 | `evaluation/online_evaluator.py:173` | C | `float(routing_attrs.get("confidence", 0.5))` on a routing-span attribute — label → ValueError crashes `_eval_confidence_calibration` | `parse_confidence(..., default=0.5)` |
| 3 | `finetuning/evaluation/adapter_evaluator.py:256,261` | C | `confidence = pred_json.get("confidence", 0.5)` then `total_confidence += confidence`; a string confidence raises `TypeError` **not** caught by the `except (JSONDecodeError, KeyError)` → whole eval loop dies | `parse_confidence(..., default=0.5)` |
| 4 | `agents/search_agent.py:1895` | C | `if dspy_result.confidence > 0.7:` — `"high" > 0.7` raises `TypeError`, swallowed by the surrounding `except` → DSPy query enhancement silently disabled for any LM returning a label | `if parse_confidence(dspy_result.confidence) > 0.7:` |
| 5 | `messaging/telegram_handler.py:57` | C | `float(score)` on a backend result score — malformed score → ValueError drops the **entire** result list from the Telegram reply | local `try/except (TypeError, ValueError)` (messaging has no foundation dep, and the value is a display score, not a confidence band) |

## Tests (all fail on pre-fix code)

| Finding | Test | Seam | Strong assertion |
|---------|------|------|------------------|
| 1 | `tests/core/unit/test_vlm_interface_relevance.py` | stub `dspy.Predict` returning `relevance_score="high"` | `relevance_score == 0.9` |
| 2 | `tests/evaluation/unit/test_online_evaluator_confidence.py` | call `_eval_confidence_calibration` with `{"confidence": "high"}` | `score == 0.9`, `label == "well_calibrated"`; missing → `0.5` |
| 3 | `tests/finetuning/test_adapter_evaluator.py::TestEvaluateModelConfidenceCoercion` | fake model+tokenizer into `_evaluate_model`, prediction JSON with `"confidence": "high"` | `accuracy == 1.0`, `avg_confidence == 0.9` |
| 4 | `tests/agents/unit/test_search_agent.py::TestDspyConfidenceGate` | real `SearchAgent`, `call_dspy` → `confidence="high"`, capture `_search_by_text` query | enhanced query (`"enhanced cats"`) is the one searched |
| 5 | `tests/messaging/unit/test_telegram_handler.py::TestFormatResults` | call `_format_results` with `score="high"` and with `score=0.85` | malformed → result kept, no `%`; numeric → `"1. Clip (85%)"` |

`parse_confidence` itself is exhaustively unit-tested in
`tests/foundation/unit/test_confidence.py` (20 shape cases) and the agents
re-export is asserted to be the same function object.

## Not in this cluster (related, deferred)

- `core/query/encoders.py` VideoPrism `"lvt" in name.lower()` substring routing
  (Class C, needs model-loader boundary) — tracked separately.
- Other `float()` LM coercions surfaced by the Class-C hunt regex should reuse
  `cogniverse_foundation.confidence.parse_confidence`.
