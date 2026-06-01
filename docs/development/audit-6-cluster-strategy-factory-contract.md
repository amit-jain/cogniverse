# Audit Cycle 6 — Cluster: StrategyFactory swallowed config typos

Review summary for the Class-E "code contradicts its own documented contract"
finding in the ingestion strategy factory.

## Finding & fix

| Site | Class | Failure on happy path | Fix |
|------|-------|-----------------------|-----|
| `runtime/ingestion/strategy_factory.py` `_create_strategy_instance` | E | both docstrings promise a param typo (e.g. `"models"` for `"model"`) "raises TypeError at construction so the misconfiguration is loud, not silent" — but the method wrapped construction in `try/except TypeError: return None`. A typo silently dropped the strategy, so the pipeline ran without it (e.g. no transcription) — silent data-quality loss | removed the catch; let `TypeError` from a param mismatch propagate (the unresolved-class path still returns `None` for the intentional skip-one-bad-class behaviour) |

## Test-fidelity fix (Class A)

`test_unknown_param_in_profile_surfaces_as_typeerror` was named/docstringed for
the loud-failure contract but asserted the **broken** behaviour
(`strategy_set.transcription is None`) and hedged in a comment ("Factory
currently logs and returns None ... Either behaviour proves..."). Rewritten to
assert `pytest.raises(TypeError)` — the contract its name promises.

## Verification

- Corrected test fails on pre-fix code (`DID NOT RAISE <TypeError>` — the catch
  returned `None`), passes after.
- 61 consumer tests (`test_audio_ingestion`, `test_document_ingestion`,
  `test_image_ingestion`) pass — removing the catch causes no regression
  (valid configs never hit the TypeError path).
