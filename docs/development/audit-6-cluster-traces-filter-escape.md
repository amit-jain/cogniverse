# Audit Cycle 6 — Cluster: trace experiment filter interpolation

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `evaluation/data/traces.py:139` `get_traces_by_experiment` | E/C | `filter_condition` interpolated `profile`/`strategy` into single-quoted literals raw; Phoenix evaluates the filter as a Python expression, so a value containing a quote broke (or could inject) the filter | use `{profile!r}`/`{strategy!r}` — `repr()` yields a correctly-quoted/escaped Python string literal |

## Test (`tests/evaluation/unit/test_traces_filter_escape.py`, fails on pre-fix)

Captures the `filter_condition` for `profile="pro'file"` and asserts it contains
`repr("pro'file")`. Pre-fix the raw single-quote interpolation produced a broken
literal.
