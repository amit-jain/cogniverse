# Audit Cycle 6 — Cluster: dashboard str.contains without na=False

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `dashboard/tabs/optimization.py:232,518` | C | `spans_df["name"].str.contains("search", case=False)` omitted `na=False`; a null span name (Phoenix can return one) makes `str.contains` yield NaN, which raises `ValueError: Cannot mask with non-boolean array containing NA/NaN` when used as a boolean mask — crashing the tab render | extracted `_filter_search_spans()` with `na=False`, used at both sites (the other 3 optimization sites already had `na=False`) |

## Test (`tests/dashboard/unit/test_optimization_filter.py`)

`_filter_search_spans` on a DataFrame with a `None` name returns only the
search row without raising. The pre-fix idiom raises `ValueError` on the
null-name mask (demonstrated directly).

## Related, deferred

`app.py:1242-1251` (interactive trace search) has the same `str.contains`
without `na=False` on `trace_id`/`operation`. `app.py` runs
`create_default_config_manager()` at module import, so a clean unit test needs
the filter extracted to a side-effect-free util — tracked as a follow-up
rather than shipped untested here.
