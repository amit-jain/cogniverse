# Audit Cycle 6 — Cluster: cache cleanup_on_startup never ran

Review summary for the Class-E "flag set in `__init__`, never read" finding in
the structured-filesystem cache backend.

## Finding & fix

| Site | Class | Failure on happy path | Fix |
|------|-------|-----------------------|-----|
| `core/common/cache/backends/structured_filesystem.py` | E | `__init__` set `self._needs_cleanup = True` when `cleanup_on_startup and enable_ttl`, but the flag was **never read** and `_cleanup_expired()` had **no production caller** — so the configured startup sweep never ran and expired entries from a previous run accumulated forever | added `_run_startup_cleanup_if_needed()` (consumes the flag once, clears it before awaiting so concurrent first calls don't double-sweep) and call it at the top of every public async op (`get`/`set`/`delete`/`exists`/`clear`/`get_stats`/`get_metadata`/`list_keys`). `__init__` is sync and cannot await, so the sweep is deferred to first async use. |

## Tests (`tests/core/unit/test_structured_filesystem_cleanup.py`)

Real backend over a real tmp filesystem; an entry is written then its metadata
back-dated so it is expired (the previous-run scenario).

| Test | Assertion |
|------|-----------|
| `test_startup_cleanup_purges_expired_entry_on_first_op` | with `cleanup_on_startup=True`, the expired file survives sync `__init__` but is gone after the first `get`; `_needs_cleanup` flips to `False`. Fails on pre-fix (file never purged). |
| `test_no_startup_cleanup_leaves_expired_file_until_accessed` | with `cleanup_on_startup=False`, the expired file remains after an unrelated `get` (no proactive sweep) |
