# Audit Cycle 6 — Cluster: ApprovalStorage ABC contract drift

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `agents/approval/interfaces.py` `ApprovalStorage.update_item` | A | the ABC declared `update_item(self, item)`, but the concrete `ApprovalStorageImpl` and all 3 `human_approval_agent` call sites use `update_item(item, batch_id=...)`. A faithful subclass written to the ABC would break those callers with a `TypeError` on the `batch_id` kwarg. | added `batch_id: Optional[str] = None` to the abstract signature + docstring, matching the real contract |

## Test

`TestApprovalStorageContract::test_update_item_abc_declares_batch_id`
(`tests/routing/unit/synthetic/test_approval_system.py`) asserts both the ABC
and the concrete impl signatures contain `batch_id`. Fails on pre-fix (ABC had
only `self, item`).
