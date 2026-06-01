# Audit Cycle 6 — Cluster: pin/promote ignore add_memory's None return

`MemoryManager.add_memory` is documented `Optional[str]` — None when storage
deduplicated/dropped; "callers that require storage to succeed should check."

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `core/memory/pinning.py:247` `PinService.pin` | E | built `PinRecord(memory_id=None)` (field typed `str`) on a None return → a pin with no id that unpin/lookup can't resolve | raise RuntimeError("pin … not persisted") when add_memory returns None |
| `core/memory/federation.py:227` `FederationService.promote_to_org_trunk` | E | `promoted_memory_id=str(promoted_id or "")` → empty-string id, silent fake success | raise RuntimeError("promotion … not persisted") when add_memory returns None; return the real id otherwise |

## Tests (fail on pre-fix)

- `test_pin_service.py::TestPinStorageFailure` — pin-record write returns None → `pin()` raises.
- `test_federation.py::TestPromotionStorageFailure` — org-trunk write returns None → `promote_to_org_trunk()` raises.
