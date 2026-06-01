# Audit Cycle 6 — Cluster: profile UPDATE skipped value validation

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `core/validation/profile_validator.py` `validate_update_fields` | B | only checked immutable-field changes; the VALUES of mutable fields were never validated, so a profile UPDATE could write a malformed `strategies` block (e.g. a nonexistent strategy class) that create-time `validate_profile` rejects | also run `_validate_strategies(update_fields["strategies"])` on update |

## Tests

- `TestUpdateValidatesStrategyValues` (new): a non-dict strategy value and a
  description-only update — first errors, second clean. The malformed-value
  test fails on pre-fix (no value validation).
- `test_valid_update_fields` (corrected): used a nonexistent `"NewStrategy"`
  class and asserted "valid" — it relied on the validation gap. Now uses the
  real `FrameSegmentationStrategy`.

Admin profile-API integration suite (21 tests) green — no route regression.
