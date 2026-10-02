import pytest

# Helper modules whose functions assert on behalf of tests: rewrite them so a
# failing assertion reports its operands, as it does inside a test module.
pytest.register_assert_rewrite(
    "tests.e2e.batch_optimization",
    "tests.e2e.coding_contract",
)
