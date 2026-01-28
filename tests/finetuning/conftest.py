"""
Pytest configuration for finetuning tests.

Handles test isolation for mocking issues with trl imports.
"""

import sys
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True, scope="function")
def isolate_transformers_imports():
    """
    Isolate transformers imports between tests to prevent mock pollution with trl.

    The trl library uses lazy imports that inspect transformers.TrainingArguments.__mro__.
    If TrainingArguments is mocked without proper spec, trl imports fail.
    This fixture ensures transformers modules are cleared between tests.
    """
    # Yield to run the test
    yield

    # After test: clear trl trainer modules that depend on transformers
    modules_to_clear = [
        "trl.trainer.sft_trainer",
        "trl.trainer.dpo_trainer",
        "trl.trainer",
    ]

    for module_name in modules_to_clear:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Also ensure no patches remain active
    patch.stopall()
