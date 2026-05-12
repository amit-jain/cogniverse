"""Project-root pytest configuration.

Per pytest 9, ``pytest_plugins`` in non-rootdir conftests is no longer
respected — declare them here so ``tests/fixtures/sidecars.py`` (which
provides ``vllm_sidecar``, ``pylate_sidecar``) and ``tests/fixtures/llm.py``
load for the whole test tree.
"""

pytest_plugins = ["tests.fixtures.llm", "tests.fixtures.sidecars"]
