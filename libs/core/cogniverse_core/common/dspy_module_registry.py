"""DSPy module / optimizer registries.

The implementation moved to ``cogniverse_foundation.common.dspy_module_registry``
so foundation's config API mixin can list available modules without importing
upward into core. Re-exported here unchanged so existing
``from cogniverse_core.common.dspy_module_registry import ...`` call sites keep
working.
"""

from cogniverse_foundation.common.dspy_module_registry import (
    DSPyModuleRegistry,
    DSPyOptimizerRegistry,
)

__all__ = ["DSPyModuleRegistry", "DSPyOptimizerRegistry"]
