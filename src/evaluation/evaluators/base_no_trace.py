"""
Base class for evaluators without tracing
"""

import os
from contextlib import contextmanager


@contextmanager
def no_tracing():
    """Context manager to temporarily disable tracing"""
    original = os.environ.get("OTEL_SDK_DISABLED")
    os.environ["OTEL_SDK_DISABLED"] = "true"
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("OTEL_SDK_DISABLED", None)
        else:
            os.environ["OTEL_SDK_DISABLED"] = original
