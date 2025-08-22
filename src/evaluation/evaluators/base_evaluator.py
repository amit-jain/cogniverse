"""
Base evaluator class that doesn't create spans
"""

from phoenix.experiments.evaluators.base import Evaluator as PhoenixEvaluator
import functools


class NoSpanEvaluator(PhoenixEvaluator):
    """Base evaluator that prevents span creation"""

    def evaluate(self, **kwargs):
        """Wrapper that prevents span creation"""
        # Get the actual evaluate method from the subclass
        if hasattr(self, "_evaluate"):
            return self._evaluate(**kwargs)
        else:
            raise NotImplementedError("Subclass must implement _evaluate method")


def no_span(func):
    """Decorator to prevent span creation in evaluator methods"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Just call the function without any tracing
        return func(*args, **kwargs)

    # Mark this function to skip instrumentation
    wrapper._skip_instrumentation = True
    return wrapper
