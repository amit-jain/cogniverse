"""Load the modules dspy would otherwise replace with lazy proxies.

dspy 3.4 puts a proxy into ``sys.modules`` for each of these that is not yet
imported, and a later submodule import through the proxy fails with a
circular ImportError (stanfordnlp/dspy#10516). Every package that imports
dspy calls ``load_dspy_dependencies`` before its first dspy import.
"""

import anyio
import jiter
import numpy
import openai


def load_dspy_dependencies() -> None:
    """Load a proxy dspy already put in place for any of these modules."""
    for module in (anyio, jiter, numpy, openai):
        dir(module)
