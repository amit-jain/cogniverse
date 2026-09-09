"""Agent Client Protocol (ACP) surface: cogniverse as an editor-launchable
JSON-RPC-over-stdio agent, backed by the dispatcher.

The public names are resolved lazily so ``python -m cogniverse_runtime.acp``
can claim the stdout protocol channel before the server module pulls in
dspy/fastapi at import time.
"""

STDIN_LINE_LIMIT = 64 * 1024 * 1024

__all__ = ["ACPServer", "ACPError", "ClientConnection", "handle_message", "serve"]


def __getattr__(name: str):
    if name in __all__:
        from cogniverse_runtime.acp import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
