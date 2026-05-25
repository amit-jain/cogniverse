"""Graph-manager binding for KG-aware agents."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cogniverse_agents.graph.graph_manager import GraphManager


class GraphBindableMixin:
    """Bind a single ``GraphManager`` to a KG-aware agent.

    The seven single-graph KG agents (citation tracing, temporal reasoning,
    audit explanation, KG traversal, knowledge summarization, multi-document
    synthesis, contradiction reconciliation) each bind one GraphManager after
    construction and read structured Node/Edge rows from it. This mixin owns
    the ``_graph_manager`` slot, the ``set_graph_manager`` setter, and the
    ``_require_graph_manager`` guard so each agent doesn't redeclare them.

    Agents that bind *multiple* managers (FederatedQueryAgent,
    CrossTenantComparisonAgent) keep their own plural ``set_graph_managers``.
    """

    _graph_manager: Optional["GraphManager"] = None

    def set_graph_manager(self, graph_manager: "GraphManager") -> None:
        """Bind the GraphManager this agent reads Node/Edge rows from."""
        self._graph_manager = graph_manager

    def _require_graph_manager(self, method: str) -> "GraphManager":
        """Return the bound GraphManager or raise, naming the calling method.

        ``method`` is the public method that needs the binding (e.g.
        ``"trace"``); it appears in the error so the caller knows what to bind
        before invoking.
        """
        if self._graph_manager is None:
            raise RuntimeError(
                f"{type(self).__name__}.{method} requires a GraphManager — call "
                f"set_graph_manager(...) before invoking .{method}()."
            )
        return self._graph_manager
