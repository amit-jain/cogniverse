"""Knowledge graph extraction and storage for Cogniverse.

Unified graph model: every file type produces `Node` and `Edge` objects
with the same shape. Per-file-type extractors (code via tree-sitter,
docs via GLiNER + DSPy) are implementation details — consumers only see
the extracted graph.
"""

from cogniverse_agents.graph.graph_manager import GraphManager
from cogniverse_agents.graph.graph_schema import Edge, Node

__all__ = ["Node", "Edge", "GraphManager"]
