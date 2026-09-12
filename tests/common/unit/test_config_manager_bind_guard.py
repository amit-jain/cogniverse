"""An agent's ConfigManager is written in exactly one place.

``ConfigManagerAware.bind_config_manager`` is the only writer of the
``_config_manager`` slot: it refuses ``None`` at the construction site and
names the agent that is missing one. A direct ``agent._config_manager = ...``
reaches around that check, so a ``None`` is accepted in silence and surfaces
later as a request-time failure in another module.

Two shapes are offences, and both are found structurally rather than by name:

* a write to another object's slot (``agent._config_manager = ...``) — the
  seam the accessor exists to remove;
* a ``self._config_manager = ...`` inside a class that inherits the accessor —
  the same bypass, from inside.

A class that has nothing to do with the accessor and keeps its own manager
under that private name owns its attribute; it is not an offence. The class
hierarchy is resolved from the sources themselves, so a new agent base class
extends the guard without editing it.

The detectors run on synthetic sources below as well as on the shipped tree: a
repo-wide "no offenders remain" assertion cannot protect its own detector.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBS = REPO_ROOT / "libs"

SLOT = "_config_manager"
ACCESSOR_ROOT = "ConfigManagerAware"
ACCESSOR_MODULE = "libs/core/cogniverse_core/agents/base.py"
ACCESSOR_METHOD = "bind_config_manager"

CROSS_OBJECT = (
    "writes another object's {slot}; call {receiver}.bind_config_manager(...)"
)
SELF_BYPASS = (
    "{cls} inherits the accessor and assigns self.{slot}; "
    "call self.bind_config_manager(...)"
)

# Every class that reaches the slot through the accessor. Agents are added to
# the product often; one that misses the accessor misses the guard with it.
ACCESSOR_HIERARCHY = frozenset(
    {
        "A2AAgent",
        "AgentBase",
        "AudioAnalysisAgent",
        "AuditExplanationAgent",
        "CitationTracingAgent",
        "CodingAgent",
        "ConfigManagerAware",
        "ContradictionReconciliationAgent",
        "CrossTenantComparisonAgent",
        "DeepResearchAgent",
        "DetailedReportAgent",
        "DocumentAgent",
        "EntityExtractionAgent",
        "FederatedQueryAgent",
        "GatewayAgent",
        "ImageSearchAgent",
        "KnowledgeGraphTraversalAgent",
        "KnowledgeSummarizationAgent",
        "MemoryAwareMixin",
        "MultiDocumentSynthesisAgent",
        "OrchestratorAgent",
        "ProfileSelectionAgent",
        "QueryEnhancementAgent",
        "RLMAwareMixin",
        "SearchAgent",
        "SummarizerAgent",
        "TemporalReasoningAgent",
        "TenantAwareAgentMixin",
        "TextAnalysisAgent",
    }
)

# Classes outside the hierarchy that keep their own manager under the same
# private name. Each owns the attribute it writes in its own ``__init__``.
SLOT_OWNING_NON_AGENTS = {
    ("libs/agents/cogniverse_agents/graph/claim_extractor.py", "ClaimExtractor"),
    ("libs/agents/cogniverse_agents/wiki/wiki_manager.py", "WikiManager"),
    ("libs/core/cogniverse_core/registries/schema_registry.py", "DeployedSchemaNames"),
    ("libs/core/cogniverse_core/registries/schema_registry.py", "SchemaRegistry"),
    ("libs/foundation/cogniverse_foundation/config/api_mixin.py", "ConfigAPIMixin"),
    (
        "libs/foundation/cogniverse_foundation/config/tenant_tiers.py",
        "TenantRouterTiers",
    ),
    ("libs/foundation/cogniverse_foundation/config/utils.py", "ConfigUtils"),
    ("libs/runtime/cogniverse_runtime/agent_dispatcher.py", "AgentDispatcher"),
    (
        "libs/runtime/cogniverse_runtime/ingestion/pipeline_builder.py",
        "VideoIngestionPipelineBuilder",
    ),
    ("libs/vespa/cogniverse_vespa/search_backend.py", "VespaSearchBackend"),
}


def _base_name(node: ast.expr) -> str | None:
    """The class name a base expression names, generic subscript unwrapped."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return None


def accessor_subclasses(sources: Mapping[str, str]) -> frozenset[str]:
    """Every class that inherits the accessor, the accessor itself included."""
    bases: dict[str, set[str]] = {}
    for source in sources.values():
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            named = {
                name
                for name in (_base_name(base) for base in node.bases)
                if name is not None
            }
            bases.setdefault(node.name, set()).update(named)

    resolved = {ACCESSOR_ROOT}
    while True:
        grown = {
            name
            for name, parents in bases.items()
            if name not in resolved and parents & resolved
        }
        if not grown:
            return frozenset(resolved)
        resolved |= grown


def _receiver(target: ast.expr) -> str | None:
    """Source text of the object being written, for an attribute target."""
    if not isinstance(target, ast.Attribute) or target.attr != SLOT:
        return None
    return ast.unparse(target.value)


def _setattr_receiver(call: ast.Call) -> str | None:
    """Source text of the object a ``setattr(obj, "_config_manager", ...)`` writes."""
    named = call.func.id if isinstance(call.func, ast.Name) else None
    if named != "setattr" or len(call.args) < 2:
        return None
    slot = call.args[1]
    if not (isinstance(slot, ast.Constant) and slot.value == SLOT):
        return None
    return ast.unparse(call.args[0])


def slot_writes(module_rel: str, source: str, subclasses: frozenset[str]) -> list[str]:
    """``file:line: reason`` for every write that bypasses the accessor."""
    findings: list[str] = []

    def walk(node: ast.AST, cls: str | None, func: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, child.name, None)
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, cls, child.name)
                continue
            receivers: list[str] = []
            if isinstance(child, ast.Assign):
                receivers = [_receiver(target) for target in child.targets]
            elif isinstance(child, ast.AnnAssign):
                receivers = [_receiver(child.target)]
            elif isinstance(child, ast.Call):
                receivers = [_setattr_receiver(child)]
            for receiver in receivers:
                if receiver is None:
                    continue
                if receiver not in {"self", "cls"}:
                    findings.append(
                        f"{module_rel}:{child.lineno}: "
                        + CROSS_OBJECT.format(slot=SLOT, receiver=receiver)
                    )
                elif cls in subclasses and not (
                    module_rel == ACCESSOR_MODULE and func == ACCESSOR_METHOD
                ):
                    findings.append(
                        f"{module_rel}:{child.lineno}: "
                        + SELF_BYPASS.format(cls=cls, slot=SLOT)
                    )
            walk(child, cls, func)

    walk(ast.parse(source), None, None)
    return findings


@pytest.fixture(scope="module")
def library_sources() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): path.read_text(errors="replace")
        for path in sorted(LIBS.rglob("*.py"))
    }


# --------------------------------------------------------------------------
# the shipped tree
# --------------------------------------------------------------------------


class TestShippedLibraries:
    def test_the_accessor_is_the_only_writer(self, library_sources):
        subclasses = accessor_subclasses(library_sources)
        findings = [
            finding
            for module_rel, source in library_sources.items()
            for finding in slot_writes(module_rel, source, subclasses)
        ]
        assert findings == []

    def test_the_hierarchy_the_guard_covers_is_every_agent_class(self, library_sources):
        """A class added here without inheriting the accessor drops out of the
        guard, so the set is written out rather than sampled."""
        assert accessor_subclasses(library_sources) == ACCESSOR_HIERARCHY

    def test_classes_outside_the_hierarchy_keep_their_own_slot(self, library_sources):
        """Non-agent owners of the same private name are not offenders, and the
        guard knows the difference — otherwise the shipped-tree assertion above
        would be green for the wrong reason."""
        subclasses = accessor_subclasses(library_sources)
        owners = {
            (module_rel, node.name)
            for module_rel, source in library_sources.items()
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ClassDef)
            and node.name not in subclasses
            and any(
                _receiver(target) == "self"
                for inner in ast.walk(node)
                if isinstance(inner, ast.Assign)
                for target in inner.targets
            )
        }
        assert owners == SLOT_OWNING_NON_AGENTS


# --------------------------------------------------------------------------
# the detectors, on synthetic input
# --------------------------------------------------------------------------

_SYNTHETIC = "libs/synthetic/offender.py"

_CROSS_OBJECT_SOURCE = (
    "def build(config_manager):\n"
    "    agent = Agent()\n"
    "    agent._config_manager = config_manager\n"
    "    return agent\n"
)
_SELF_BYPASS_SOURCE = (
    "class SyntheticAgent(MemoryAwareMixin):\n"
    "    def __init__(self, config_manager):\n"
    "        self._config_manager = config_manager\n"
)
_SETATTR_SOURCE = (
    "def build(agent, config_manager):\n"
    '    setattr(agent, "_config_manager", config_manager)\n'
)
_OWN_SLOT_SOURCE = (
    "class SyntheticRegistry:\n"
    "    def __init__(self, config_manager):\n"
    "        self._config_manager = config_manager\n"
)
_ACCESSOR_SOURCE = (
    "class ConfigManagerAware:\n"
    "    _config_manager = None\n"
    "    def bind_config_manager(self, config_manager):\n"
    "        self._config_manager = require_config_manager(config_manager)\n"
)
_GENERIC_BASE_SOURCE = (
    "class SyntheticTypedAgent(AgentBase[SyntheticInput, SyntheticOutput]):\n"
    "    def __init__(self, config_manager):\n"
    "        self._config_manager = config_manager\n"
)
_HIERARCHY = accessor_subclasses(
    {
        "accessor": _ACCESSOR_SOURCE,
        "mixin": "class MemoryAwareMixin(ConfigManagerAware):\n    pass\n",
        "typed": "class AgentBase(ConfigManagerAware):\n    pass\n",
        _SYNTHETIC: _SELF_BYPASS_SOURCE,
        "generic": _GENERIC_BASE_SOURCE,
    }
)


class TestDetectorOnSyntheticInput:
    def test_hierarchy_resolves_through_intermediate_and_generic_bases(self):
        assert _HIERARCHY == frozenset(
            {
                ACCESSOR_ROOT,
                "AgentBase",
                "MemoryAwareMixin",
                "SyntheticAgent",
                "SyntheticTypedAgent",
            }
        )

    def test_self_write_under_a_generic_base_is_reported_with_its_line(self):
        assert slot_writes(_SYNTHETIC, _GENERIC_BASE_SOURCE, _HIERARCHY) == [
            f"{_SYNTHETIC}:3: SyntheticTypedAgent inherits the accessor and "
            "assigns self._config_manager; call self.bind_config_manager(...)"
        ]

    def test_cross_object_write_is_reported_with_its_line(self):
        assert slot_writes(_SYNTHETIC, _CROSS_OBJECT_SOURCE, _HIERARCHY) == [
            f"{_SYNTHETIC}:3: writes another object's _config_manager; "
            "call agent.bind_config_manager(...)"
        ]

    def test_self_write_inside_the_hierarchy_is_reported_with_its_line(self):
        assert slot_writes(_SYNTHETIC, _SELF_BYPASS_SOURCE, _HIERARCHY) == [
            f"{_SYNTHETIC}:3: SyntheticAgent inherits the accessor and assigns "
            "self._config_manager; call self.bind_config_manager(...)"
        ]

    def test_dynamic_setattr_write_is_reported_with_its_line(self):
        """The name-based form is not the only way to reach the slot."""
        assert slot_writes(_SYNTHETIC, _SETATTR_SOURCE, _HIERARCHY) == [
            f"{_SYNTHETIC}:2: writes another object's _config_manager; "
            "call agent.bind_config_manager(...)"
        ]

    def test_self_write_outside_the_hierarchy_is_not_reported(self):
        assert slot_writes(_SYNTHETIC, _OWN_SLOT_SOURCE, _HIERARCHY) == []

    def test_the_accessor_body_is_the_one_exempt_write(self):
        assert (
            slot_writes(ACCESSOR_MODULE, _ACCESSOR_SOURCE, _HIERARCHY),
            slot_writes(_SYNTHETIC, _ACCESSOR_SOURCE, _HIERARCHY),
        ) == (
            [],
            [
                f"{_SYNTHETIC}:4: ConfigManagerAware inherits the accessor and "
                "assigns self._config_manager; call self.bind_config_manager(...)"
            ],
        )
