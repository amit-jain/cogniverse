# Code Style Guide

Coding standards and conventions for the Cogniverse codebase.

---

## Table of Contents

1. [General Principles](#general-principles)
2. [Python Style](#python-style)
3. [Type Annotations](#type-annotations)
4. [Naming Conventions](#naming-conventions)
5. [Import Organization](#import-organization)
6. [Documentation](#documentation)
7. [Error Handling](#error-handling)
8. [Testing](#testing)
9. [Tools](#tools)

---

## General Principles

1. **Clarity over cleverness**: Write code that's easy to understand
2. **Explicit over implicit**: Make dependencies and behavior visible
3. **Consistency**: Follow existing patterns in the codebase
4. **Minimal changes**: Only modify what's necessary for the task
5. **No premature abstraction**: Wait until patterns emerge before abstracting

---

## Python Style

### Formatting

We use **Ruff** for formatting and linting. Run before every commit:

```bash
# Format code
uv run ruff format .

# Lint and auto-fix
uv run ruff check --fix .

# Check without fixing
uv run ruff check .
```

### Line Length

- **Maximum**: 88 characters (Ruff default)
- **Docstrings**: 79 characters

### Quotes

- **Strings**: Double quotes (`"hello"`)
- **Docstrings**: Triple double quotes (`"""Docstring."""`)

### Trailing Commas

Use trailing commas in multi-line structures:

```python
# Good
config = {
    "tenant_id": "acme",
    "profile": "default",
}

# Bad
config = {
    "tenant_id": "acme",
    "profile": "default"
}
```

---

## Type Annotations

### Required Everywhere

All function signatures must have type annotations:

```python
# Good
def search(query: str, top_k: int = 10) -> list[SearchResult]:
    ...

# Bad
def search(query, top_k=10):
    ...
```

### Generic Types

Use Python 3.10+ syntax:

```python
# Good
def process(items: list[str]) -> dict[str, int]:
    ...

# Avoid (old style)
from typing import List, Dict
def process(items: List[str]) -> Dict[str, int]:
    ...
```

### Optional vs Union

Use `|` for unions:

```python
# Good
def get_config(name: str) -> Config | None:
    ...

# Avoid (old style)
from typing import Optional
def get_config(name: str) -> Optional[Config]:
    ...
```

### Type Aliases

Define complex types as aliases:

```python
# Good
SearchResults = list[dict[str, Any]]

def search(query: str) -> SearchResults:
    ...
```

### Run Type Checking

```bash
uv run mypy libs/
```

---

## Naming Conventions

### Variables and Functions

- **snake_case** for variables and functions
- **Descriptive names**: `user_config` not `uc`
- **Verbs for functions**: `get_config()`, `process_query()`, `validate_input()`

```python
# Good
def get_tenant_config(tenant_id: str) -> TenantConfig:
    tenant_config = load_config(tenant_id)
    return tenant_config

# Bad
def config(t):
    c = load(t)
    return c
```

### Classes

- **PascalCase** for class names
- **Nouns or noun phrases**: `SearchAgent`, `ConfigManager`

```python
# Good
class VideoSearchAgent(A2AAgent):
    ...

# Bad
class video_search_agent(A2AAgent):  # Wrong case
    ...
```

### Constants

- **UPPER_SNAKE_CASE** for module-level constants

```python
# Good
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3

# Bad
defaultTimeout = 30
```

### Private Members

- **Single underscore** for internal use: `_internal_method()`
- **Double underscore** only for name mangling (rare)

```python
class Agent:
    def process(self, input: Input) -> Output:
        """Public API."""
        return self._generate_response(input)

    def _generate_response(self, input: Input) -> Output:
        """Internal implementation."""
        ...
```

### File Names

- **snake_case** for Python files: `search_agent.py`
- **No prefixes**: Avoid `base_`, `simple_`, `final_`, `v2_`
- **Module structure**: `package/subpackage/module.py`

---

## Import Organization

### Order

1. Standard library
2. Third-party packages
3. Local packages (cogniverse_*)

```python
# Standard library
import asyncio
import logging
from pathlib import Path
from typing import Any

# Third-party
import dspy
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Local packages
from cogniverse_core.agents.base import AgentBase, AgentInput, AgentOutput
from cogniverse_foundation.config.manager import ConfigManager
```

### Absolute vs Relative

- **Absolute imports** for cross-package: `from cogniverse_core.agents import AgentBase`
- **Relative imports** within package: `from .types import AgentInput`

```python
# In cogniverse_agents/routing/agent.py

# Cross-package (absolute)
from cogniverse_core.agents.a2a_agent import A2AAgent

# Within package (relative)
from .types import RoutingInput, RoutingOutput
from .module import RoutingModule
```

### Avoid Star Imports

```python
# Good
from cogniverse_core.agents.base import AgentBase, AgentInput, AgentOutput

# Bad
from cogniverse_core.agents.base import *
```

---

## Documentation

### Module Docstrings

Every module should have a docstring:

```python
"""
Search Agent Implementation

Provides multi-modal search with profile-based configuration
and tenant isolation.
"""
```

### Function Docstrings

Use Google style:

```python
def search(
    query: str,
    profile: str = "default",
    top_k: int = 10
) -> list[SearchResult]:
    """
    Execute a search query.

    Args:
        query: The search query string
        profile: Backend profile to use
        top_k: Number of results to return

    Returns:
        List of search results ordered by relevance

    Raises:
        ValueError: If query is empty
        BackendError: If backend connection fails
    """
```

### Class Docstrings

```python
class SearchAgent(A2AAgent[SearchInput, SearchOutput, SearchDeps]):
    """
    Agent for multi-modal search.

    Supports semantic, hybrid, and learned ranking strategies.
    Integrates with Vespa backend for vector search.

    Attributes:
        AGENT_NAME: Unique identifier for registration
        CAPABILITIES: List of supported operations

    Example:
        agent = SearchAgent(deps=deps)
        result = await agent.process(SearchInput(query="hello"))
    """
```

### When NOT to Comment

- Don't add comments for self-explanatory code
- Don't add type annotations to code you didn't change
- Don't add docstrings to obvious methods

```python
# Bad - unnecessary comment
# Get the user's name
name = user.name

# Good - no comment needed
name = user.name
```

---

## Error Handling

### Be Specific

Catch specific exceptions, not bare `except`:

```python
# Good
try:
    result = await backend.search(query)
except ConnectionError as e:
    logger.error(f"Backend connection failed: {e}")
    raise BackendError(f"Search failed: {e}") from e

# Bad
try:
    result = await backend.search(query)
except:
    pass
```

### Use Custom Exceptions

Define domain-specific exceptions:

```python
# Good
class BackendError(Exception):
    """Error communicating with backend."""

class ConfigurationError(Exception):
    """Invalid configuration."""

# Raise with context
raise BackendError(f"Failed to connect to {url}")
```

### Return vs Raise

- **Raise** for unexpected errors that should stop execution
- **Return error output** for expected failure modes

```python
# Agent process - return error in output
async def process(self, input: Input) -> Output:
    if not input.query:
        return Output(result=None, error="Query cannot be empty")
    ...

# Utility function - raise exception
def validate_config(config: dict) -> None:
    if "tenant_id" not in config:
        raise ConfigurationError("tenant_id is required")
```

---

## Testing

### Test Organization

```
tests/
├── agents/
│   ├── unit/
│   │   ├── test_routing_agent.py
│   │   └── test_search_agent.py
│   └── integration/
│       └── test_agent_orchestration.py
├── foundation/
│   ├── unit/
│   │   └── test_config_manager.py
│   └── integration/
│       └── test_telemetry.py
└── conftest.py  # Shared fixtures
```

### Test Naming

```python
class TestSearchAgent:
    """Tests for SearchAgent."""

    def test_process_returns_results(self):
        """Test that process returns search results."""
        ...

    def test_empty_query_returns_error(self):
        """Test that empty query returns error output."""
        ...

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        ...
```

### Fixtures

Use pytest fixtures for common setup:

```python
# conftest.py
@pytest.fixture
def config_manager():
    """Create test config manager."""
    return ConfigManager(store=InMemoryStore())

@pytest.fixture
def search_agent(config_manager):
    """Create test search agent."""
    deps = SearchDeps(config_manager=config_manager)
    return SearchAgent(deps=deps)
```

### Assertions

Be specific in assertions:

```python
# Good
assert result.summary == "Expected summary"
assert len(result.key_points) == 3
assert result.confidence > 0.8

# Bad
assert result  # Too vague
assert result is not None  # Still vague
```

---

## Tools

### Pre-commit Workflow

```bash
# 1. Format
uv run ruff format .

# 2. Lint
uv run ruff check --fix .

# 3. Type check
uv run mypy libs/

# 4. Test
JAX_PLATFORM_NAME=cpu uv run pytest tests/ -v

# 5. All at once
uv run make lint-all
```

### Editor Configuration

**.vscode/settings.json**:
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.formatting.provider": "none",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true
    },
    "ruff.lint.run": "onSave"
}
```

### Ruff Configuration

**pyproject.toml**:
```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "C4"]
ignore = ["E501"]  # Line length handled by formatter

[tool.ruff.lint.isort]
known-first-party = ["cogniverse_*"]
```

---

## Summary

1. Use Ruff for formatting and linting
2. Type annotate all function signatures
3. Use snake_case for variables/functions, PascalCase for classes
4. Organize imports: stdlib → third-party → local
5. Write Google-style docstrings for public APIs
6. Catch specific exceptions, raise with context
7. Run `uv run make lint-all` before every commit
