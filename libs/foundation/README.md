# Cogniverse Foundation

Cross-cutting concerns and shared infrastructure for the Cogniverse platform.

## Overview

This package provides foundational infrastructure that sits between the pure interfaces (`cogniverse-sdk`) and the core implementations. It contains cross-cutting concerns like configuration base classes and telemetry interface definitions.

## Key Components

### Configuration Base (`cogniverse_foundation.config`)

Base classes and utilities for configuration management:
- `BaseConfig`: Abstract configuration base class
- `ConfigValidator`: Configuration validation utilities
- Schema validation support

### Telemetry Interfaces (`cogniverse_foundation.telemetry`)

Provider-agnostic telemetry interface definitions:
- `TelemetryProvider`: Abstract telemetry provider interface
- `TelemetryConfig`: Telemetry configuration base
- OpenTelemetry integration points

## Installation

```bash
pip install cogniverse-foundation
```

## Dependencies

**Internal:**
- `cogniverse-sdk`: Pure backend interfaces

**External:**
- `opentelemetry-api>=1.20.0`: OpenTelemetry interfaces
- `opentelemetry-sdk>=1.20.0`: OpenTelemetry SDK
- `pydantic>=2.0.0`: Data validation
- `sqlalchemy>=2.0.0`: Database support
- `pandas>=2.0.0`: Data manipulation

## Usage

```python
from cogniverse_foundation.config import BaseConfig
from cogniverse_foundation.telemetry import TelemetryProvider

# Extend base configuration
class MyConfig(BaseConfig):
    service_name: str
    environment: str

# Implement telemetry provider
class MyTelemetryProvider(TelemetryProvider):
    def get_tracer(self, name: str):
        # Implementation
        pass
```

## Architecture Position

Foundation sits in the **Foundation Layer** of the Cogniverse architecture:

```
Foundation Layer:
  cogniverse-sdk (zero internal dependencies)
    ↓
  cogniverse-foundation ← YOU ARE HERE
    ↓
Core Layer:
  cogniverse-core, cogniverse-evaluation
```

## Development

```bash
# Install in editable mode
cd libs/foundation
pip install -e .

# Run tests
pytest tests/foundation/
```

## License

MIT
