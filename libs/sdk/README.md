# Cogniverse SDK

Core interfaces for Cogniverse backend implementations.

## Overview

This package provides the foundational interfaces that all backend implementations must satisfy:

- **Backend**: Document/vector storage interface (search + ingestion)
- **ConfigStore**: Configuration storage interface
- **SchemaLoader**: Schema template loading interface
- **Document**: Universal document model

## Installation

```bash
pip install cogniverse-sdk
```

## Usage

```python
from cogniverse_sdk.interfaces.backend import Backend
from cogniverse_sdk.document import Document, ContentType

# Implement backend interface
class MyBackend(Backend):
    def _initialize_backend(self, config):
        # Implementation
        pass
```

## Dependencies

This package has zero dependencies on other Cogniverse packages, making it the foundation layer.

## License

MIT
