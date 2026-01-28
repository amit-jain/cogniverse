"""
Event Queue Backends

Available backends:
- memory: In-memory implementation for single-pod deployments and development
- (future) redis: Redis Pub/Sub for multi-pod production deployments
"""

from cogniverse_core.events.backends.memory import (
    InMemoryEventQueue,
    InMemoryQueueManager,
    get_queue_manager,
    reset_queue_manager,
)

__all__ = [
    "InMemoryEventQueue",
    "InMemoryQueueManager",
    "get_queue_manager",
    "reset_queue_manager",
]
