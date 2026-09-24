"""
Custom exceptions for registry operations.

These exceptions provide clear error types for different failure modes
in schema deployment and registry operations.
"""


class SchemaDeploymentError(Exception):
    """
    Base exception for schema deployment failures.

    All schema deployment errors inherit from this base exception,
    allowing callers to catch all deployment-related errors easily.
    """

    pass


class SchemaLoadError(SchemaDeploymentError):
    """The base schema definition could not be loaded.

    Chained from the loader's own error so callers can tell a missing
    schema file (permanent) from the schema store being unreachable
    (transient)."""


class BackendDeploymentError(SchemaDeploymentError):
    """
    Backend failed to deploy schemas.

    Raised when the backend's deploy_schemas() method fails. This indicates
    a failure in the underlying storage system (connection error, validation
    error, resource constraint, etc.).

    This error occurs BEFORE ConfigStore registration, so state is consistent.
    """

    pass


class SchemaConvergenceError(BackendDeploymentError):
    """The config server activated the package, but the generation did not
    reach every service, or a new schema refused a feed, inside the budget.

    The schema is live in Vespa; only its registration is still owed.
    """

    def __init__(self, message: str, *, generation: int) -> None:
        super().__init__(message)
        self.generation = generation


class RegistryStorageError(SchemaDeploymentError):
    """
    ConfigStore failed to register schema.

    Raised when ConfigStore operations fail (database write, connection timeout,
    disk full, etc.). This error occurs AFTER backend deployment succeeds,
    requiring rollback to maintain consistency.
    """

    pass


class RegistryConflictError(RegistryStorageError):
    """A conditional registration found a newer registry revision.

    Another process registered or tombstoned the schema since the caller read
    it; that revision is authoritative and must not be rolled back over.
    """


class SchemaRevisionConflictError(SchemaDeploymentError):
    """A peer's registry revision superseded the one a deploy was decided from.

    ``peer_revision`` is ``"tombstone"`` when the peer deleted the schema and
    ``"registration"`` when it registered it again. Raised before activation
    when the deploy lease finds the revision moved, so nothing was activated
    or registered; raised after activation when the conditional registration
    finds it moved, so the activation stands and the peer's revision was kept.
    Either way the peer's revision is authoritative and the deploy may simply
    be retried.
    """

    retryable = True

    def __init__(self, schema_name: str, peer_revision: str, *, activated: bool):
        self.schema_name = schema_name
        self.peer_revision = peer_revision
        self.activated = activated
        action = "deleted" if peer_revision == "tombstone" else "re-registered"
        outcome = (
            "the activation stands and that revision was not overwritten"
            if activated
            else "nothing was activated or registered"
        )
        super().__init__(
            f"Schema {schema_name!r} was {action} by another process after this "
            f"deploy read its registry row; {outcome}. Retry the deploy."
        )


class SchemaRegistryInitializationError(Exception):
    """
    SchemaRegistry failed to initialize.

    Raised during SchemaRegistry construction when critical initialization
    steps fail after bounded storage retries.

    Empty storage is valid and loads as an empty registry.
    """

    pass
