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


class RegistryStorageError(SchemaDeploymentError):
    """
    ConfigStore failed to register schema.

    Raised when ConfigStore operations fail (database write, connection timeout,
    disk full, etc.). This error occurs AFTER backend deployment succeeds,
    requiring rollback to maintain consistency.
    """

    pass


class SchemaRegistryInitializationError(Exception):
    """
    SchemaRegistry failed to initialize.

    Raised during SchemaRegistry construction when critical initialization
    steps fail after bounded storage retries.

    Empty storage is valid and loads as an empty registry.
    """

    pass
