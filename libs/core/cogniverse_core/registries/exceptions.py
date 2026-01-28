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
    steps fail (loading schemas from storage, validating state, etc.).

    In strict_mode=True, this exception is raised to fail fast.
    In strict_mode=False, initialization continues with empty registry.
    """

    pass
