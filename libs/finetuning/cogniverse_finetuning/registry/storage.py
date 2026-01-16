"""
Adapter Storage Abstraction

Handles uploading and downloading adapter files to/from various storage backends.
Supports local filesystem now, extensible for Modal volumes and cloud storage.
"""

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class AdapterStorage(ABC):
    """
    Abstract interface for adapter storage backends.

    Implementations:
    - LocalStorage: Local filesystem (file://)
    - ModalVolumeStorage: Modal persistent volumes (modal://) [future]
    - S3Storage: AWS S3 or S3-compatible (s3://) [future]
    """

    @abstractmethod
    def upload(self, local_path: str, destination_uri: str) -> str:
        """
        Upload adapter from local path to storage.

        Args:
            local_path: Local filesystem path to adapter directory
            destination_uri: Target URI (storage-specific)

        Returns:
            Final URI where adapter was stored
        """
        pass

    @abstractmethod
    def download(self, source_uri: str, local_path: str) -> str:
        """
        Download adapter from storage to local path.

        Args:
            source_uri: Source URI to download from
            local_path: Local filesystem path to download to

        Returns:
            Local path where adapter was downloaded
        """
        pass

    @abstractmethod
    def exists(self, uri: str) -> bool:
        """Check if adapter exists at URI."""
        pass


class HuggingFaceStorage(AdapterStorage):
    """
    Hugging Face Hub storage for adapters.

    URIs: hf://org/repo-name or hf://org/repo-name/revision

    Advantages:
    - Built-in versioning
    - Easy integration with transformers/PEFT/vLLM
    - Private repos for tenant isolation
    - No need to manage S3/storage infrastructure

    Example:
        >>> storage = HuggingFaceStorage(token="hf_xxx")
        >>> uri = storage.upload(
        ...     "/tmp/adapter",
        ...     "hf://myorg/routing-adapter-v1"
        ... )
        >>> # Later, download
        >>> path = storage.download(uri, "/tmp/cache/adapter")
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize HuggingFace storage.

        Args:
            token: HuggingFace API token. If None, uses HF_TOKEN env var or cached login.
        """
        self.token = token

    def upload(self, local_path: str, destination_uri: str) -> str:
        """
        Upload adapter to HuggingFace Hub.

        Creates or updates a model repo with the adapter files.
        """
        from huggingface_hub import HfApi

        # Parse URI: hf://org/repo-name or hf://org/repo-name/revision
        if not destination_uri.startswith("hf://"):
            raise ValueError(f"Invalid HuggingFace URI: {destination_uri}")

        parts = destination_uri[5:].split("/")  # Remove hf://
        if len(parts) < 2:
            raise ValueError(f"Invalid HuggingFace URI format: {destination_uri}")

        repo_id = f"{parts[0]}/{parts[1]}"
        revision = parts[2] if len(parts) > 2 else "main"

        api = HfApi(token=self.token)

        # Create repo if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create repo {repo_id}: {e}")

        # Upload folder
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            revision=revision,
            commit_message=f"Upload adapter from {local_path}",
        )

        logger.info(f"Uploaded adapter to hf://{repo_id}/{revision}")
        return f"hf://{repo_id}/{revision}"

    def download(self, source_uri: str, local_path: str) -> str:
        """
        Download adapter from HuggingFace Hub.
        """
        from huggingface_hub import snapshot_download

        # Parse URI
        if not source_uri.startswith("hf://"):
            raise ValueError(f"Invalid HuggingFace URI: {source_uri}")

        parts = source_uri[5:].split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid HuggingFace URI format: {source_uri}")

        repo_id = f"{parts[0]}/{parts[1]}"
        revision = parts[2] if len(parts) > 2 else "main"

        # Download to local path
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_path,
            token=self.token,
        )

        logger.info(f"Downloaded adapter from hf://{repo_id}/{revision} to {local_path}")
        return downloaded_path

    def exists(self, uri: str) -> bool:
        """Check if adapter exists on HuggingFace Hub."""
        from huggingface_hub import repo_exists

        if not uri.startswith("hf://"):
            return False

        parts = uri[5:].split("/")
        if len(parts) < 2:
            return False

        repo_id = f"{parts[0]}/{parts[1]}"

        try:
            return repo_exists(repo_id, token=self.token)
        except Exception:
            return False


class LocalStorage(AdapterStorage):
    """
    Local filesystem storage for adapters.

    URIs: file:///path/to/adapter or just /path/to/adapter
    """

    def upload(self, local_path: str, destination_uri: str) -> str:
        """
        Copy adapter to destination path.

        For local storage, this is essentially a copy operation.
        The destination_uri should be file:// or a plain path.
        """
        # Parse destination
        if destination_uri.startswith("file://"):
            dest_path = destination_uri[7:]  # Remove file://
        else:
            dest_path = destination_uri

        source = Path(local_path)
        dest = Path(dest_path)

        if not source.exists():
            raise FileNotFoundError(f"Source adapter not found: {local_path}")

        # If source and dest are the same, nothing to do
        if source.resolve() == dest.resolve():
            logger.info(f"Adapter already at destination: {dest_path}")
            return f"file://{dest.resolve()}"

        # Copy directory
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)

        logger.info(f"Copied adapter from {local_path} to {dest_path}")
        return f"file://{dest.resolve()}"

    def download(self, source_uri: str, local_path: str) -> str:
        """
        Copy adapter from source to local path.

        For local storage, this is essentially a copy operation.
        """
        # Parse source
        if source_uri.startswith("file://"):
            src_path = source_uri[7:]
        else:
            src_path = source_uri

        source = Path(src_path)
        dest = Path(local_path)

        if not source.exists():
            raise FileNotFoundError(f"Source adapter not found: {src_path}")

        # If source and dest are the same, nothing to do
        if source.resolve() == dest.resolve():
            return local_path

        # Copy directory
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)

        logger.info(f"Copied adapter from {src_path} to {local_path}")
        return local_path

    def exists(self, uri: str) -> bool:
        """Check if adapter exists at path."""
        if uri.startswith("file://"):
            path = uri[7:]
        else:
            path = uri
        return Path(path).exists()


def get_storage_backend(uri: str, **kwargs) -> AdapterStorage:
    """
    Get appropriate storage backend for a URI.

    Args:
        uri: Storage URI (file://, hf://, s3://, etc.)
        **kwargs: Additional arguments passed to storage backend (e.g., token for HF)

    Returns:
        AdapterStorage implementation

    Raises:
        ValueError: If URI scheme is not supported

    Supported schemes:
        - file:// or plain path: Local filesystem
        - hf://org/repo: HuggingFace Hub (recommended for production)
        - s3://bucket/path: AWS S3 (not yet implemented)
        - modal://volume/path: Modal volumes (not yet implemented)
    """
    parsed = urlparse(uri)
    scheme = parsed.scheme or "file"

    if scheme == "file" or not parsed.scheme:
        return LocalStorage()
    elif scheme == "hf":
        return HuggingFaceStorage(token=kwargs.get("token"))
    elif scheme == "s3":
        raise NotImplementedError(
            "S3 storage not yet implemented. "
            "Add cogniverse_finetuning.registry.storage_s3 for S3 support."
        )
    elif scheme == "modal":
        raise NotImplementedError(
            "Modal volume storage not yet implemented. "
            "Add cogniverse_finetuning.registry.storage_modal for Modal support."
        )
    else:
        raise ValueError(f"Unsupported storage scheme: {scheme}")


def upload_adapter(local_path: str, destination_uri: str) -> str:
    """
    Upload adapter to storage.

    Convenience function that selects the appropriate backend.

    Args:
        local_path: Local path to adapter directory
        destination_uri: Target URI

    Returns:
        Final URI where adapter was stored

    Example:
        >>> uri = upload_adapter(
        ...     "/tmp/adapter",
        ...     "file:///data/adapters/routing_sft_v1"
        ... )
        >>> print(uri)
        file:///data/adapters/routing_sft_v1
    """
    storage = get_storage_backend(destination_uri)
    return storage.upload(local_path, destination_uri)


def download_adapter(source_uri: str, local_path: str) -> str:
    """
    Download adapter from storage.

    Convenience function that selects the appropriate backend.

    Args:
        source_uri: Source URI to download from
        local_path: Local path to download to

    Returns:
        Local path where adapter was downloaded

    Example:
        >>> path = download_adapter(
        ...     "file:///data/adapters/routing_sft_v1",
        ...     "/tmp/adapter"
        ... )
    """
    storage = get_storage_backend(source_uri)
    return storage.download(source_uri, local_path)


def adapter_exists(uri: str) -> bool:
    """
    Check if adapter exists at URI.

    Args:
        uri: Storage URI

    Returns:
        True if adapter exists
    """
    try:
        storage = get_storage_backend(uri)
        return storage.exists(uri)
    except (NotImplementedError, ValueError):
        return False
