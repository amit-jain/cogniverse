"""
Adapter Storage Abstraction

Handles uploading and downloading adapter files to/from various storage backends.
Supports local filesystem, Hugging Face Hub, S3-compatible object stores, and
Modal volumes.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class AdapterStorage(ABC):
    """
    Abstract interface for adapter storage backends.

    Implementations:
    - LocalStorage: Local filesystem (file://)
    - HuggingFaceStorage: Hugging Face Hub (hf://)
    - S3Storage: AWS S3 or S3-compatible object storage (s3://)
    - ModalVolumeStorage: Modal persistent volumes (modal://)
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


@dataclass(frozen=True, slots=True)
class S3StorageConfig:
    """Connection settings for S3-compatible adapter storage."""

    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    region: Optional[str] = None


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

        logger.info(
            f"Downloaded adapter from hf://{repo_id}/{revision} to {local_path}"
        )
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


class S3Storage(AdapterStorage):
    """
    S3-compatible storage for adapter directories.

    URIs: s3://bucket/path/to/adapter

    Connection settings come from ``S3StorageConfig``. The bucket comes from
    the URI netloc and the object prefix comes from the URI path. A directory
    upload stores files recursively under that prefix.
    """

    def __init__(self, config: S3StorageConfig):
        """Initialize S3 storage with explicit connection settings."""
        self.config = config

    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme != "s3":
            raise ValueError(f"Invalid S3 URI: {uri}")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket:
            raise ValueError(f"S3 URI must include a bucket name: {uri}")
        if not key:
            raise ValueError(f"S3 URI must include an adapter path: {uri}")
        return bucket, key

    @staticmethod
    def _not_found_code(exc) -> bool:
        from botocore.exceptions import ClientError

        if not isinstance(exc, ClientError):
            return False
        code = exc.response.get("Error", {}).get("Code")
        return code in {"404", "NoSuchKey", "NotFound", "NoSuchBucket"}

    def _client(self):
        import boto3
        from botocore.config import Config

        client_kwargs = {
            "region_name": self.config.region or "us-east-1",
            "config": Config(signature_version="s3v4"),
        }
        endpoint_url = self.config.endpoint_url
        access_key = self.config.access_key
        secret_key = self.config.secret_key
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if access_key:
            client_kwargs["aws_access_key_id"] = access_key
        if secret_key:
            client_kwargs["aws_secret_access_key"] = secret_key

        return boto3.client("s3", **client_kwargs)

    @staticmethod
    def _bucket_uri(bucket: str, key: str) -> str:
        return f"s3://{bucket}/{key.lstrip('/')}"

    def upload(self, local_path: str, destination_uri: str) -> str:
        source = Path(local_path)
        if not source.exists():
            raise FileNotFoundError(f"Source adapter not found: {local_path}")

        bucket, key = self._parse_uri(destination_uri)
        client = self._client()

        try:
            if source.is_dir():
                prefix = key.rstrip("/")
                if not prefix:
                    raise ValueError(
                        f"S3 destination URI must include an object prefix: {destination_uri}"
                    )
                uploaded = False
                for file_path in sorted(source.rglob("*")):
                    if not file_path.is_file():
                        continue
                    rel_path = file_path.relative_to(source).as_posix()
                    object_key = f"{prefix}/{rel_path}"
                    client.upload_file(str(file_path), bucket, object_key)
                    uploaded = True
                if not uploaded:
                    raise ValueError(f"Adapter directory is empty: {local_path}")
                return self._bucket_uri(bucket, prefix)

            object_key = (
                key.rstrip("/")
                if not key.endswith("/")
                else f"{key.rstrip('/')}/{source.name}"
            )
            if not object_key:
                raise ValueError(
                    f"S3 destination URI must include an object key: {destination_uri}"
                )
            client.upload_file(str(source), bucket, object_key)
            return self._bucket_uri(bucket, object_key)
        except Exception as exc:
            raise RuntimeError(
                f"failed to upload adapter to {destination_uri}: {exc}"
            ) from exc

    def download(self, source_uri: str, local_path: str) -> str:
        bucket, key = self._parse_uri(source_uri)
        client = self._client()
        dest = Path(local_path)
        prefix = key.rstrip("/")
        dest_parent = dest.parent
        dest_name = dest.name or "adapter"
        dest_parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{dest_name}.staging-",
                dir=str(dest_parent),
            )
        )
        backup_dir: Path | None = None

        try:
            paginator = client.get_paginator("list_objects_v2")
            matching_keys: list[str] = []
            for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
                matching_keys.extend(
                    obj["Key"] for obj in page.get("Contents", []) or []
                )

            if matching_keys:
                for object_key in matching_keys:
                    relative_key = object_key[len(prefix) + 1 :]
                    target = staging_dir / relative_key
                    target.parent.mkdir(parents=True, exist_ok=True)
                    client.download_file(bucket, object_key, str(target))
            else:
                target = staging_dir / Path(prefix).name
                client.download_file(bucket, prefix, str(target))

            if dest.exists():
                backup_dir = dest_parent / f".{dest_name}.backup-{uuid.uuid4().hex}"
                os.replace(dest, backup_dir)

            try:
                os.replace(staging_dir, dest)
            except Exception:
                if backup_dir is not None:
                    os.replace(backup_dir, dest)
                    backup_dir = None
                raise

            if backup_dir is not None:
                if backup_dir.is_dir():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                else:
                    backup_dir.unlink(missing_ok=True)

            return str(dest)
        except Exception as exc:
            from botocore.exceptions import ClientError

            if isinstance(exc, ClientError) and self._not_found_code(exc):
                raise FileNotFoundError(
                    f"Source adapter not found: {source_uri}"
                ) from exc
            raise RuntimeError(
                f"failed to download adapter from {source_uri}: {exc}"
            ) from exc
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def exists(self, uri: str) -> bool:
        bucket, key = self._parse_uri(uri)
        client = self._client()

        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception as exc:
            if not self._not_found_code(exc):
                raise RuntimeError(f"failed to check adapter at {uri}: {exc}") from exc

        try:
            response = client.list_objects_v2(
                Bucket=bucket,
                Prefix=f"{key.rstrip('/')}/",
                MaxKeys=1,
            )
        except Exception as exc:
            raise RuntimeError(f"failed to check adapter at {uri}: {exc}") from exc

        return bool(response.get("Contents"))


class ModalVolumeStorage(AdapterStorage):
    """
    Modal volume storage for adapter directories.

    URIs: modal://volume-name/path/to/adapter
    """

    def __init__(
        self,
        volume_name: str,
        volume_path: str = "",
        *,
        environment_name: Optional[str] = None,
        create_if_missing: bool = False,
        volume=None,
    ):
        self.volume_name = volume_name
        self.volume_path = volume_path.strip("/")
        self.environment_name = environment_name
        self.create_if_missing = create_if_missing
        self._volume = volume

    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme != "modal":
            raise ValueError(f"Invalid Modal URI: {uri}")

        volume_name = parsed.netloc
        volume_path = parsed.path.lstrip("/")
        if not volume_name:
            raise ValueError(f"Modal URI must include a volume name: {uri}")
        if not volume_path:
            raise ValueError(f"Modal URI must include an adapter path: {uri}")
        return volume_name, volume_path

    def _resolve_volume(self):
        if self._volume is not None:
            return self._volume

        import modal

        return modal.Volume.from_name(
            self.volume_name,
            environment_name=self.environment_name,
            create_if_missing=self.create_if_missing,
        )

    def _volume_root(self) -> str:
        return f"/{self.volume_path}" if self.volume_path else "/"

    def _canonical_uri(self) -> str:
        path = self.volume_path.strip("/")
        return (
            f"modal://{self.volume_name}/{path}"
            if path
            else f"modal://{self.volume_name}"
        )

    @staticmethod
    def _local_root(local_path: str) -> Path:
        return Path(local_path)

    async def _upload_async(self, local_path: str, destination_uri: str) -> str:
        source = self._local_root(local_path)
        if not source.exists():
            raise FileNotFoundError(f"Source adapter not found: {local_path}")

        volume = self._resolve_volume()
        remote_root = self._volume_root()

        try:
            async with volume.batch_upload(force=True) as batch:
                if source.is_dir():
                    await asyncio.to_thread(
                        batch.put_directory, str(source), remote_root
                    )
                else:
                    remote_path = remote_root.rstrip("/") or "/"
                    if remote_path == "/":
                        remote_path = f"/{source.name}"
                    else:
                        remote_path = f"{remote_path}/{source.name}"
                    await asyncio.to_thread(batch.put_file, str(source), remote_path)
        except Exception as exc:
            raise RuntimeError(
                f"failed to upload adapter to {destination_uri}: {exc}"
            ) from exc

        return self._canonical_uri()

    async def _download_async(self, source_uri: str, local_path: str) -> str:
        volume = self._resolve_volume()
        remote_root = self._volume_root()
        dest = Path(local_path)

        try:
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            dest.mkdir(parents=True, exist_ok=True)

            entries = await asyncio.to_thread(
                volume.listdir, remote_root, recursive=True
            )
            file_entries = [
                entry
                for entry in entries
                if getattr(entry.type, "name", None) == "FILE"
            ]
            if not file_entries:
                raise FileNotFoundError(f"Source adapter not found: {source_uri}")

            remote_base = remote_root.lstrip("/")
            for entry in file_entries:
                entry_path = entry.path.lstrip("/")
                if remote_base and entry_path == remote_base:
                    relative_path = Path(entry_path).name
                elif remote_base and entry_path.startswith(f"{remote_base}/"):
                    relative_path = entry_path[len(remote_base) + 1 :]
                else:
                    relative_path = entry_path

                target = dest / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as fh:
                    await asyncio.to_thread(
                        volume.read_file_into_fileobj, entry.path, fh
                    )
            return str(dest)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"failed to download adapter from {source_uri}: {exc}"
            ) from exc

    async def _exists_async(self, uri: str) -> bool:
        volume = self._resolve_volume()
        remote_root = self._volume_root()
        try:
            entries = await asyncio.to_thread(
                volume.listdir, remote_root, recursive=True
            )
        except FileNotFoundError:
            return False
        except Exception as exc:
            raise RuntimeError(f"failed to check adapter at {uri}: {exc}") from exc
        return bool(entries)

    @staticmethod
    def _run_async(coro):
        return asyncio.run(coro)

    def upload(self, local_path: str, destination_uri: str) -> str:
        return self._run_async(self._upload_async(local_path, destination_uri))

    def download(self, source_uri: str, local_path: str) -> str:
        return self._run_async(self._download_async(source_uri, local_path))

    def exists(self, uri: str) -> bool:
        return self._run_async(self._exists_async(uri))


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
        **kwargs: Additional arguments passed to storage backend (e.g., token for
            HF, endpoint_url/access_key/secret_key for S3, volume injection for
            Modal tests)

    Returns:
        AdapterStorage implementation

    Raises:
        ValueError: If URI scheme is not supported

    Supported schemes:
        - file:// or plain path: Local filesystem
        - hf://org/repo: HuggingFace Hub (recommended for production)
        - s3://bucket/path: S3-compatible object storage
        - modal://volume/path: Modal persistent volumes
    """
    parsed = urlparse(uri)
    scheme = parsed.scheme or "file"

    if scheme == "file" or not parsed.scheme:
        return LocalStorage()
    elif scheme == "hf":
        return HuggingFaceStorage(token=kwargs.get("token"))
    elif scheme == "s3":
        # Resolve S3 connection settings here so S3Storage stays config-only.
        S3Storage._parse_uri(uri)
        return S3Storage(
            S3StorageConfig(
                endpoint_url=kwargs.get("endpoint_url")
                or os.environ.get("MINIO_ENDPOINT")
                or os.environ.get("S3_ENDPOINT_URL"),
                access_key=kwargs.get("access_key")
                or os.environ.get("MINIO_ACCESS_KEY")
                or os.environ.get("AWS_ACCESS_KEY_ID"),
                secret_key=kwargs.get("secret_key")
                or os.environ.get("MINIO_SECRET_KEY")
                or os.environ.get("AWS_SECRET_ACCESS_KEY"),
                region=kwargs.get("region")
                or os.environ.get("AWS_DEFAULT_REGION")
                or os.environ.get("AWS_REGION"),
            )
        )
    elif scheme == "modal":
        volume_name, volume_path = ModalVolumeStorage._parse_uri(uri)
        return ModalVolumeStorage(
            volume_name=volume_name,
            volume_path=volume_path,
            environment_name=kwargs.get("environment_name"),
            create_if_missing=kwargs.get("create_if_missing", False),
            volume=kwargs.get("volume"),
        )
    else:
        raise ValueError(f"Unsupported storage scheme: {scheme}")


def upload_adapter(
    local_path: str, destination_uri: str, token: Optional[str] = None
) -> str:
    """
    Upload adapter to storage.

    Convenience function that selects the appropriate backend.

    Args:
        local_path: Local path to adapter directory
        destination_uri: Target URI
        token: Storage token (e.g. HuggingFace token for hf:// URIs). When
            None the backend falls back to env var / cached login.

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
    storage = get_storage_backend(destination_uri, token=token)
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
