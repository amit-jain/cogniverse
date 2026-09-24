"""Validate the wheel and sdist built for each release package and write the
build manifest describing exactly those artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

from packaging.metadata import InvalidMetadata, Metadata
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)


class ReleaseArtifactError(Exception):
    pass


def _single(stage_dir: Path, pattern: str) -> Path:
    matches = sorted(stage_dir.glob(pattern))
    if len(matches) != 1:
        raise ReleaseArtifactError(
            f"{stage_dir}: expected exactly one {pattern}, found "
            f"{[path.name for path in matches]}"
        )
    return matches[0]


def _parse_metadata(artifact: Path, member: str, raw: bytes) -> Metadata:
    try:
        return Metadata.from_email(raw, validate=True)
    except (ExceptionGroup, InvalidMetadata) as error:
        raise ReleaseArtifactError(f"{artifact.name}: invalid {member}: {error!r}")


def _wheel_metadata(wheel: Path, stem: str) -> Metadata:
    member = f"{stem}.dist-info/METADATA"
    with zipfile.ZipFile(wheel) as archive:
        if member not in archive.namelist():
            raise ReleaseArtifactError(f"{wheel.name}: missing {member}")
        return _parse_metadata(wheel, member, archive.read(member))


def _sdist_metadata(sdist: Path, stem: str) -> Metadata:
    member = f"{stem}/PKG-INFO"
    with tarfile.open(sdist) as archive:
        try:
            info = archive.getmember(member)
        except KeyError:
            raise ReleaseArtifactError(f"{sdist.name}: missing {member}")
        if not info.isfile():
            raise ReleaseArtifactError(f"{sdist.name}: {member} is not a regular file")
        return _parse_metadata(sdist, member, archive.extractfile(info).read())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_package(package_dir: Path, stage_dir: Path, workspace: set[str]) -> dict:
    """Return the manifest record for the one wheel and sdist in ``stage_dir``."""
    project = tomllib.loads((package_dir / "pyproject.toml").read_text())["project"]
    declared = canonicalize_name(project["name"])
    wheel = _single(stage_dir, "*.whl")
    sdist = _single(stage_dir, "*.tar.gz")
    try:
        wheel_name, wheel_version, _, _ = parse_wheel_filename(wheel.name)
        sdist_name, sdist_version = parse_sdist_filename(sdist.name)
    except (InvalidWheelFilename, InvalidSdistFilename) as error:
        raise ReleaseArtifactError(str(error))

    wheel_metadata = _wheel_metadata(wheel, "-".join(wheel.name.split("-")[:2]))
    sdist_metadata = _sdist_metadata(sdist, sdist.name.removesuffix(".tar.gz"))

    for artifact, filename_version, metadata, member in (
        (wheel, wheel_version, wheel_metadata, "METADATA"),
        (sdist, sdist_version, sdist_metadata, "PKG-INFO"),
    ):
        if metadata.version != filename_version:
            raise ReleaseArtifactError(
                f"{artifact.name}: {member} version {metadata.version} does not "
                f"match filename version {filename_version}"
            )
    names = {
        canonicalize_name(wheel_metadata.name),
        canonicalize_name(sdist_metadata.name),
        wheel_name,
        sdist_name,
    }
    if names != {declared}:
        raise ReleaseArtifactError(
            f"{package_dir}: declares {declared} but built {sorted(names)}"
        )
    if wheel_version != sdist_version:
        raise ReleaseArtifactError(
            f"{declared}: wheel version {wheel_version} != sdist version "
            f"{sdist_version}"
        )
    wheel_requires = sorted(map(str, wheel_metadata.requires_dist or []))
    sdist_requires = sorted(map(str, sdist_metadata.requires_dist or []))
    if wheel_requires != sdist_requires:
        raise ReleaseArtifactError(
            f"{declared}: wheel Requires-Dist {wheel_requires} != sdist "
            f"Requires-Dist {sdist_requires}"
        )

    return {
        "name": declared,
        "version": str(wheel_version),
        "requires": sorted(
            {
                canonicalize_name(requirement.name)
                for requirement in wheel_metadata.requires_dist or []
            }
            & workspace
        ),
        "wheel": {"filename": wheel.name, "sha256": _sha256(wheel)},
        "sdist": {"filename": sdist.name, "sha256": _sha256(sdist)},
    }


def build_manifest(libs_dir: Path, stage_root: Path, packages: list[str]) -> dict:
    """Validate every staged release package and return the build manifest."""
    workspace = {
        canonicalize_name(tomllib.loads(path.read_text())["project"]["name"])
        for path in libs_dir.glob("*/pyproject.toml")
    }
    records = [
        inspect_package(libs_dir / package, stage_root / package, workspace)
        for package in packages
    ]

    versions = {record["version"] for record in records}
    if len(versions) != 1:
        raise ReleaseArtifactError(
            "release packages built different versions: "
            + ", ".join(f"{record['name']}={record['version']}" for record in records)
        )
    built: set[str] = set()
    for record in records:
        for requirement in record["requires"]:
            if requirement not in built:
                raise ReleaseArtifactError(
                    f"{record['name']} requires {requirement}, which the release "
                    "set does not build before it"
                )
        built.add(record["name"])

    return {"version": versions.pop(), "packages": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libs-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("packages", nargs="+")
    args = parser.parse_args(argv)
    try:
        manifest = build_manifest(args.libs_dir, args.stage_dir, args.packages)
    except ReleaseArtifactError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
