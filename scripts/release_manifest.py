"""Validate the wheel and sdist built for each release package and write the
build manifest describing exactly those artifacts; check that the manifest's
artifacts are publishable and that a package index serves them unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
import tomllib
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

from packaging.metadata import InvalidMetadata, Metadata
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

MANIFEST_NAME = "BUILD_MANIFEST.json"


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


def publication_plan(dist_dir: Path) -> list[dict]:
    """Return the manifest's packages once every listed artifact is in ``dist_dir``
    with the manifest's name, version and sha256, under a version an index accepts."""
    manifest_path = dist_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ReleaseArtifactError(
            f"{manifest_path} not found; run scripts/build_packages.sh first"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
        version = Version(manifest["version"])
        packages = manifest["packages"]
        entries = [
            (package["name"], Version(package["version"]), package[kind], parse)
            for package in packages
            for kind, parse in (
                ("wheel", parse_wheel_filename),
                ("sdist", parse_sdist_filename),
            )
        ]
    except (KeyError, TypeError, ValueError, InvalidVersion) as error:
        raise ReleaseArtifactError(f"{manifest_path}: malformed manifest: {error!r}")
    if not packages:
        raise ReleaseArtifactError(f"{manifest_path} lists no packages")
    if version.local is not None:
        raise ReleaseArtifactError(
            f"release version {version} has the local segment +{version.local}, "
            "which PyPI and TestPyPI reject; publish a build of a release tag"
        )

    for name, package_version, entry, parse in entries:
        path = dist_dir / entry["filename"]
        try:
            file_name, file_version = parse(entry["filename"])[:2]
        except (InvalidWheelFilename, InvalidSdistFilename) as error:
            raise ReleaseArtifactError(str(error))
        if (file_name, file_version, package_version) != (name, version, version):
            raise ReleaseArtifactError(
                f"{manifest_path}: {entry['filename']} is listed for {name} "
                f"{package_version} in release {version}"
            )
        if not path.is_file():
            raise ReleaseArtifactError(f"{path} is listed in the manifest but missing")
        digest = _sha256(path)
        if digest != entry["sha256"]:
            raise ReleaseArtifactError(
                f"{path}: sha256 {digest} does not match the manifest sha256 "
                f"{entry['sha256']}"
            )
    return packages


class _IndexLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.digests: dict[str, str | None] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        url, _, fragment = (dict(attrs).get("href") or "").partition("#")
        algorithm, _, digest = fragment.partition("=")
        filename = unquote(url.rsplit("/", 1)[-1])
        self.digests[filename] = digest if algorithm == "sha256" else None


def _served_digests(index_url: str, name: str) -> dict[str, str | None]:
    import requests

    url = f"{index_url.rstrip('/')}/{name}/"
    try:
        response = requests.get(url, headers={"Accept": "text/html"}, timeout=60)
    except requests.RequestException as error:
        raise ReleaseArtifactError(f"{url}: {error!r}")
    if response.status_code == 404:
        return {}
    if response.status_code != 200:
        raise ReleaseArtifactError(f"{url}: HTTP {response.status_code}")
    links = _IndexLinks()
    links.feed(response.text)
    return links.digests


def check_index(
    dist_dir: Path, index_url: str, timeout: float, interval: float = 10.0
) -> int:
    """Wait until ``index_url`` serves every manifest artifact and return their
    count; any served artifact whose sha256 differs from the manifest fails."""
    packages = publication_plan(dist_dir)
    deadline = time.monotonic() + timeout
    while True:
        missing = []
        for package in packages:
            served = _served_digests(index_url, package["name"])
            for entry in (package["wheel"], package["sdist"]):
                if entry["filename"] not in served:
                    missing.append(entry["filename"])
                elif served[entry["filename"]] != entry["sha256"]:
                    raise ReleaseArtifactError(
                        f"{index_url}: {entry['filename']} has sha256 "
                        f"{served[entry['filename']]}, manifest has {entry['sha256']}"
                    )
        if not missing:
            return 2 * len(packages)
        if time.monotonic() >= deadline:
            raise ReleaseArtifactError(
                f"{index_url} does not serve {', '.join(missing)}"
            )
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="validate staged builds, write manifest")
    build.add_argument("--libs-dir", type=Path, required=True)
    build.add_argument("--stage-dir", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("packages", nargs="+")
    publishable = commands.add_parser(
        "publishable", help="verify dist/ against its manifest, print upload order"
    )
    publishable.add_argument("--dist-dir", type=Path, required=True)
    index = commands.add_parser(
        "check-index", help="verify an index serves the manifest artifacts"
    )
    index.add_argument("--dist-dir", type=Path, required=True)
    index.add_argument("--index-url", required=True)
    index.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_manifest(args.libs_dir, args.stage_dir, args.packages)
            args.output.write_text(json.dumps(manifest, indent=2) + "\n")
        elif args.command == "publishable":
            for package in publication_plan(args.dist_dir):
                print(
                    "\t".join(
                        (
                            package["name"],
                            package["version"],
                            package["wheel"]["filename"],
                            package["sdist"]["filename"],
                        )
                    )
                )
        else:
            count = check_index(args.dist_dir, args.index_url, args.timeout)
            print(
                f"Verified {count} file(s) at {args.index_url} against the "
                "manifest digests"
            )
    except ReleaseArtifactError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
