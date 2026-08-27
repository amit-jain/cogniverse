"""The runtime image must install every media binary the ingestion processors
shell out to; the ingestor runs this image, and a missing binary surfaced in a
cluster as an empty video instead of an install error."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO / "libs" / "runtime" / "Dockerfile"
PROCESSORS = REPO / "libs" / "runtime" / "cogniverse_runtime" / "ingestion"


def _binaries_invoked_by_processors() -> set[str]:
    found: set[str] = set()
    for path in PROCESSORS.rglob("*.py"):
        text = path.read_text()
        for name in ("ffprobe", "ffmpeg"):
            if re.search(rf'"{name}"\s*,', text):
                found.add(name)
    return found


def _apt_packages_in_final_stage() -> set[str]:
    text = DOCKERFILE.read_text()
    final = text[text.rindex("\nFROM ") :]
    packages: set[str] = set()
    for block in re.findall(
        r"apt-get install -y\s*\\\n((?:[ \t]+\S+[ \t]*\\?\n)+)", final
    ):
        for line in block.splitlines():
            token = line.strip().rstrip("\\").strip()
            if token and not token.startswith("&&"):
                packages.add(token)
    return packages


def test_processor_media_binaries_are_installed_in_the_runtime_image():
    assert _binaries_invoked_by_processors() == {"ffprobe", "ffmpeg"}
    assert _apt_packages_in_final_stage() == {
        "ffmpeg",
        "libgomp1",
        "libgl1",
        "libglib2.0-0",
        "curl",
        "unzip",
    }
