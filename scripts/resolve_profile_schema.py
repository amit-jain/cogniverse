"""Resolve a backend profile name to its Vespa schema name.

The tenant-provisioning workflow deploys and verifies a schema per profile,
but the schema FILE and the tenant-scoped schema id are named after the
profile's ``schema_name`` — which is not always the profile name (e.g.
``audio_clap_semantic`` -> ``audio_content``). This resolves that mapping from
the rendered config so the workflow deploys the right file.

Usage:
    python scripts/resolve_profile_schema.py <profile_name> [config_path]

Prints the schema_name to stdout. Exits non-zero (and prints nothing to
stdout) if the profile is unknown, so a caller using command substitution
fails loudly rather than deploying a wrong/empty name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def resolve_profile_schema(profile: str, config_path: Path) -> str:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profiles = config.get("backend", {}).get("profiles", {})
    if profile not in profiles:
        raise KeyError(
            f"Profile {profile!r} not found in {config_path}. "
            f"Known profiles: {sorted(profiles)}"
        )
    schema_name = profiles[profile].get("schema_name")
    if not schema_name:
        raise KeyError(f"Profile {profile!r} has no schema_name in {config_path}")
    return schema_name


def main(argv: list[str]) -> int:
    if not argv:
        print(
            "usage: resolve_profile_schema.py <profile> [config_path]", file=sys.stderr
        )
        return 2
    profile = argv[0]
    config_path = Path(argv[1]) if len(argv) > 1 else Path("configs/config.json")
    try:
        print(resolve_profile_schema(profile, config_path))
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
