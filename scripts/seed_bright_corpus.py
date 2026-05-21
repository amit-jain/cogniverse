#!/usr/bin/env python3
"""One-shot seeder for the BRIGHT-probe corpus.

Deploys the ``video_colpali_smol500_mv_frame_bright_probe_test`` schema
to the live Vespa pointed at by ``configs/config.json``, then feeds the
30-row engineered BRIGHT corpus. Idempotent — re-running redeploys the
schema (no-op via SchemaRegistry) and re-feeds documents (deterministic
``data_id``, Vespa overwrites in place).

Run:
    uv run python scripts/seed_bright_corpus.py
    uv run python scripts/seed_bright_corpus.py --verify-only

This decouples corpus seeding from the H-test fixture lifecycle so the
goldens recorded under the test harness stay valid across pytest
sessions and machine reboots.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_probe_rows() -> List[Dict[str, str]]:
    csv_path = REPO_ROOT / "data" / "testset" / "evaluation" / "bright_video_probes.csv"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = []
        for idx, row in enumerate(reader, start=1):
            row["query_id"] = f"bright_q{idx}"
            rows.append(row)
    return rows


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip schema deploy + ingest; just probe document count.",
    )
    args = parser.parse_args()

    # Re-use the helpers the H-test fixture defines so this script and
    # the fixture stay byte-equal in what they ingest.
    from tests.agents.integration.test_bright_video_probes import (
        BRIGHT_FULL_SCHEMA,
        _deploy_bright_schema,
        _ingest_bright_corpus,
        _live_vespa_endpoint,
        _wait_for_corpus_searchable,
    )

    base_url, http_port, _ = _live_vespa_endpoint()
    print(f"Vespa endpoint: {base_url}:{http_port}")
    print(f"Schema:         {BRIGHT_FULL_SCHEMA}")

    if args.verify_only:
        import httpx

        resp = httpx.post(
            f"{base_url}:{http_port}/search/",
            json={
                "yql": f"select * from sources {BRIGHT_FULL_SCHEMA} where true",
                "hits": 0,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        count = resp.json().get("root", {}).get("fields", {}).get("totalCount", 0)
        print(f"Document count: {count}")
        return 0 if count > 0 else 2

    rows = _load_probe_rows()
    print(f"Probe rows:     {len(rows)}")

    print("Deploying schema...")
    _deploy_bright_schema()

    print("Ingesting engineered corpus...")
    start = time.monotonic()
    total = _ingest_bright_corpus(rows)
    print(f"Fed {total} documents in {time.monotonic() - start:.1f}s")

    print(f"Waiting for {total} docs to become searchable...")
    _wait_for_corpus_searchable(expected_min=total)
    print("OK — BRIGHT corpus is live in Vespa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
