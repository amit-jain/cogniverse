"""One-off cleanup for the ``config_metadata`` Vespa schema.

``VespaConfigStore.set_config`` now prunes per write, but legacy rows
that accumulated before that change (~8k+ observed in dev clusters)
will not naturally clear unless every config_id is rewritten. Run this
script once against the live cluster to drain the backlog.

Usage:
    uv run python scripts/prune_config_metadata.py \\
        --host http://localhost --port 8080 --keep 10
"""

from __future__ import annotations

import argparse
import logging
import sys

from cogniverse_vespa.config.config_store import VespaConfigStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--keep",
        type=int,
        default=10,
        help="versions per config_id to retain (latest N)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report distinct config_ids and their version counts; do not delete",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    store = VespaConfigStore(
        backend_url=args.host,
        backend_port=args.port,
        keep_versions=args.keep,
    )

    if args.dry_run:
        counts = store.count_version_rows()
        total_rows = sum(counts.values())
        would_drop = sum(max(0, c - args.keep) for c in counts.values())
        print(f"distinct config_ids: {len(counts)}")
        print(f"total rows:          {total_rows}")
        print(f"would drop:          {would_drop} (keeping latest {args.keep} per id)")
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("top-10 by version count:")
        for cid, n in top:
            print(f"  {n:>6}  {cid}")
        return 0

    dropped = store.prune_all_configs(keep=args.keep)
    print(f"prune complete: deleted {dropped} stale version rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
