# Serving Blob Ring Migration

`ArtifactManager.load_blob` resolves serving blobs from the ring slot datasets
`dspy-{kind}-{tenant}-{key}--r0`, `--r1` and `--r2`. A Phoenix store written by
an earlier release holds each blob under its base name
`dspy-{kind}-{tenant}-{key}`, which the ring does not read.

Every serving blob is affected: `config/artefact_state_{agent}` (unmigrated, a
tenant serves un-optimized prompts), `config/pin_quotas`,
`config/signature_variants`, each `config/blob_state_{kind}_{key}` activation
pointer, `config/*_ground_truth`, `model/*`, `xgboost/*` and `workflow/*`.

## Deploy sequence

The old and new pods read and write different datasets for the same blob, so
the fleet is replaced, not rolled: old replicas are stopped, the migration is
run once, new replicas are started.

1. Stop every replica that writes serving blobs — the runtime deployment and
   any optimization CronWorkflow in flight.
2. Collect the tenant ids to migrate:

   ```bash
   uv run python scripts/discover_tenants.py
   ```

3. Run the migration once, naming every tenant:

   ```bash
   uv run python scripts/migrate_blob_ring_slots.py \
       --phoenix-url http://phoenix:6006 \
       --tenant acme:acme --tenant beta:beta
   ```

   `--dry-run` reports what would be copied without writing.

4. Start the new replicas.

## What the script does

Each base-name blob is copied into ring slot 0 as revision 0. The base-name
dataset is left in place. The next publication of that blob takes revision 1 in
slot 1 and prunes nothing, so the migrated content stays readable until a
second publication supersedes it.

A blob that already has a populated ring slot is reported `already_in_ring` and
left alone, so a second run changes nothing; `--force` overwrites slot 0.

The summary lists, per tenant, every base-name dataset and its outcome
(`migrated`, `already_in_ring`, `would_migrate`). A base-name blob dataset that
belongs to no tenant named on the command line is printed as `UNATTRIBUTED` and
the script exits 2 — that tenant's serving blobs are not in the ring.
