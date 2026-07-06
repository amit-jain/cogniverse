# Persistence & Backup

How cogniverse stores stateful data, what survives which failures, and
how to configure the backup destination per environment.

## The two-tier model

Cogniverse keeps a clear separation between live data and backups, with
each tier in its own failure domain:

| Tier | What | Examples |
|---|---|---|
| **Primary** | Where live data lives. Read+write hot path. | Vespa document store, Phoenix sqlite, MinIO bucket contents |
| **Backup** | Periodic snapshots in a different failure domain. Read on disaster recovery. | S3-compatible object storage (in-cluster MinIO for dev, R2 / B2 / AWS S3 for prod) |

A backup target in the **same failure domain** as primary is the
[OVH SBG2 fire (2021)](https://www.datacenterdynamics.com/en/news/ovhcloud-ordered-to-pay-250k-to-two-customers-who-lost-data-in-strasbourg-data-center-fire/)
and [GitLab.com 2017 incident](https://about.gitlab.com/blog/gitlab-dot-com-database-incident/)
mistake. Cogniverse's backup CronWorkflow defaults to the in-cluster
MinIO for dev convenience, but operators MUST point it at a separate
failure domain for any production workload — see [Backup destination](#backup-destination)
below.

## Durability matrix

What survives what, per primary-storage configuration:

| Operation | `hostStorage.enabled=true` (laptop dev) | `local-path` PVC (single-node prod) | Replicated CSI (Longhorn / Rook / cloud SC) |
|---|---|---|---|
| Pod restart | ✓ | ✓ | ✓ |
| StatefulSet recreate | ✓ | ✓ | ✓ |
| Laptop / node reboot | ✓ | ✓ | ✓ |
| `helm uninstall cogniverse` | ✓ (host fs untouched) | ✗ (`reclaimPolicy: Delete`) | depends on SC |
| `k3d cluster delete` (or equivalent) | ✓ (data on host fs) | ✗ (Docker volume goes with cluster) | ✗ |
| Single node loss in multi-node cluster | N/A | ✗ (node-pinned) | ✓ (replicated) |
| Disk failure on the only node | ✗ | ✗ | ✗ (need backup tier) |

**The backup tier is what defends against disk failure, accidental
`helm uninstall`, and full cluster loss.** Set up Tier B even when Tier A
looks safe.

## Per-component primary storage

Each stateful component has its own `<name>.persistence` block in
`values.yaml`. The chart honours the same shape for all of them:

```yaml
<component>:
  persistence:
    enabled: true              # off → ephemeral emptyDir (don't do this)
    storageClass: "fast-ssd"   # cloud: gp3 / pd-ssd / managed-csi
    accessMode: ReadWriteOnce
    size: "100Gi"
    annotations: {}
```

### Components

| Component | values key | Default size | Notes |
|---|---|---|---|
| Vespa | `vespa.persistence` | 100 Gi | Document store + config server — holds every schema (video/image/document/audio embeddings, `agent_memories`, knowledge graph, provenance, tenant/org metadata, adapter registry) **and** per-tenant config overrides written through `ConfigStore` (schema `config_metadata`; scopes `backend`, `gateway_agent`, `telemetry` all land here — `ConfigManager` methods default to `service="backend"`). One tar backs up all of it. `hostStorage.enabled=true` overrides to hostPath bind-mount. |
| Phoenix | `phoenix.persistence` | 50 Gi | sqlite traces. Distroless container — backed up in `mode: volume-mount` (tar of the mounted volume; `.db` files are staged through SQLite's online-backup API first so the tar isn't a torn read), not `kubectl exec` + tar. |
| MinIO | `minio.persistence` | 100 Gi | Default backup destination on dev. See [MinIO durability](#minio-durability) below. |
| HF model cache (per pod) | `hfCache.persistence` | 50 Gi each | Off by default (`enabled: false`). When enabled, one PVC per inference svc + runtime + ingestor, pre-warmed via init container. |
| Redis | `redis.persistence` | 10 Gi | Job queue + status-stream state (AOF on). Lose it = re-ingest in-flight jobs. |
| LLM (builtin) | `llm.ollama.persistence` (`llm.engine: ollama`) or `llm.vllm.persistence` (`llm.engine: vllm`) | 100 Gi | Model files for the in-cluster LLM. Only the PVC matching the selected `llm.engine` renders. |
| Semantic router | `semanticRouter.router.persistence` | 10 Gi | Classifier bundle cache (~GB) for the `vllm-sr` sidecar so it isn't re-downloaded on every restart/rollout. `emptyDir` when disabled. |

## MinIO durability (load-bearing for dev)

**MinIO IS the backup destination on dev clusters.** If MinIO data dies
with the cluster, the backup strategy is theatre. The chart provides
two MinIO storage modes:

### Mode 1 — PVC (default, suitable for cloud / multi-node prod)

```yaml
minio:
  persistence:
    storageClass: "fast-ssd"   # any real CSI provider
    size: "200Gi"
```

Backed by the configured `storageClass`. Durability follows whatever
that SC provides — replicated CSI (Longhorn, Rook, cloud SC) gives node-loss
tolerance; `local-path` does not.

### Mode 2 — hostPath bind-mount (dev / single-laptop)

```yaml
minio:
  persistence:
    hostPath: /host-data/minio   # k3d node path; bind-mounted from laptop
```

The chart skips PVC provisioning entirely. The MinIO Deployment mounts
`/host-data/minio` directly. The cogniverse cluster CLI bind-mounts
`~/.local/share/cogniverse` → `/host-data` on the k3d node, so MinIO data
ends up at `~/.local/share/cogniverse/minio` on the laptop fs and survives
`k3d cluster delete`.

## Backup destination

The vespa backup CronWorkflow (and any other backup CronWorkflow added
later) uploads to an S3-compatible endpoint. **One config block switches
between dev and cloud — same template, same code path, same workflow.**

### Enabling backup for local dev

`hostStorage.backup.enabled` defaults to `false` — opt in explicitly:

```yaml
hostStorage:
  backup:
    enabled: true
    bucket: cogniverse-backups
    schedule: "0 3 * * *"
    retainLast: 7
    services:
      - name: vespa
        dataPath: /opt/vespa/var
        podLabel: app.kubernetes.io/component=vespa
```

No `s3` block needed. Backup goes to in-cluster `cogniverse-minio`. Pair
this with `minio.persistence.hostPath` (above) so the bucket data
actually survives cluster destruction.

If `services` is omitted, the chart's built-in default already backs up
both `vespa` (via `kubectl exec` + tar) and `phoenix` (via a volume
mount, since the distroless Phoenix image has no `tar`/shell) nightly —
only override `services` to change which pods are covered or add more.

To point at an in-cluster MinIO Deployment whose secret was renamed,
set `hostStorage.backup.existingSecret` directly (a sibling of `bucket`
and `schedule`, not under `s3:`) instead of the full `s3` block. The
CronWorkflow template resolves the credentials secret in this order:
`s3.existingSecret` → `existingSecret` → the chart's own
`<release>-minio` secret.

The chart's own local-dev overlay, `charts/cogniverse/values.k3s.yaml`
(what `cogniverse up` deploys with on k3d), already sets
`hostStorage.enabled: true` and `hostStorage.backup.enabled: true`. It
does **not** set `minio.persistence.hostPath`, so on that overlay today
MinIO falls back to a `local-path` PVC and the nightly backups do
**not** survive `k3d cluster delete` — add the `minio.persistence.hostPath`
override from [MinIO durability](#minio-durability) on top of it (e.g.
via `--set minio.persistence.hostPath=/host-data/minio`) to close that
gap.

### Cloud — Cloudflare R2

```yaml
hostStorage:
  backup:
    enabled: true
    bucket: my-cogniverse-backups   # bucket pre-created in R2
    schedule: "0 3 * * *"
    retainLast: 30
    services:
      - name: vespa
        dataPath: /opt/vespa/var
        podLabel: app.kubernetes.io/component=vespa
    s3:
      endpoint: "https://<account>.r2.cloudflarestorage.com"
      existingSecret: "cogniverse-r2-creds"
      region: "auto"
```

Operator pre-creates the credentials secret with both the access key and
secret under specific keys (`rootUser` + `rootPassword`, regardless of
the actual S3 provider — this keeps the chart provider-agnostic):

```bash
kubectl -n cogniverse create secret generic cogniverse-r2-creds \
  --from-literal=rootUser=<r2-access-key-id> \
  --from-literal=rootPassword=<r2-secret-access-key>
```

### Cloud — AWS S3

```yaml
hostStorage:
  backup:
    s3:
      endpoint: "https://s3.us-east-1.amazonaws.com"
      existingSecret: "cogniverse-aws-creds"
      region: "us-east-1"
```

Same secret shape: `rootUser` = AWS access key id, `rootPassword` = AWS
secret key. For long-term retention enable bucket-level [object lock](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock.html)
+ versioning on the AWS side; the chart doesn't manage bucket policy.

### Cloud — Backblaze B2

```yaml
hostStorage:
  backup:
    s3:
      endpoint: "https://s3.us-west-002.backblazeb2.com"
      existingSecret: "cogniverse-b2-creds"
      region: "us-west-002"
```

## Recipes by environment

### Single-laptop dev (k3d)

Goal: data + backups survive `k3d cluster delete`.

```yaml
# charts/cogniverse/values.k3s.yaml already ships hostStorage.enabled=true
# and hostStorage.backup.enabled=true (this is what `cogniverse up` deploys
# on k3d). Layer the minio block below on top — it isn't in that overlay
# yet — to get full k3d-cluster-delete survival:
hostStorage:
  enabled: true                # vespa + phoenix bind-mounted to host
  path: /host-data
  backup:
    enabled: true              # backup CronWorkflow on
    services:
      - name: vespa
        dataPath: /opt/vespa/var
        podLabel: app.kubernetes.io/component=vespa
    # No s3 block → defaults to in-cluster MinIO

minio:
  persistence:
    hostPath: /host-data/minio  # MinIO bucket also on host fs
```

What survives:
- Pod / STS restart: ✓
- Laptop reboot: ✓
- `helm uninstall`: ✓ (host fs untouched)
- `k3d cluster delete`: ✓ (everything lives in `~/.local/share/cogniverse/`)
- Laptop nvme failure: ✗ (no offsite copy)

### Single-node bare-metal prod

Goal: tolerate node disk failures via offsite backup; node loss = full
restore from R2.

```yaml
# values-singlenode.yaml
hostStorage:
  enabled: false               # use real PVCs, not hostPath
  backup:
    enabled: true
    schedule: "0 */6 * * *"    # every 6h
    retainLast: 30
    services:
      - {name: vespa, dataPath: /opt/vespa/var, podLabel: app.kubernetes.io/component=vespa}
    s3:
      endpoint: "https://<account>.r2.cloudflarestorage.com"
      existingSecret: "cogniverse-r2-creds"
      region: "auto"

vespa: {persistence: {storageClass: "local-path", size: "500Gi"}}
phoenix: {persistence: {storageClass: "local-path", size: "100Gi"}}
minio: {persistence: {storageClass: "local-path", size: "1Ti"}}
```

### Multi-node cloud prod (EKS / GKE / AKS)

Goal: tolerate node loss via replicated CSI; survive AZ loss via offsite
backup; survive region loss via cross-region object replication.

```yaml
# values-cloud.yaml
hostStorage:
  enabled: false
  backup:
    enabled: true
    schedule: "0 */4 * * *"
    retainLast: 90
    services:
      - {name: vespa, dataPath: /opt/vespa/var, podLabel: app.kubernetes.io/component=vespa}
    s3:
      endpoint: "https://s3.us-east-1.amazonaws.com"
      existingSecret: "cogniverse-aws-creds"
      region: "us-east-1"

vespa: {persistence: {storageClass: "gp3", size: "1Ti"}}
phoenix: {persistence: {storageClass: "gp3", size: "200Gi"}}
minio: {persistence: {storageClass: "gp3", size: "5Ti"}}
```

For prod, additionally wire up [Velero](https://velero.io/) at the
cluster layer for resource-level backups (Deployments, ConfigMaps,
Secrets) — the cogniverse CronWorkflow only covers the application data
volumes, not the K8s metadata.

## Restore procedure

The dance to recover a stateful component from a MinIO/S3 backup:

1. **Take a final pre-restore backup** (in case the restore overwrites
   live state you didn't intend). The backup schedule is a `CronWorkflow`
   (`cogniverse-backup-vespa`) with an inline `workflowSpec`, not a
   standalone `WorkflowTemplate`, so trigger an ad hoc run of it with:
   ```bash
   argo submit --from cronworkflow/cogniverse-backup-vespa -n cogniverse
   ```

2. **Stop the StatefulSet:**
   ```bash
   kubectl -n cogniverse scale sts cogniverse-vespa --replicas=0
   ```

3. **Mount the volume in a utility pod** that has the backup credentials
   + tar:
   ```yaml
   apiVersion: v1
   kind: Pod
   metadata: {name: restore-vespa, namespace: cogniverse}
   spec:
     restartPolicy: Never
     securityContext: {runAsUser: 0}
     containers:
     - name: restore
       image: cogniverse/runtime-rocm:dev
       env:
       - {name: MINIO_ENDPOINT, value: "http://cogniverse-minio:9000"}
       - {name: MINIO_BUCKET, value: "cogniverse-backups"}
       - {name: MINIO_KEY, value: "vespa/vespa-<TIMESTAMP>.tar"}
       - {name: MINIO_ACCESS_KEY, valueFrom: {secretKeyRef: {name: cogniverse-minio, key: rootUser}}}
       - {name: MINIO_SECRET_KEY, valueFrom: {secretKeyRef: {name: cogniverse-minio, key: rootPassword}}}
       command: ["sh","-c"]
       args:
       - |
         set -eu
         python -c "import boto3, os; boto3.client('s3', endpoint_url=os.environ['MINIO_ENDPOINT'], aws_access_key_id=os.environ['MINIO_ACCESS_KEY'], aws_secret_access_key=os.environ['MINIO_SECRET_KEY']).download_file(os.environ['MINIO_BUCKET'], os.environ['MINIO_KEY'], '/tmp/restore.tar')"
         rm -rf /data/* /data/.[!.]* 2>/dev/null || true
         tar -xf /tmp/restore.tar -C /data --strip-components=1
       volumeMounts: [{name: data, mountPath: /data}]
     volumes:
     - name: data
       hostPath: {path: /host-data/vespa, type: DirectoryOrCreate}  # adjust for cloud SC
   ```
   Apply with `kubectl apply -f restore-pod.yaml` and wait for `Succeeded`.

4. **Scale the StatefulSet back up:**
   ```bash
   kubectl -n cogniverse scale sts cogniverse-vespa --replicas=1
   kubectl -n cogniverse rollout status sts/cogniverse-vespa
   ```

5. **Verify** by querying the runtime as the rest of the system would:
   ```bash
   kubectl -n cogniverse exec deploy/cogniverse-runtime -c runtime -- \
     curl -s 'http://cogniverse-vespa:8080/search/?yql=select+%2A+from+sources+%2A+where+true&hits=0' \
     | python3 -c 'import json,sys; print(json.load(sys.stdin)["root"]["fields"])'
   ```

## Backup verification (the "0" in 3-2-1-1-0)

A backup that's never been restored is a hypothesis. Cogniverse exercises
restore through normal cluster lifecycle ops:

- **Every `helm upgrade`** that touches a stateful component re-mounts
  the same hostPath / PVC and the data must come back intact. If it
  doesn't, the upgrade fails fast.
- **Every `cogniverse up` / `k3d cluster delete`** cycle is implicit
  hostStorage durability proof — pods restart against the same
  `~/.local/share/cogniverse/{vespa,phoenix,minio}` directories.
- **Periodic explicit restore drill** (recommended quarterly): follow
  the [Restore procedure](#restore-procedure) above using a recent
  backup tarball, pointed at a fresh sandbox cluster. Verify the doc
  count + a known query match production.

For cloud deployments where the backup target is offsite (R2 / B2 / AWS
S3), this drill also exercises the network path + credentials —
catching expired secrets, IAM mis-grants, region misconfigs that
silently rot if never tested.

## Why these defaults

Drawing on:

- **3-2-1-1-0 rule**: 3 copies, 2 media, 1 offsite, 1 immutable, 0
  verification errors ([SNIA cloud-native interpretation](https://www.snia.org/educational-library/re-imagining-3-2-1-backup-rule-cloud-native-applications-running-kubernetes-2020))
- **Failure-domain separation**: backup must not share power, network,
  hypervisor, or credentials with primary
- **Vespa data management** ([docs](https://docs.vespa.ai/en/operations/data-management.html)):
  Vespa OSS has no built-in snapshot — the chart's `kubectl exec tar`
  approach is what they recommend for self-hosted operators (with the
  caveat that it's not crash-consistent)
- **Velero best practices**: backup target must be S3-compatible object
  storage, never the source cluster's storage
- **MinIO production guide**: distributed mode (4+ nodes, erasure
  coding) is the only production-grade self-hosted MinIO. Single-node
  MinIO is staging / cache / dev — exactly how cogniverse uses it.

## Out of scope (track separately)

- **Velero integration** for K8s resource backup (Deployments, ConfigMaps).
  The CronWorkflow we ship only covers application data volumes; for
  full cluster recovery use Velero alongside.
- **CSI VolumeSnapshots** for crash-consistent snapshots. Requires a real
  CSI provider (Longhorn, cloud SC). The current `tar` approach reads
  the live filesystem while the source process keeps writing.
- **HF model cache backup.** Explicitly NOT backed up — it's rebuildable
  from HF Hub via the populate Job (`hfCache.persistence.minio.models`).
