"""Shared constants for the cogniverse CLI.

Single source of truth so changing the cluster namespace or runtime
URL doesn't require N parallel edits across the cli/ tree.
"""

# Kubernetes namespace the cogniverse stack deploys into. Set by the
# Helm chart's release.namespace and matched by every cli command
# that talks to the cluster (deploy, secrets, cluster, sandbox).
NAMESPACE = "cogniverse"

# Helm release name the stack installs under. The chart's fullname helper
# collapses to this (chart name and release name match), so Secret and
# Service names the cli has to address are "<RELEASE_NAME>-<suffix>".
RELEASE_NAME = "cogniverse"

# Default runtime HTTP endpoint when the cli is run outside the
# cluster (typical for dev). NodePort 28000 is the chart's
# externally-exposed runtime port; production callers running
# inside the cluster should override via env or arg if they need
# the in-cluster service URL.
RUNTIME_URL = "http://localhost:28000"
