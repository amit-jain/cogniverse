"""Queue-driven ingestion path.

Replaces the in-process ``VideoIngestionPipeline`` invocation in the
runtime's HTTP handlers with a Redis-Streams queue + worker pods.

  POST /ingest        → submit_api.submit_ingest
  worker pods         → worker.run_worker (blocking claim loop)
  GET /ingest/{id}/events → status_api.stream_events (SSE)

Idempotency: ``sha256(source_url + profile + tenant_id)`` keys an
``ingest:done:<sha>`` set. Re-submissions hit the set and return the
existing ingest_id without re-enqueuing.

Backpressure: per-tenant active counter ``ingest:active:<tenant>`` plus
cluster-wide ``XLEN ingest:queue`` cap. Either threshold returns 429.
"""
