import { useEffect, useState } from 'react';
import { Alert, Panel, messageOf, useAction } from './common';
import { errorMessage, runtimeJson, seg } from './http';
import { parseSse } from './sse';
import { TenantChooser } from './tenants';

interface IngestEvent {
  state: string;
  source_url?: string;
  filename?: string;
  profile?: string;
  error?: string;
  error_type?: string;
  reason?: string;
  cleanup_error?: string;
  result?: {
    video_id?: string;
    chunks?: number;
    documents_fed?: number;
    graph_nodes?: number;
    graph_edges?: number;
  };
}

interface Upload {
  ingest_id: string;
  state: string;
  existing: boolean;
  filename: string;
}

interface Followed {
  ingestId: string;
  filename?: string;
  /** The upload matched bytes an earlier ingest already took. */
  existing?: boolean;
}

/** The states after which the runtime closes an ingest's event stream. */
export const TERMINAL = new Set(['complete', 'failed', 'cancelled']);

/** One ingest's events, replayed from its start and then followed live
 * until a terminal state; a stream the runtime closes while idle resumes
 * after the last event seen. */
function useIngestEvents(ingestId: string): { events: IngestEvent[]; error: string } {
  const [events, setEvents] = useState<IngestEvent[]>([]);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    const follow = async () => {
      let lastId: string | undefined;
      for (;;) {
        const query = lastId ? `?last-event-id=${seg(lastId)}` : '';
        const response = await fetch(`/ui-api/runtime/ingestion/${seg(ingestId)}/events${query}`, {
          headers: { accept: 'text/event-stream' },
          signal: controller.signal,
        });
        if (!response.ok || !response.body) {
          const body = await response.json().catch(() => null);
          throw new Error(errorMessage(body, response.status));
        }
        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
        let buffer = '';
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          const parsed = parseSse(buffer + value);
          buffer = parsed.rest;
          for (const frame of parsed.frames) {
            lastId = frame.id ?? lastId;
            const event = JSON.parse(frame.data) as IngestEvent;
            setEvents((previous) => [...previous, event]);
            if (TERMINAL.has(event.state)) return;
          }
        }
      }
    };
    follow().catch((e: unknown) => {
      if (!controller.signal.aborted) setError(messageOf(e));
    });
    return () => controller.abort();
  }, [ingestId]);
  return { events, error };
}

/** What an ingest's latest event says, and whether it ended without the
 * content being ingested: failed, cancelled, or complete without feeding a
 * document. ``existing`` is a re-upload of bytes already ingested, whose
 * completion may no longer carry its counts. */
export function ingestOutcome(event: IngestEvent | undefined, existing: boolean): { text: string; failed: boolean } {
  if (!event) return { text: '', failed: false };
  let text = '';
  let failed = false;
  if (event.state === 'complete') {
    const r = event.result ?? {};
    const video = r.video_id ? `${r.video_id}: ` : '';
    const fed = r.documents_fed as unknown;
    if (fed === undefined && existing) text = `${video}already ingested; nothing was fed again.`;
    else if (fed === undefined) {
      text = `${video}completed without reporting the documents it fed.`;
      failed = true;
    } else if (typeof fed !== 'number' || !Number.isInteger(fed)) {
      text = `${video}completed with an invalid documents_fed (${JSON.stringify(fed)}).`;
      failed = true;
    } else if (fed <= 0) {
      text = `${video}completed without feeding any documents.`;
      failed = true;
    } else {
      text = `${video}${r.chunks ?? 0} chunks, ${fed} documents fed.`;
      if (r.graph_nodes !== undefined) text += ` Graph: ${r.graph_nodes} nodes, ${r.graph_edges ?? 0} edges.`;
    }
  } else if (event.state === 'failed') {
    text = `${event.error_type}: ${event.error}`;
    failed = true;
  } else if (event.state === 'cancelled') {
    text = `Cancelled: ${event.reason || 'no reason given'}`;
    failed = true;
  } else if (event.state === 'retrying') text = `Retrying after ${event.error_type}: ${event.error}`;
  if (event.cleanup_error) text += ` Cleanup failed: ${event.cleanup_error}`;
  return { text, failed };
}

function sourceName(sourceUrl?: string): string | undefined {
  return sourceUrl?.split('/').pop();
}

function IngestRow({ ingest }: { ingest: Followed }) {
  const { events, error } = useIngestEvents(ingest.ingestId);
  const latest = events[events.length - 1];
  const queued = events.find((event) => event.source_url);
  const outcome = ingestOutcome(latest, ingest.existing ?? false);
  return (
    <tr>
      <td>{ingest.ingestId}</td>
      <td>{ingest.filename ?? queued?.filename ?? sourceName(queued?.source_url) ?? '—'}</td>
      <td>{queued?.profile ?? '—'}</td>
      <td>{latest?.state ?? (error ? '—' : 'connecting…')}</td>
      <td>
        {outcome.failed ? <Alert>{outcome.text}</Alert> : outcome.text}
        {error && <Alert>{error}</Alert>}
      </td>
    </tr>
  );
}

export function IngestionView() {
  const [tenant, setTenant] = useState('');
  const [ingests, setIngests] = useState<Followed[]>([]);
  const [notice, setNotice] = useState('');
  const follow = (ingest: Followed) =>
    setIngests((previous) => [ingest, ...previous.filter((item) => item.ingestId !== ingest.ingestId)]);

  return (
    <div className="ops-view">
      <TenantChooser
        action="Use tenant"
        onChoose={(chosen) => {
          setTenant(chosen);
          setNotice('');
        }}
      />
      {notice && <Alert tone="ok">{notice}</Alert>}
      {tenant && (
        <UploadContent
          key={tenant}
          tenant={tenant}
          onUploaded={(upload) => {
            setNotice(
              upload.existing
                ? `${upload.filename} matches ingest ${upload.ingest_id} (${upload.state}); following it.`
                : `Queued ${upload.filename} as ingest ${upload.ingest_id}.`,
            );
            follow({ ingestId: upload.ingest_id, filename: upload.filename, existing: upload.existing });
          }}
        />
      )}
      <FollowIngest onFound={(ingestId) => follow({ ingestId })} />
      <Panel title="Ingests">
        {ingests.length === 0 ? (
          <p className="muted">No ingests followed yet. Upload a file or follow an ingest by its ID.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Ingest</th>
                <th>Source</th>
                <th>Profile</th>
                <th>State</th>
                <th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {ingests.map((ingest) => (
                <IngestRow key={ingest.ingestId} ingest={ingest} />
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </div>
  );
}

function UploadContent({ tenant, onUploaded }: { tenant: string; onUploaded: (upload: Upload) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [profile, setProfile] = useState('');
  const [force, setForce] = useState(false);
  const [formKey, setFormKey] = useState(0);
  const action = useAction();
  return (
    <Panel title={`Upload to ${tenant}`}>
      <form
        key={formKey}
        className="inline-form"
        aria-label="Upload content"
        onSubmit={(e) => {
          e.preventDefault();
          if (!file) return;
          action.run(async () => {
            const body = new FormData();
            body.set('file', file);
            body.set('tenant_id', tenant);
            if (profile.trim()) body.set('profile', profile.trim());
            const upload = await runtimeJson<Upload>(`/ingestion/upload${force ? '?force=true' : ''}`, {
              method: 'POST',
              body,
            });
            onUploaded(upload);
            setFile(null);
            setFormKey((n) => n + 1);
          });
        }}
      >
        <label>
          File
          <input required type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        </label>
        <label>
          Profile
          <input value={profile} onChange={(e) => setProfile(e.target.value)} placeholder="tenant's default" />
        </label>
        <label className="check">
          <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
          Ingest again even if already ingested
        </label>
        <button type="submit" disabled={action.pending}>
          {action.pending ? 'Uploading…' : 'Upload and ingest'}
        </button>
        {action.error && <Alert>{action.error}</Alert>}
      </form>
    </Panel>
  );
}

function FollowIngest({ onFound }: { onFound: (ingestId: string) => void }) {
  const [ingestId, setIngestId] = useState('');
  const action = useAction();
  return (
    <Panel title="Follow an ingest">
      <form
        className="inline-form"
        aria-label="Follow an ingest"
        onSubmit={(e) => {
          e.preventDefault();
          action.run(async () => {
            const id = ingestId.trim();
            // A status read answers 404 for an unknown ID; the event stream
            // would wait for events that never come.
            await runtimeJson(`/ingestion/${seg(id)}/status`);
            onFound(id);
            setIngestId('');
          });
        }}
      >
        <label>
          Ingest ID
          <input required value={ingestId} onChange={(e) => setIngestId(e.target.value)} />
        </label>
        <button type="submit" disabled={action.pending}>
          Follow
        </button>
        {action.error && <Alert>{action.error}</Alert>}
      </form>
    </Panel>
  );
}
