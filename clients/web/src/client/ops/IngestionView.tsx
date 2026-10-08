import { useCallback, useEffect, useState } from 'react';
import { Alert, Panel, messageOf, useAction, useLoad } from './common';
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
  ingest_id?: string;
  state: string;
  existing: boolean;
  filename: string;
}

/** A profile ``/ingestion/upload`` ingests a file into, as
 * ``/ingestion/profiles`` lists it. */
export interface UploadProfile {
  name: string;
  type?: string;
  kind: string;
  extensions: string[];
}

interface UploadTargets {
  backend: string;
  default_profile: string | null;
  profiles: UploadProfile[];
}

interface Followed {
  ingestId: string;
  filename?: string;
  /** The upload matched bytes an earlier ingest already took. */
  existing?: boolean;
}

/** One file sent to several profiles: an ingest per profile, or the reason
 * the runtime refused it. */
export interface Batch {
  id: string;
  filename: string;
  entries: { profile: string; ingestId?: string; error?: string }[];
}

/** The states after which the runtime closes an ingest's event stream. */
export const TERMINAL = new Set(['complete', 'failed', 'cancelled']);

/** How long an ingest is followed for a terminal state before the view
 * gives up on it. */
export const INGEST_DEADLINE_MS = 900_000;

const FOLLOWED_KEY = 'cogniverse.ingests';

/** One ingest's events, replayed from its start and then followed live
 * until a terminal state or the deadline; a stream the runtime closes while
 * idle resumes after the last event seen. */
function useIngestEvents(ingestId: string): { events: IngestEvent[]; error: string; timedOut: boolean } {
  const [events, setEvents] = useState<IngestEvent[]>([]);
  const [error, setError] = useState('');
  const [timedOut, setTimedOut] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    const deadline = setTimeout(() => {
      setTimedOut(true);
      controller.abort();
    }, INGEST_DEADLINE_MS);
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
    follow()
      .catch((e: unknown) => {
        if (!controller.signal.aborted) setError(messageOf(e));
      })
      .finally(() => clearTimeout(deadline));
    return () => {
      clearTimeout(deadline);
      controller.abort();
    };
  }, [ingestId]);
  return { events, error, timedOut };
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

/** The outcome of an ingest followed past the deadline without ending. */
export function deadlineOutcome(lastState: string | undefined): { text: string; failed: boolean } {
  return {
    text: `No terminal state within ${INGEST_DEADLINE_MS / 1000} s; the last state was ${lastState ?? 'none'}. Follow it by ID to keep watching.`,
    failed: true,
  };
}

/** Why ``filename`` cannot go to ``profile``, in the words the runtime
 * refuses it with; nothing when the profile reads it. */
export function unreadableReason(filename: string, profile: UploadProfile): string | undefined {
  const dot = filename.lastIndexOf('.');
  const suffix = dot > 0 ? filename.slice(dot).toLowerCase() : '';
  if (profile.extensions.includes(suffix)) return undefined;
  const what = suffix ? `is a ${suffix} file` : 'has no file extension';
  return `${filename} ${what}; profile '${profile.name}' ingests ${profile.kind} files (${profile.extensions.join(', ')}).`;
}

/** A batch's line once every profile of it has ended, or how far it got. */
export function batchSummary(batch: Batch, outcomes: Record<string, { failed: boolean }>): string {
  const total = batch.entries.length;
  const ended = batch.entries.filter((entry) => entry.error || (entry.ingestId && outcomes[entry.ingestId]));
  if (ended.length < total) return `${batch.filename}: ${ended.length} of ${total} profiles finished.`;
  const failed = batch.entries
    .filter((entry) => entry.error || (entry.ingestId && outcomes[entry.ingestId].failed))
    .map((entry) => entry.profile);
  if (!failed.length) return `${batch.filename}: all ${total} profile${total === 1 ? '' : 's'} ingested.`;
  return `${batch.filename}: ingestion failed for ${failed.join(', ')} (${failed.length} of ${total} profiles).`;
}

function sourceName(sourceUrl?: string): string | undefined {
  return sourceUrl?.split('/').pop();
}

function IngestRow({
  ingest,
  onOutcome,
}: {
  ingest: Followed;
  onOutcome: (ingestId: string, failed: boolean) => void;
}) {
  const { events, error, timedOut } = useIngestEvents(ingest.ingestId);
  const latest = events[events.length - 1];
  const queued = events.find((event) => event.source_url);
  const ended = latest && TERMINAL.has(latest.state);
  const outcome = timedOut && !ended ? deadlineOutcome(latest?.state) : ingestOutcome(latest, ingest.existing ?? false);
  const settled = ended || timedOut || Boolean(error);
  const failed = outcome.failed || Boolean(error);
  useEffect(() => {
    if (settled) onOutcome(ingest.ingestId, failed);
  }, [settled, failed, ingest.ingestId, onOutcome]);
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

interface Remembered {
  ingests: Followed[];
  batches: Batch[];
}

function remembered(): Remembered {
  try {
    const raw = localStorage.getItem(FOLLOWED_KEY);
    const value = raw ? (JSON.parse(raw) as Partial<Remembered>) : {};
    return { ingests: value.ingests ?? [], batches: value.batches ?? [] };
  } catch {
    return { ingests: [], batches: [] };
  }
}

function remember(value: Remembered) {
  try {
    localStorage.setItem(FOLLOWED_KEY, JSON.stringify(value));
  } catch {
    // Without storage the list lasts as long as the page.
  }
}

export function IngestionView() {
  const [tenant, setTenant] = useState('');
  const [followed, setFollowed] = useState<Remembered>(remembered);
  const [outcomes, setOutcomes] = useState<Record<string, { failed: boolean }>>({});
  const [notice, setNotice] = useState('');
  const update = (change: (previous: Remembered) => Remembered) =>
    setFollowed((previous) => {
      const next = change(previous);
      remember(next);
      return next;
    });
  const follow = (ingests: Followed[], batch?: Batch) =>
    update((previous) => {
      const ids = new Set(ingests.map((ingest) => ingest.ingestId));
      return {
        ingests: [...ingests, ...previous.ingests.filter((item) => !ids.has(item.ingestId))],
        batches: batch ? [batch, ...previous.batches] : previous.batches,
      };
    });
  const onOutcome = useCallback(
    (ingestId: string, failed: boolean) =>
      setOutcomes((previous) =>
        previous[ingestId]?.failed === failed ? previous : { ...previous, [ingestId]: { failed } },
      ),
    [],
  );

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
          onUploaded={(batch, uploads) => {
            const several = batch.entries.length > 1;
            if (!uploads.length) {
              follow([], batch);
              return;
            }
            setNotice(
              uploads
                .map(({ profile, upload }) =>
                  upload.existing
                    ? `${upload.filename} matches ingest ${upload.ingest_id} (${upload.state}); following it.`
                    : `Queued ${upload.filename} as ingest ${upload.ingest_id}${several ? ` (${profile})` : ''}.`,
                )
                .join(' '),
            );
            follow(
              uploads.map(({ upload }) => ({
                ingestId: upload.ingest_id!,
                filename: upload.filename,
                existing: upload.existing,
              })),
              batch,
            );
          }}
        />
      )}
      <FollowIngest onFound={(ingestId) => follow([{ ingestId }])} />
      {followed.batches.length > 0 && (
        <Panel title="Batches">
          <ul className="batches">
            {followed.batches.map((batch) => (
              <li key={batch.id}>{batchSummary(batch, outcomes)}</li>
            ))}
          </ul>
        </Panel>
      )}
      <Panel
        title="Ingests"
        actions={
          followed.ingests.length > 0 && (
            <button type="button" onClick={() => update(() => ({ ingests: [], batches: [] }))}>
              Clear list
            </button>
          )
        }
      >
        {followed.ingests.length === 0 ? (
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
              {followed.ingests.map((ingest) => (
                <IngestRow key={ingest.ingestId} ingest={ingest} onOutcome={onOutcome} />
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </div>
  );
}

function UploadContent({
  tenant,
  onUploaded,
}: {
  tenant: string;
  onUploaded: (batch: Batch, uploads: { profile: string; upload: Upload }[]) => void;
}) {
  const targets = useLoad(
    (signal) => runtimeJson<UploadTargets>(`/ingestion/profiles?tenant_id=${seg(tenant)}`, { signal }),
    [tenant],
  );
  const [file, setFile] = useState<File | null>(null);
  const [chosen, setChosen] = useState<string[]>();
  const [force, setForce] = useState(false);
  const [formKey, setFormKey] = useState(0);
  const [refused, setRefused] = useState<string[]>([]);
  const action = useAction();
  const profiles = targets.data?.profiles ?? [];
  const selected = chosen ?? (targets.data?.default_profile ? [targets.data.default_profile] : []);
  const selectedProfiles = profiles.filter((profile) => selected.includes(profile.name));
  const accept = [...new Set(selectedProfiles.flatMap((profile) => profile.extensions))].join(',');
  const unreadable = file
    ? selectedProfiles.flatMap((profile) => unreadableReason(file.name, profile) ?? [])
    : [];
  const ready = Boolean(targets.data) && selected.length > 0;

  return (
    <Panel title={`Upload to ${tenant}`}>
      {targets.loading && !targets.data && <p className="muted">Reading what the runtime accepts…</p>}
      {targets.error && (
        <div>
          <Alert>{`The runtime cannot take uploads now: ${targets.error}`}</Alert>
          <button type="button" onClick={targets.reload}>
            Check again
          </button>
        </div>
      )}
      <form
        key={formKey}
        className="inline-form"
        aria-label="Upload content"
        onSubmit={(e) => {
          e.preventDefault();
          if (!file || !ready || unreadable.length) return;
          action.run(async () => {
            const batch: Batch = { id: crypto.randomUUID(), filename: file.name, entries: [] };
            const uploads: { profile: string; upload: Upload }[] = [];
            const failures: string[] = [];
            for (const profile of selected) {
              const body = new FormData();
              body.set('file', file);
              body.set('tenant_id', tenant);
              body.set('profile', profile);
              body.set('backend', targets.data!.backend);
              try {
                const upload = await runtimeJson<Upload>(`/ingestion/upload${force ? '?force=true' : ''}`, {
                  method: 'POST',
                  body,
                });
                if (!upload?.ingest_id)
                  throw new Error(`The runtime accepted ${file.name} but answered no ingest ID.`);
                uploads.push({ profile, upload });
                batch.entries.push({ profile, ingestId: upload.ingest_id });
              } catch (error) {
                const message = messageOf(error);
                failures.push(selected.length > 1 ? `${profile}: ${message}` : message);
                batch.entries.push({ profile, error: message });
              }
            }
            setRefused(failures);
            onUploaded(batch, uploads);
            if (!failures.length) {
              setFile(null);
              setFormKey((n) => n + 1);
            }
          });
        }}
      >
        {targets.data && (
          <p className="muted">{`Backend: ${targets.data.backend}`}</p>
        )}
        {targets.data && (
          <fieldset className="check-list">
            <legend>Profiles</legend>
            {profiles.length === 0 && <p className="muted">This tenant has no profile that ingests an uploaded file.</p>}
            {profiles.map((profile) => (
              <label key={profile.name} className="check">
                <input
                  type="checkbox"
                  aria-label={profile.name}
                  checked={selected.includes(profile.name)}
                  onChange={(e) =>
                    setChosen(
                      e.target.checked
                        ? [...selected, profile.name]
                        : selected.filter((name) => name !== profile.name),
                    )
                  }
                />
                {profile.name}
                <span className="muted">{` ${profile.kind}: ${profile.extensions.join(' ')}`}</span>
              </label>
            ))}
          </fieldset>
        )}
        <label>
          File
          <input
            required
            type="file"
            accept={accept || undefined}
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null);
              setRefused([]);
            }}
          />
        </label>
        <label className="check">
          <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
          Ingest again even if already ingested
        </label>
        <button type="submit" disabled={action.pending || !ready}>
          {action.pending ? 'Uploading…' : 'Upload and ingest'}
        </button>
        {targets.data && selected.length === 0 && <p className="muted">Choose at least one profile.</p>}
        {unreadable.map((reason) => (
          <Alert key={reason}>{reason}</Alert>
        ))}
        {refused.map((reason) => (
          <Alert key={reason}>{reason}</Alert>
        ))}
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
