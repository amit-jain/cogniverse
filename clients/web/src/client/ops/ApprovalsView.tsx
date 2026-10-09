import { Fragment, useState } from 'react';
import { Alert, Panel, useAction, useLoad } from './common';
import { jsonText, sameJson, type JsonObject } from './forms';
import { confidenceBand } from './framework';
import { runtimeJson, seg } from './http';
import { Bars, percent } from './metrics';
import { TenantChooser } from './tenants';

interface PendingItem {
  item_id: string;
  batch_id: string;
  status: string;
  confidence: number;
  data: JsonObject;
  metadata: JsonObject;
  created_at: string | null;
  schema_name: string | null;
  correction_template: JsonObject | null;
  /** A rejection merges the corrections instead of regenerating. */
  corrections_required: boolean;
  reasoning: string;
}

interface Decision {
  status: 'approved' | 'regenerated' | 'rejected';
  item: PendingItem;
}

export interface ReviewedItem {
  item_id: string;
  batch_id: string;
  status: string;
  confidence: number;
  query: string;
  data: JsonObject;
  created_at: string | null;
  reviewed_at: string | null;
  schema_name: string | null;
  reviewer: string | null;
  feedback: string | null;
  corrections: JsonObject;
  replacement_id: string | null;
  replacement_status: string | null;
}

interface ReviewHistory {
  approved: ReviewedItem[];
  rejected: ReviewedItem[];
}

export interface ReviewStats {
  total: number;
  pending: number;
  auto_approved: number;
  approved: number;
  rejected: number;
  approval_rate: number;
  average_confidence: Record<string, number>;
}

const SECTIONS = ['Pending', 'Approved', 'Rejected', 'Statistics'] as const;
type Section = (typeof SECTIONS)[number];

/** The statistics groups, in the order the page shows them. */
export const GROUP_LABELS: [keyof ReviewStats & string, string][] = [
  ['pending', 'Awaiting review'],
  ['auto_approved', 'Auto-approved'],
  ['approved', 'Approved'],
  ['rejected', 'Rejected'],
];

function approvalsPath(tenant: string): string {
  return `/admin/tenant/${seg(tenant)}/approvals`;
}

/** The fields of ``edited`` whose values differ from ``template``. */
export function changedCorrections(template: JsonObject, edited: JsonObject): JsonObject {
  return Object.fromEntries(Object.entries(edited).filter(([key, value]) => !sameJson(value, template[key])));
}

/** One line per mention the self-consistency check sampled, with how often
 * the samples agreed on it and whether that calls for review. */
export function selfConsistencyLines(metadata: JsonObject): string[] {
  const block = metadata.self_consistency as
    | { samples?: number; entities?: { text: string; type: string; agreement: number; needs_review: boolean }[] }
    | undefined;
  if (!block || !Array.isArray(block.entities)) return [];
  return block.entities.map(
    (entity) =>
      `Agreement (${block.samples} samples): ${entity.text} (${entity.type}) ${entity.agreement.toFixed(2)}` +
      (entity.needs_review ? ' — needs review' : ''),
  );
}

/** The generator's record of how it produced ``data``, or ``undefined``. */
export function generationMetadata(data: JsonObject): JsonObject | undefined {
  const metadata = data.metadata as JsonObject | undefined;
  const generation = metadata?._generation_metadata;
  return generation && typeof generation === 'object' && !Array.isArray(generation)
    ? (generation as JsonObject)
    : undefined;
}

/** How many times the generator retried ``data``; 0 when it never did. */
export function retryCount(data: JsonObject): number {
  const count = generationMetadata(data)?.retry_count;
  return typeof count === 'number' ? count : 0;
}

/** Each entity of ``data`` as ``text (TYPE)``. */
export function entityLabels(data: JsonObject): string[] {
  const entities = data.entities;
  if (!Array.isArray(entities)) return [];
  return entities.map((entity) => {
    const { text, type } = (entity ?? {}) as { text?: unknown; type?: unknown };
    return typeof text === 'string' && typeof type === 'string' ? `${text} (${type})` : JSON.stringify(entity);
  });
}

export type FieldKind = 'text' | 'number' | 'boolean' | 'json';

/** How the corrections editor edits a field holding ``value``. */
export function fieldKind(value: unknown): FieldKind {
  if (typeof value === 'string') return 'text';
  if (typeof value === 'number') return 'number';
  if (typeof value === 'boolean') return 'boolean';
  return 'json';
}

/** The editor's text for a field holding ``value``. */
export function fieldText(value: unknown): string {
  const kind = fieldKind(value);
  if (kind === 'text') return value as string;
  if (kind === 'number' || kind === 'boolean') return String(value);
  return jsonText(value);
}

/** ``text`` from the editor of field ``name`` as the value of its kind, or an
 * error naming the field. */
export function parseField(name: string, kind: FieldKind, text: string): unknown {
  if (kind === 'text') return text;
  if (kind === 'boolean') return text === 'true';
  if (kind === 'number') {
    const number = Number(text);
    if (!text.trim() || !Number.isFinite(number)) throw new Error(`${name} must be a number.`);
    return number;
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`${name} is not valid JSON.`);
  }
}

function decisionNotice(item: PendingItem, decision: Decision): string {
  if (decision.status === 'approved') return `Approved ${item.item_id} into the training dataset.`;
  if (decision.status === 'regenerated') {
    const kind = item.corrections_required ? 'corrected' : 'regenerated';
    return `Rejected ${item.item_id}; its ${kind} replacement ${decision.item.item_id} awaits review.`;
  }
  return `Rejected ${item.item_id}.`;
}

function rejectLabel(item: PendingItem): string {
  if (!item.correction_template) return 'Reject';
  return item.corrections_required ? 'Reject with corrections' : 'Reject and regenerate';
}

export function ApprovalsView() {
  const [tenant, setTenant] = useState('');
  const [reviewer, setReviewer] = useState('');
  const [section, setSection] = useState<Section>('Pending');
  const [version, setVersion] = useState(0);
  const [notice, setNotice] = useState('');
  const changed = (message: string) => {
    setNotice(message);
    setVersion((n) => n + 1);
  };
  return (
    <div className="ops-view">
      <TenantChooser
        action="Show review queue"
        onChoose={(chosen) => {
          setTenant(chosen);
          setNotice('');
        }}
      />
      {notice && <Alert tone="ok">{notice}</Alert>}
      {tenant && (
        <>
          <nav className="section-tabs" aria-label="Approval sections">
            {SECTIONS.map((name) => (
              <button key={name} aria-pressed={section === name} onClick={() => setSection(name)}>
                {name}
              </button>
            ))}
          </nav>
          {section === 'Pending' && (
            <Queue
              key={`${tenant}-${version}`}
              tenant={tenant}
              reviewer={reviewer}
              onReviewer={setReviewer}
              onDecided={changed}
            />
          )}
          {section === 'Approved' && <Approved key={`${tenant}-${version}`} tenant={tenant} />}
          {section === 'Rejected' && <Rejected key={`${tenant}-${version}`} tenant={tenant} onRegenerated={changed} />}
          {section === 'Statistics' && <Statistics key={`${tenant}-${version}`} tenant={tenant} />}
        </>
      )}
    </div>
  );
}

function useHistory(tenant: string) {
  return useLoad(
    (signal) => runtimeJson<ReviewHistory>(`${approvalsPath(tenant)}/history`, { signal }),
    [tenant],
  );
}

function Queue({
  tenant,
  reviewer,
  onReviewer,
  onDecided,
}: {
  tenant: string;
  reviewer: string;
  onReviewer: (reviewer: string) => void;
  onDecided: (notice: string) => void;
}) {
  const items = useLoad(
    (signal) => runtimeJson<{ items: PendingItem[] }>(approvalsPath(tenant), { signal }).then((body) => body.items),
    [tenant],
  );
  return (
    <>
      <Panel title={`Awaiting review in ${tenant}`} actions={<button onClick={items.reload}>Refresh</button>}>
        <label className="inline-form">
          Reviewer
          <input value={reviewer} onChange={(e) => onReviewer(e.target.value)} placeholder="you@example.com" />
        </label>
        {items.error && <Alert>{items.error}</Alert>}
        {items.data && (
          <p className="muted">
            {items.data.length === 0
              ? `Nothing awaits review in ${tenant}.`
              : `${items.data.length} ${items.data.length === 1 ? 'item awaits' : 'items await'} review.`}
          </p>
        )}
      </Panel>
      {(items.data ?? []).map((item) => (
        <ReviewItem key={`${item.batch_id}/${item.item_id}`} tenant={tenant} item={item} reviewer={reviewer} onDecided={onDecided} />
      ))}
    </>
  );
}

function ReviewItem({
  tenant,
  item,
  reviewer,
  onDecided,
}: {
  tenant: string;
  item: PendingItem;
  reviewer: string;
  onDecided: (notice: string) => void;
}) {
  const template = item.correction_template;
  const [rejecting, setRejecting] = useState(false);
  const action = useAction();
  const agreement = selfConsistencyLines(item.metadata);
  const entities = entityLabels(item.data);
  const generation = generationMetadata(item.data);
  const query = typeof item.data.query === 'string' ? item.data.query : '';
  const band = confidenceBand(item.confidence);
  const decide = (body: JsonObject) =>
    action.run(async () => {
      if (!reviewer.trim()) throw new Error('Enter your name as the reviewer first.');
      const decision = await runtimeJson<Decision>(
        `${approvalsPath(tenant)}/${seg(item.batch_id)}/${seg(item.item_id)}`,
        { method: 'POST', body: { ...body, reviewer: reviewer.trim() } },
      );
      onDecided(decisionNotice(item, decision));
    });
  return (
    <Panel title={`Review ${item.item_id}`}>
      <dl className="facts">
        <dt>Batch</dt>
        <dd>{item.batch_id}</dd>
        <dt>Schema</dt>
        <dd>{item.schema_name ?? '—'}</dd>
        <dt>Status</dt>
        <dd>{item.status}</dd>
        <dt>Confidence</dt>
        <dd>{item.confidence.toFixed(2)}</dd>
        <dt>Band</dt>
        <dd className={`band ${band.tone}`}>{band.label}</dd>
        <dt>Retry count</dt>
        <dd>{retryCount(item.data)}</dd>
        <dt>Created</dt>
        <dd>{item.created_at ?? '—'}</dd>
        {query && (
          <>
            <dt>Query</dt>
            <dd>{query}</dd>
          </>
        )}
        {item.reasoning && (
          <>
            <dt>Reasoning</dt>
            <dd>{item.reasoning}</dd>
          </>
        )}
        {entities.length > 0 && (
          <>
            <dt>Entities</dt>
            <dd>
              <ul aria-label="Entities">
                {entities.map((entity) => (
                  <li key={entity}>{entity}</li>
                ))}
              </ul>
            </dd>
          </>
        )}
        {agreement.length > 0 && (
          <>
            <dt>Self-consistency</dt>
            <dd>
              <ul aria-label="Self-consistency">
                {agreement.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </dd>
          </>
        )}
      </dl>
      {generation && (
        <details>
          <summary>Generation metadata</summary>
          <pre aria-label="Generation metadata">{jsonText(generation)}</pre>
        </details>
      )}
      <details>
        <summary>Example as JSON</summary>
        <pre aria-label="Example">{jsonText(item.data)}</pre>
      </details>
      {!template && (
        <p className="muted">No example schema describes this item, so it can be approved or rejected but not corrected.</p>
      )}
      {rejecting ? (
        <RejectionForm
          item={item}
          pending={action.pending}
          onSubmit={(body) => decide({ approved: false, ...body })}
          onCancel={() => {
            setRejecting(false);
            action.clear();
          }}
        />
      ) : (
        <div className="inline-form">
          <button disabled={action.pending} onClick={() => decide({ approved: true })}>
            Approve
          </button>
          <button className="danger" disabled={action.pending} onClick={() => setRejecting(true)}>
            {rejectLabel(item)}
          </button>
        </div>
      )}
      {action.pending && <span className="muted">Recording the decision…</span>}
      {action.error && <Alert>{action.error}</Alert>}
    </Panel>
  );
}

function RejectionForm({
  item,
  pending,
  onSubmit,
  onCancel,
}: {
  item: PendingItem;
  pending: boolean;
  onSubmit: (body: JsonObject) => void;
  onCancel: () => void;
}) {
  const template = item.correction_template;
  const [feedback, setFeedback] = useState('');
  const [fields, setFields] = useState<Record<string, string>>(() =>
    Object.fromEntries(Object.entries(template ?? {}).map(([name, value]) => [name, fieldText(value)])),
  );
  const [invalid, setInvalid] = useState('');
  const regenerates = Boolean(template) && !item.corrections_required;
  return (
    <form
      className="stacked-form"
      aria-label={`Reject ${item.item_id}`}
      onSubmit={(e) => {
        e.preventDefault();
        setInvalid('');
        const body: JsonObject = { feedback };
        if (template) {
          try {
            const edited = Object.fromEntries(
              Object.entries(template).map(([name, value]) => [name, parseField(name, fieldKind(value), fields[name])]),
            );
            const changed = changedCorrections(template, edited);
            if (Object.keys(changed).length) body.corrections = changed;
          } catch (error) {
            setInvalid(error instanceof Error ? error.message : String(error));
            return;
          }
        }
        onSubmit(body);
      }}
    >
      <label>
        Feedback
        <textarea
          rows={2}
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          placeholder={regenerates ? 'what the regenerated example should fix' : 'optional'}
        />
      </label>
      {template && (
        <fieldset>
          <legend>{item.schema_name} corrections</legend>
          {Object.entries(template).map(([name, value]) => {
            const kind = fieldKind(value);
            const set = (text: string) => setFields((current) => ({ ...current, [name]: text }));
            if (kind === 'boolean')
              return (
                <label key={name} className="check">
                  <input
                    type="checkbox"
                    aria-label={name}
                    checked={fields[name] === 'true'}
                    onChange={(e) => set(String(e.target.checked))}
                  />
                  {name}
                </label>
              );
            return (
              <label key={name}>
                {name}
                {kind === 'json' ? (
                  <textarea aria-label={name} rows={4} value={fields[name]} onChange={(e) => set(e.target.value)} />
                ) : (
                  <input
                    aria-label={name}
                    inputMode={kind === 'number' ? 'decimal' : undefined}
                    value={fields[name]}
                    onChange={(e) => set(e.target.value)}
                  />
                )}
              </label>
            );
          })}
        </fieldset>
      )}
      <div className="inline-form">
        <button type="submit" className="danger" disabled={pending}>
          Submit rejection
        </button>
        <button type="button" disabled={pending} onClick={onCancel}>
          Cancel
        </button>
      </div>
      {invalid && <Alert>{invalid}</Alert>}
    </form>
  );
}

function Approved({ tenant }: { tenant: string }) {
  const history = useHistory(tenant);
  const items = history.data?.approved;
  return (
    <Panel title={`Approved items of ${tenant}`} actions={<button onClick={history.reload}>Refresh</button>}>
      {history.error && <Alert>{history.error}</Alert>}
      {items && items.length === 0 && <p className="muted">No approved items in {tenant}.</p>}
      {items && items.length > 0 && (
        <>
          <p className="muted">
            {items.length} {items.length === 1 ? 'item' : 'items'} approved.
          </p>
          <table aria-label="Approved items">
            <thead>
              <tr>
                <th>Item</th>
                <th>Query</th>
                <th>Confidence</th>
                <th>Status</th>
                <th>Reviewer</th>
                <th>Approved at</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={`${item.batch_id}/${item.item_id}`}>
                  <td>{item.item_id}</td>
                  <td>{item.query || '—'}</td>
                  <td>{item.confidence.toFixed(2)}</td>
                  <td>{item.status}</td>
                  <td>{item.reviewer ?? '—'}</td>
                  <td>{item.reviewed_at ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Panel>
  );
}

/** Whether the page offers to regenerate ``item``: rejected, described by a
 * schema, and never replaced. */
export function regenerable(item: ReviewedItem): boolean {
  return item.schema_name !== null && item.replacement_id === null;
}

function Rejected({ tenant, onRegenerated }: { tenant: string; onRegenerated: (notice: string) => void }) {
  const history = useHistory(tenant);
  const items = history.data?.rejected;
  return (
    <Panel title={`Rejected items of ${tenant}`} actions={<button onClick={history.reload}>Refresh</button>}>
      {history.error && <Alert>{history.error}</Alert>}
      {items && items.length === 0 && <p className="muted">No rejected items in {tenant}.</p>}
      {items && items.length > 0 && (
        <>
          <p className="muted">
            {items.length} {items.length === 1 ? 'item' : 'items'} rejected.
          </p>
          <table aria-label="Rejected items">
            <thead>
              <tr>
                <th>Item</th>
                <th>Query</th>
                <th>Feedback</th>
                <th>Corrections</th>
                <th>Reviewer</th>
                <th>Replacement</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <RejectedRow key={`${item.batch_id}/${item.item_id}`} tenant={tenant} item={item} onRegenerated={onRegenerated} />
              ))}
            </tbody>
          </table>
        </>
      )}
    </Panel>
  );
}

function RejectedRow({
  tenant,
  item,
  onRegenerated,
}: {
  tenant: string;
  item: ReviewedItem;
  onRegenerated: (notice: string) => void;
}) {
  const action = useAction();
  const corrections = Object.keys(item.corrections).length ? JSON.stringify(item.corrections) : '—';
  return (
    <tr>
      <td>{item.item_id}</td>
      <td>{item.query || '—'}</td>
      <td>{item.feedback || '—'}</td>
      <td>
        <code>{corrections}</code>
      </td>
      <td>{item.reviewer ?? '—'}</td>
      <td>{item.replacement_id ? `${item.replacement_id} (${item.replacement_status})` : '—'}</td>
      <td>
        {regenerable(item) && (
          <button
            disabled={action.pending}
            onClick={() =>
              action.run(async () => {
                const decision = await runtimeJson<Decision>(
                  `${approvalsPath(tenant)}/${seg(item.batch_id)}/${seg(item.item_id)}/regenerate`,
                  { method: 'POST' },
                );
                onRegenerated(`Regenerated ${item.item_id} as ${decision.item.item_id}; it awaits review.`);
              })
            }
          >
            {action.pending ? 'Regenerating…' : 'Regenerate'}
          </button>
        )}
        {action.error && <Alert>{action.error}</Alert>}
      </td>
    </tr>
  );
}

/** The mean confidence of each group that holds items, labelled. */
export function confidenceBars(stats: ReviewStats): { label: string; value: number }[] {
  return GROUP_LABELS.filter(([group]) => group in stats.average_confidence).map(([group, label]) => ({
    label,
    value: stats.average_confidence[group],
  }));
}

function Statistics({ tenant }: { tenant: string }) {
  const stats = useLoad(
    (signal) => runtimeJson<ReviewStats>(`${approvalsPath(tenant)}/stats`, { signal }),
    [tenant],
  );
  const data = stats.data;
  return (
    <Panel title={`Review statistics of ${tenant}`} actions={<button onClick={stats.reload}>Refresh</button>}>
      {stats.error && <Alert>{stats.error}</Alert>}
      {data && data.total === 0 && <p className="muted">No items in {tenant} yet.</p>}
      {data && data.total > 0 && (
        <>
          <dl className="facts" aria-label="Review totals">
            <dt>Total items</dt>
            <dd>{data.total}</dd>
            {GROUP_LABELS.map(([group, label]) => (
              <Fragment key={group}>
                <dt>{label}</dt>
                <dd>{data[group] as number}</dd>
              </Fragment>
            ))}
            <dt>Approval rate</dt>
            <dd>{percent(data.approval_rate)}</dd>
          </dl>
          <Bars
            title="Average confidence by status"
            entries={confidenceBars(data)}
            format={(value) => value.toFixed(2)}
          />
        </>
      )}
    </Panel>
  );
}
