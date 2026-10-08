import { useState } from 'react';
import { Alert, Panel, useAction, useLoad } from './common';
import { jsonText, parseJsonObject, sameJson, type JsonObject } from './forms';
import { runtimeJson, seg } from './http';
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
  const [version, setVersion] = useState(0);
  const [notice, setNotice] = useState('');
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
        <Queue
          key={`${tenant}-${version}`}
          tenant={tenant}
          reviewer={reviewer}
          onReviewer={setReviewer}
          onDecided={(message) => {
            setNotice(message);
            setVersion((n) => n + 1);
          }}
        />
      )}
    </div>
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
  const [feedback, setFeedback] = useState('');
  const [corrections, setCorrections] = useState(jsonText(template));
  const action = useAction();
  const agreement = selfConsistencyLines(item.metadata);
  const decide = (approved: boolean) =>
    action.run(async () => {
      if (!reviewer.trim()) throw new Error('Enter your name as the reviewer first.');
      const body: JsonObject = { approved, reviewer: reviewer.trim(), feedback };
      if (!approved && template) {
        const edited = parseJsonObject('Corrections', corrections);
        const changed = edited ? changedCorrections(template, edited) : {};
        if (Object.keys(changed).length) body.corrections = changed;
      }
      const decision = await runtimeJson<Decision>(
        `${approvalsPath(tenant)}/${seg(item.batch_id)}/${seg(item.item_id)}`,
        { method: 'POST', body },
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
        <dt>Created</dt>
        <dd>{item.created_at ?? '—'}</dd>
        {item.reasoning && (
          <>
            <dt>Reasoning</dt>
            <dd>{item.reasoning}</dd>
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
        <dt>Example</dt>
        <dd>
          <pre aria-label="Example">{jsonText(item.data)}</pre>
        </dd>
      </dl>
      <form
        className="stacked-form"
        aria-label={`Decide on ${item.item_id}`}
        onSubmit={(e) => {
          e.preventDefault();
          decide(true);
        }}
      >
        <label>
          Feedback
          <textarea rows={2} value={feedback} onChange={(e) => setFeedback(e.target.value)} placeholder="needed to reject" />
        </label>
        {template ? (
          <label>
            Corrections (JSON)
            <textarea rows={6} value={corrections} onChange={(e) => setCorrections(e.target.value)} />
          </label>
        ) : (
          <p className="muted">No example schema describes this item, so it can be approved or rejected but not corrected.</p>
        )}
        <div className="inline-form">
          <button type="submit" disabled={action.pending}>
            Approve
          </button>
          <button type="button" className="danger" disabled={action.pending} onClick={() => decide(false)}>
            {rejectLabel(item)}
          </button>
          {action.pending && <span className="muted">Recording the decision…</span>}
        </div>
        {action.error && <Alert>{action.error}</Alert>}
      </form>
    </Panel>
  );
}
