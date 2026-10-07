import { useState } from 'react';
import { Alert, Panel, useAction, useLoad } from './common';
import { runtimeJson, seg } from './http';

interface AnnotationRequest {
  span_id: string;
  timestamp: string;
  query: string;
  chosen_agent: string;
  routing_confidence: number;
  outcome: string;
  priority: string;
  reason: string;
  status: string;
  assigned_to: string | null;
  sla_deadline: string | null;
  agent_type: string;
  tenant_id: string | null;
}

interface QueueSnapshot {
  statistics: { total: number; by_status: Record<string, number>; by_priority: Record<string, number> };
  pending: AnnotationRequest[];
  assigned: AnnotationRequest[];
  expired: AnnotationRequest[];
}

const QUEUE = '/agents/annotations/queue';
/** The most requests of each list the queue route returns. */
const LIST_LIMIT = 50;
const STATUSES = ['pending', 'assigned', 'expired', 'completed'] as const;

export function queueSummary(byStatus: Record<string, number>): string {
  return `${STATUSES.map((status) => `${byStatus[status] ?? 0} ${status}`).join(', ')}.`;
}

export function AnnotationsView() {
  const [reviewer, setReviewer] = useState('');
  const [version, setVersion] = useState(0);
  const [notice, setNotice] = useState('');
  const queue = useLoad((signal) => runtimeJson<QueueSnapshot>(QUEUE, { signal }), [version]);
  const labels = useLoad(
    (signal) => runtimeJson<{ labels: string[] }>('/agents/annotations/labels', { signal }).then((body) => body.labels),
    [],
  );
  const changed = (message: string) => {
    setNotice(message);
    setVersion((n) => n + 1);
  };
  const byStatus = queue.data?.statistics.by_status ?? {};
  const capped = STATUSES.some((status) => status !== 'completed' && (byStatus[status] ?? 0) > LIST_LIMIT);
  const lists = { reviewer, labels: labels.data ?? [], onChanged: changed };
  return (
    <div className="ops-view">
      <Panel title="Annotation queue" actions={<button onClick={() => setVersion((n) => n + 1)}>Refresh</button>}>
        <label className="inline-form">
          Reviewer
          <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="you@example.com" />
        </label>
        {queue.error && <Alert>{queue.error}</Alert>}
        {labels.error && <Alert>Review labels are unavailable: {labels.error}</Alert>}
        {queue.data && (
          <p className="muted">
            {queueSummary(byStatus)}
            {capped && ` Each list shows its first ${LIST_LIMIT}.`}
          </p>
        )}
      </Panel>
      {notice && <Alert tone="ok">{notice}</Alert>}
      {queue.data && (
        <>
          <RequestList title="Pending" requests={queue.data.pending} actions={['assign', 'annotate']} {...lists} />
          <RequestList title="Assigned" requests={queue.data.assigned} actions={['annotate']} {...lists} />
          <RequestList title="Expired" requests={queue.data.expired} actions={[]} {...lists} />
        </>
      )}
    </div>
  );
}

type Action = 'assign' | 'annotate';

function RequestList({
  title,
  requests,
  actions,
  reviewer,
  labels,
  onChanged,
}: {
  title: string;
  requests: AnnotationRequest[];
  actions: Action[];
  reviewer: string;
  labels: string[];
  onChanged: (notice: string) => void;
}) {
  const [annotating, setAnnotating] = useState<string | null>(null);
  const assign = useAction();
  const showsAssignee = title !== 'Pending';
  return (
    <Panel title={title}>
      {requests.length === 0 ? (
        <p className="muted">No {title.toLowerCase()} requests.</p>
      ) : (
        <table aria-label={`${title} requests`}>
          <thead>
            <tr>
              <th>Span</th>
              <th>Tenant</th>
              <th>Query</th>
              <th>Agent</th>
              <th>Confidence</th>
              <th>Outcome</th>
              <th>Priority</th>
              <th>Reason</th>
              {showsAssignee && <th>Assigned to</th>}
              {showsAssignee && <th>Due</th>}
              {actions.length > 0 && <th />}
            </tr>
          </thead>
          <tbody>
            {requests.map((request) => (
              <tr key={request.span_id} className={annotating === request.span_id ? 'selected' : undefined}>
                <td>{request.span_id}</td>
                <td>{request.tenant_id ?? '—'}</td>
                <td>{request.query}</td>
                <td>{request.chosen_agent}</td>
                <td>{request.routing_confidence.toFixed(2)}</td>
                <td>{request.outcome}</td>
                <td>{request.priority}</td>
                <td>{request.reason}</td>
                {showsAssignee && <td>{request.assigned_to ?? '—'}</td>}
                {showsAssignee && <td>{request.sla_deadline ?? '—'}</td>}
                {actions.length > 0 && (
                  <td>
                    <span className="confirm">
                      {actions.includes('assign') && (
                        <button
                          aria-label={`Assign ${request.span_id} to me`}
                          disabled={assign.pending}
                          onClick={() =>
                            assign.run(async () => {
                              if (!reviewer.trim()) throw new Error('Enter your name as the reviewer first.');
                              await runtimeJson(`${QUEUE}/${seg(request.span_id)}/assign`, {
                                method: 'POST',
                                body: { reviewer: reviewer.trim() },
                              });
                              onChanged(`Assigned ${request.span_id} to ${reviewer.trim()}.`);
                            })
                          }
                        >
                          Assign to me
                        </button>
                      )}
                      {actions.includes('annotate') && (
                        <button aria-label={`Annotate ${request.span_id}`} onClick={() => setAnnotating(request.span_id)}>
                          Annotate
                        </button>
                      )}
                    </span>
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {assign.error && <Alert>{assign.error}</Alert>}
      {annotating && (
        <AnnotateForm
          key={annotating}
          spanId={annotating}
          reviewer={reviewer}
          labels={labels}
          onCancel={() => setAnnotating(null)}
          onDone={onChanged}
        />
      )}
    </Panel>
  );
}

function AnnotateForm({
  spanId,
  reviewer,
  labels,
  onCancel,
  onDone,
}: {
  spanId: string;
  reviewer: string;
  labels: string[];
  onCancel: () => void;
  onDone: (notice: string) => void;
}) {
  const [label, setLabel] = useState('');
  const [reasoning, setReasoning] = useState('');
  const action = useAction();
  return (
    <form
      className="stacked-form"
      aria-label={`Annotate ${spanId}`}
      onSubmit={(e) => {
        e.preventDefault();
        action.run(async () => {
          if (!reviewer.trim()) throw new Error('Enter your name as the reviewer first.');
          const result = await runtimeJson<{ persisted: boolean }>(`${QUEUE}/${seg(spanId)}/complete`, {
            method: 'POST',
            body: { label, reasoning, annotator: reviewer.trim() },
          });
          onDone(
            result.persisted
              ? `Labelled ${spanId} ${label}.`
              : `Labelled ${spanId} ${label} in the queue only: the request names no tenant to store it for.`,
          );
        });
      }}
    >
      <label>
        Label
        <select required value={label} onChange={(e) => setLabel(e.target.value)}>
          <option value="" disabled>
            Choose a label
          </option>
          {labels.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
      </label>
      <label>
        Reasoning
        <textarea rows={2} value={reasoning} onChange={(e) => setReasoning(e.target.value)} />
      </label>
      <div className="inline-form">
        <button type="submit" disabled={action.pending}>
          {action.pending ? 'Saving…' : 'Save label'}
        </button>
        <button type="button" onClick={onCancel}>
          Cancel
        </button>
      </div>
      {action.error && <Alert>{action.error}</Alert>}
    </form>
  );
}
