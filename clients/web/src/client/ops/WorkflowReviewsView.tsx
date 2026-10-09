import { useState } from 'react';
import { Alert, Panel, useAction, useLoad } from './common';
import { ViewErrorBoundary } from './ErrorBoundary';
import { failureDetail, runtimeJson, seg } from './http';
import { LookbackSelect } from './metrics';
import { TenantChooser } from './tenants';

interface Review {
  annotator: string;
  label: string;
  score: number;
  annotation_source: string;
  pattern_is_optimal: boolean;
  agents_are_correct: boolean;
  execution_order_is_optimal: boolean;
  improvement_notes: string | null;
}

interface Workflow {
  span_id: string;
  start_time: string;
  query: string;
  workflow_id: string;
  pattern: string;
  agent_sequence: string[];
  execution_order: string[];
  execution_time: number;
  tasks_completed: number;
  success: boolean;
  error_summary: string | null;
  review: Review | null;
}

const QUALITY_LABELS = ['failed', 'poor', 'acceptable', 'good', 'excellent'];
const PATTERNS = ['parallel', 'sequential', 'conditional', 'mixed'];
const VERDICTS = ['Yes', 'No', 'Unsure'] as const;
type Verdict = (typeof VERDICTS)[number];
/** The most workflows the list route returns. */
const MAX_WORKFLOWS = 500;

/** The non-blank entries of ``text`` split on commas and line breaks. */
export function splitList(text: string): string[] {
  return text
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

/** The loaded workflows with each review this page saved in place of the
 * loaded one, which the telemetry backend may not serve yet. */
export function withSavedReviews<T extends { span_id: string }>(loaded: T[], saved: Record<string, T>): T[] {
  return loaded.map((workflow) => saved[workflow.span_id] ?? workflow);
}

function workflowsPath(tenant: string): string {
  return `/admin/tenant/${seg(tenant)}/orchestration-workflows`;
}

export function WorkflowReviewsView() {
  return (
    <ViewErrorBoundary view="Workflow reviews">
      <WorkflowReviews />
    </ViewErrorBoundary>
  );
}

function WorkflowReviews() {
  const [tenant, setTenant] = useState('');
  const [lookback, setLookback] = useState(24);
  const [limit, setLimit] = useState(50);
  const [reviewer, setReviewer] = useState('');
  const [selected, setSelected] = useState<string | null>(null);
  const [saved, setSaved] = useState<Record<string, Workflow>>({});
  const [notice, setNotice] = useState('');
  return (
    <div className="ops-view">
      <p className="muted">
        Review orchestration workflows to improve future routing and orchestration decisions. Your reviews become
        ground truth for optimization.
      </p>
      <TenantChooser
        action="Show workflows"
        onChoose={(chosen) => {
          setTenant(chosen);
          setSelected(null);
          setSaved({});
          setNotice('');
        }}
      />
      {notice && <Alert tone="ok">{notice}</Alert>}
      {tenant && (
        <Workflows
          key={`${tenant}-${lookback}-${limit}`}
          tenant={tenant}
          saved={saved}
          lookback={lookback}
          onLookback={setLookback}
          limit={limit}
          onLimit={setLimit}
          reviewer={reviewer}
          onReviewer={setReviewer}
          selected={selected}
          onSelect={setSelected}
          onReviewed={(reviewed) => {
            setNotice(`Saved the review of ${reviewed.workflow_id}: ${reviewed.review?.label}.`);
            setSaved((previous) => ({ ...previous, [reviewed.span_id]: reviewed }));
            setSelected(null);
          }}
        />
      )}
    </div>
  );
}

function Workflows({
  tenant,
  saved,
  lookback,
  onLookback,
  limit,
  onLimit,
  reviewer,
  onReviewer,
  selected,
  onSelect,
  onReviewed,
}: {
  tenant: string;
  saved: Record<string, Workflow>;
  lookback: number;
  onLookback: (hours: number) => void;
  limit: number;
  onLimit: (limit: number) => void;
  reviewer: string;
  onReviewer: (reviewer: string) => void;
  selected: string | null;
  onSelect: (spanId: string) => void;
  onReviewed: (reviewed: Workflow) => void;
}) {
  const workflows = useLoad(
    (signal) =>
      runtimeJson<{ workflows: Workflow[] }>(`${workflowsPath(tenant)}?lookback_hours=${lookback}&limit=${limit}`, {
        signal,
      }).then((body) => body.workflows),
    [tenant, lookback, limit],
  );
  const shown = workflows.data && withSavedReviews(workflows.data, saved);
  const chosen = shown?.find((workflow) => workflow.span_id === selected);
  return (
    <>
      <Panel title={`Workflows of ${tenant}`} actions={<button onClick={workflows.reload}>Refresh</button>}>
        <div className="inline-form">
          <LookbackSelect value={lookback} onChange={onLookback} />
          <label>
            Max workflows
            <select value={limit} onChange={(e) => onLimit(Number(e.target.value))}>
              {[1, 5, 10, 20, 50, 100, 200, MAX_WORKFLOWS].map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <label>
            Reviewer
            <input value={reviewer} onChange={(e) => onReviewer(e.target.value)} placeholder="you@example.com" />
          </label>
        </div>
        {workflows.error && <Alert>{workflows.error}</Alert>}
        {shown && shown.length === 0 && (
          <p className="muted">No orchestration workflows in this window.</p>
        )}
        {shown && shown.length > 0 && (
          <p className="muted">
            Found {shown.length} workflow{shown.length === 1 ? '' : 's'}
            {shown.length === limit ? `, the newest ${limit}` : ''}.
          </p>
        )}
        {shown && shown.length > 0 && (
          <table aria-label="Workflows">
            <thead>
              <tr>
                <th>Started</th>
                <th>Query</th>
                <th>Pattern</th>
                <th>Agents</th>
                <th>Time</th>
                <th>Outcome</th>
                <th>Review</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {shown.map((workflow) => (
                <tr key={workflow.span_id} className={workflow.span_id === selected ? 'selected' : undefined}>
                  <td>{new Date(workflow.start_time).toLocaleString()}</td>
                  <td>{workflow.query}</td>
                  <td>{workflow.pattern}</td>
                  <td>{workflow.agent_sequence.join(' → ')}</td>
                  <td>{workflow.execution_time.toFixed(2)}s</td>
                  <td>{workflow.success ? 'succeeded' : `failed: ${workflow.error_summary ?? ''}`}</td>
                  <td>
                    {workflow.review
                      ? `${workflow.review.label} (${workflow.review.score.toFixed(2)}) by ${workflow.review.annotator}`
                      : '—'}
                  </td>
                  <td>
                    <button aria-label={`Review ${workflow.workflow_id}`} onClick={() => onSelect(workflow.span_id)}>
                      Review
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
      {chosen && <ReviewForm key={chosen.span_id} tenant={tenant} workflow={chosen} reviewer={reviewer} onReviewed={onReviewed} />}
    </>
  );
}

function ReviewForm({
  tenant,
  workflow,
  reviewer,
  onReviewed,
}: {
  tenant: string;
  workflow: Workflow;
  reviewer: string;
  onReviewed: (reviewed: Workflow) => void;
}) {
  const [label, setLabel] = useState('');
  const [score, setScore] = useState('');
  const [patternVerdict, setPatternVerdict] = useState<Verdict>('Yes');
  const [suggestedPattern, setSuggestedPattern] = useState('');
  const [patternFeedback, setPatternFeedback] = useState('');
  const [agentsCorrect, setAgentsCorrect] = useState(true);
  const [missing, setMissing] = useState('');
  const [unnecessary, setUnnecessary] = useState('');
  const [orderOptimal, setOrderOptimal] = useState(true);
  const [suggestedOrder, setSuggestedOrder] = useState('');
  const [orderFeedback, setOrderFeedback] = useState('');
  const [wentWell, setWentWell] = useState('');
  const [wentWrong, setWentWrong] = useState('');
  const [notes, setNotes] = useState('');
  const action = useAction();
  const [detail, setDetail] = useState<string | null>(null);
  const optional = (text: string) => (text.trim() ? text.trim() : null);
  const suggestsPattern = patternVerdict === 'No';
  return (
    <Panel title={`Review ${workflow.workflow_id}`}>
      <dl className="facts">
        <dt>Query</dt>
        <dd>{workflow.query}</dd>
        <dt>Pattern</dt>
        <dd>{workflow.pattern}</dd>
        <dt>Agent sequence</dt>
        <dd>{workflow.agent_sequence.join(', ')}</dd>
        <dt>Execution order</dt>
        <dd>{workflow.execution_order.join(', ')}</dd>
        <dt>Tasks completed</dt>
        <dd>{workflow.tasks_completed}</dd>
        <dt>Outcome</dt>
        <dd>{workflow.success ? 'succeeded' : `failed: ${workflow.error_summary ?? ''}`}</dd>
      </dl>
      <form
        className="stacked-form"
        aria-label={`Review of ${workflow.workflow_id}`}
        onSubmit={(e) => {
          e.preventDefault();
          setDetail(null);
          action.run(async () => {
            if (!reviewer.trim()) throw new Error('Enter your name as the reviewer first.');
            const body = {
              start_time: workflow.start_time,
              annotator: reviewer.trim(),
              quality_label: label,
              quality_score: Number(score),
              pattern_is_optimal: patternVerdict === 'Yes',
              suggested_pattern: suggestsPattern ? optional(suggestedPattern) : null,
              pattern_feedback: suggestsPattern ? optional(patternFeedback) : null,
              agents_are_correct: agentsCorrect,
              missing_agents: agentsCorrect ? [] : splitList(missing),
              unnecessary_agents: agentsCorrect ? [] : splitList(unnecessary),
              execution_order_is_optimal: orderOptimal,
              suggested_execution_order: orderOptimal ? null : splitList(suggestedOrder),
              execution_order_feedback: orderOptimal ? null : optional(orderFeedback),
              what_went_well: optional(wentWell),
              what_went_wrong: optional(wentWrong),
              improvement_notes: optional(notes),
            };
            try {
              onReviewed(
                await runtimeJson<Workflow>(`${workflowsPath(tenant)}/${seg(workflow.span_id)}/annotation`, {
                  method: 'POST',
                  body,
                }),
              );
            } catch (error) {
              setDetail(failureDetail(error));
              throw error;
            }
          });
        }}
      >
        <div className="inline-form">
          <label>
            Quality
            <select required value={label} onChange={(e) => setLabel(e.target.value)}>
              <option value="" disabled>
                Choose a quality
              </option>
              {QUALITY_LABELS.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <label>
            Score (0–1)
            <input required type="number" min={0} max={1} step={0.05} value={score} onChange={(e) => setScore(e.target.value)} />
          </label>
        </div>
        <fieldset>
          <legend>Was the pattern optimal?</legend>
          {VERDICTS.map((verdict) => (
            <label key={verdict} className="check">
              <input
                type="radio"
                name="pattern-verdict"
                checked={patternVerdict === verdict}
                onChange={() => setPatternVerdict(verdict)}
              />
              {verdict}
            </label>
          ))}
        </fieldset>
        {suggestsPattern && (
          <div className="inline-form">
            <label>
              Suggested pattern
              <select value={suggestedPattern} onChange={(e) => setSuggestedPattern(e.target.value)}>
                <option value="">None</option>
                {PATTERNS.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Why that pattern
              <input value={patternFeedback} onChange={(e) => setPatternFeedback(e.target.value)} />
            </label>
          </div>
        )}
        <label className="check">
          <input type="checkbox" checked={agentsCorrect} onChange={(e) => setAgentsCorrect(e.target.checked)} />
          The right agents were used
        </label>
        {!agentsCorrect && (
          <div className="inline-form">
            <label>
              Missing agents
              <input value={missing} onChange={(e) => setMissing(e.target.value)} placeholder="comma-separated" />
            </label>
            <label>
              Unnecessary agents
              <input value={unnecessary} onChange={(e) => setUnnecessary(e.target.value)} placeholder="comma-separated" />
            </label>
          </div>
        )}
        <label className="check">
          <input type="checkbox" checked={orderOptimal} onChange={(e) => setOrderOptimal(e.target.checked)} />
          The execution order was optimal
        </label>
        {!orderOptimal && (
          <>
            <label>
              Suggested order
              <textarea rows={3} value={suggestedOrder} onChange={(e) => setSuggestedOrder(e.target.value)} placeholder="one agent per line" />
            </label>
            <label>
              Why that order
              <input value={orderFeedback} onChange={(e) => setOrderFeedback(e.target.value)} />
            </label>
          </>
        )}
        <label>
          What went well
          <textarea rows={2} value={wentWell} onChange={(e) => setWentWell(e.target.value)} />
        </label>
        <label>
          What went wrong
          <textarea rows={2} value={wentWrong} onChange={(e) => setWentWrong(e.target.value)} />
        </label>
        <label>
          Improvement notes
          <textarea rows={2} value={notes} onChange={(e) => setNotes(e.target.value)} />
        </label>
        <button type="submit" disabled={action.pending}>
          {action.pending ? 'Saving…' : 'Save review'}
        </button>
        {action.error && <Alert>{action.error}</Alert>}
        {action.error && detail && (
          <details open aria-label="Failure details">
            <summary>Details</summary>
            <pre className="traceback">{detail}</pre>
          </details>
        )}
      </form>
    </Panel>
  );
}
