import { useMemo, useState } from 'react';
import { Alert, Panel, useAction, useLoad } from './common';
import { ViewErrorBoundary } from './ErrorBoundary';
import { runtimeJson, seg } from './http';
import { LookbackSelect, percent } from './metrics';
import { Plot } from './Plot';
import {
  approvable,
  LABEL_FILTERS,
  MAX_LOOKBACK_HOURS,
  PRIORITIES,
  ROUTING_LOOKBACKS,
  calibrationPoints,
  decisionsPerHourByAgent,
  initialReviewLabel,
  labelState,
  labelledBy,
  lookbackHours,
  scoreColor,
  shownCandidates,
  successRatePerHour,
  withReviews,
  type AgentRouting,
  type AnnotationCandidate,
  type DecisionLabel,
  type LabelState,
  type Priority,
  type RoutingDecision,
  type RoutingDecisions,
} from './routing';
import { TenantChooser } from './tenants';

const when = (iso: string) => new Date(iso).toLocaleString();
const ms = (value: number) => `${value.toFixed(1)} ms`;
const optional = (value: number | null, format: (value: number) => string) => (value === null ? '—' : format(value));

function decisionsPath(tenant: string): string {
  return `/admin/tenant/${seg(tenant)}/routing-decisions`;
}

interface LabelStatistics {
  total: number;
  human_reviewed: number;
  pending_review: number;
  by_label: Record<string, number>;
}

export function RoutingView() {
  const [tenant, setTenant] = useState('');
  const [lookback, setLookback] = useState(24);
  return (
    <div className="ops-view">
      <ViewErrorBoundary view="Routing evaluation">
        <TenantChooser action="Show decisions" onChoose={setTenant} />
        {tenant && <Routing key={`${tenant}-${lookback}`} tenant={tenant} lookback={lookback} onLookback={setLookback} />}
      </ViewErrorBoundary>
    </div>
  );
}

function Routing({
  tenant,
  lookback,
  onLookback,
}: {
  tenant: string;
  lookback: number;
  onLookback: (hours: number) => void;
}) {
  const [notice, setNotice] = useState('');
  const [reviewer, setReviewer] = useState('');
  const [relabelling, setRelabelling] = useState<string | null>(null);
  const [version, setVersion] = useState(0);
  // Decisions as the review routes answered them; Phoenix serves a new
  // label to the list only after a short indexing delay.
  const [reviewed, setReviewed] = useState<Record<string, RoutingDecision>>({});
  const routing = useLoad(
    (signal) => runtimeJson<RoutingDecisions>(`${decisionsPath(tenant)}?lookback_hours=${lookback}`, { signal }),
    [tenant, lookback],
  );
  const labels = useLoad(
    (signal) => runtimeJson<{ labels: string[] }>('/agents/annotations/labels', { signal }).then((body) => body.labels),
    [],
  );
  const changed = (message: string, decision: RoutingDecision) => {
    setNotice(message);
    setRelabelling(null);
    setReviewed((previous) => ({ ...previous, [decision.span_id!]: decision }));
  };
  const refresh = () => {
    setReviewed({});
    setVersion((n) => n + 1);
    routing.reload();
  };
  const data = routing.data;
  const decisions = useMemo(() => withReviews(data?.decisions ?? [], reviewed), [data, reviewed]);
  const relabel = (decision: RoutingDecision) =>
    setRelabelling(relabelling === decision.span_id ? null : decision.span_id);
  const labelling = decisions.find((decision) => decision.span_id === relabelling);
  return (
    <>
      <Panel title={`Routing decisions of ${tenant}`} actions={<button onClick={refresh}>Refresh</button>}>
        <div className="inline-form">
          <LookbackSelect value={lookback} onChange={onLookback} options={windowOptions(lookback)} />
          <HoursInput hours={lookback} onHours={onLookback} />
          <label>
            Reviewer
            <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="you@example.com" />
          </label>
        </div>
        {routing.error && <Alert>{routing.error}</Alert>}
        {labels.error && <Alert>Review labels are unavailable: {labels.error}</Alert>}
        {data && <p className="muted">Spans from telemetry project {data.project}.</p>}
        {data && (
          <dl className="facts" aria-label="Routing summary">
            <dt>Decisions</dt>
            <dd>{data.total}</dd>
            <dt>Succeeded</dt>
            <dd>{data.successes}</dd>
            <dt>Failed</dt>
            <dd>{data.failures}</dd>
            <dt>Ambiguous</dt>
            <dd>{data.ambiguous}</dd>
            <dt>Unreadable</dt>
            <dd>{data.unreadable}</dd>
            <dt>Accuracy</dt>
            <dd>{optional(data.accuracy, percent)}</dd>
            <dt>Confidence calibration</dt>
            <dd>{optional(data.confidence_calibration, (value) => value.toFixed(2))}</dd>
            <dt>Latency mean</dt>
            <dd>{optional(data.latency_ms.mean, ms)}</dd>
            <dt>Latency p50</dt>
            <dd>{optional(data.latency_ms.p50, ms)}</dd>
            <dt>Latency p95</dt>
            <dd>{optional(data.latency_ms.p95, ms)}</dd>
          </dl>
        )}
        {data && data.total === 0 && <p className="muted">No routing decisions were recorded in this window.</p>}
        {data && data.unreadable > 0 && (
          <p className="muted">Unreadable decisions record no chosen agent or confidence, so they are not evaluated.</p>
        )}
        {data && data.confidence_calibration === null && data.total > 0 && (
          <p className="muted">Calibration needs decisions with different confidences and outcomes.</p>
        )}
      </Panel>
      {notice && <Alert tone="ok">{notice}</Alert>}
      {data && data.total > 0 && (
        <>
          <Agents agents={data.per_agent} />
          <Charts decisions={decisions} />
        </>
      )}
      <Labelling
        tenant={tenant}
        lookback={lookback}
        version={version}
        decisions={decisions}
        relabelling={relabelling}
        onRelabel={relabel}
        onNotice={setNotice}
      />
      {data && data.total > 0 && (
        <Decisions
          tenant={tenant}
          decisions={decisions}
          reviewer={reviewer}
          relabelling={relabelling}
          onRelabel={relabel}
          onChanged={changed}
        />
      )}
      {labelling?.span_id && labels.data && (
        <Relabel
          key={labelling.span_id}
          tenant={tenant}
          decision={labelling}
          reviewer={reviewer}
          choices={labels.data}
          onDone={changed}
        />
      )}
    </>
  );
}

/** The preset windows, plus ``hours`` when it is none of them. */
function windowOptions(hours: number) {
  return ROUTING_LOOKBACKS.some((option) => option.hours === hours)
    ? ROUTING_LOOKBACKS
    : [...ROUTING_LOOKBACKS, { hours, label: `Last ${hours} hours` }].sort((a, b) => a.hours - b.hours);
}

function HoursInput({ hours, onHours }: { hours: number; onHours: (hours: number) => void }) {
  const [text, setText] = useState(String(hours));
  const [invalid, setInvalid] = useState(false);
  const apply = () => {
    const chosen = lookbackHours(text);
    setInvalid(chosen === null);
    if (chosen !== null && chosen !== hours) onHours(chosen);
  };
  return (
    <>
      <label>
        Hours
        <input
          type="number"
          min={1}
          max={MAX_LOOKBACK_HOURS}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onBlur={apply}
          onKeyDown={(e) => e.key === 'Enter' && apply()}
        />
      </label>
      {invalid && <Alert>Hours must be a whole number from 1 to {MAX_LOOKBACK_HOURS}.</Alert>}
    </>
  );
}

const SCORES = [
  { key: 'precision', name: 'Precision', color: 'lightblue' },
  { key: 'recall', name: 'Recall', color: 'lightgreen' },
  { key: 'f1', name: 'F1', color: 'orange' },
] as const;

function Agents({ agents }: { agents: AgentRouting[] }) {
  const scores = useMemo(
    () =>
      SCORES.map((score) => ({
        type: 'bar',
        name: score.name,
        x: agents.map((agent) => agent.agent),
        y: agents.map((agent) => agent[score.key]),
        marker: { color: score.color },
      })),
    [agents],
  );
  return (
    <Panel title="Decisions by agent">
      <table aria-label="Decisions by agent">
        <thead>
          <tr>
            <th>Agent</th>
            <th>Decisions</th>
            <th>Succeeded</th>
            <th>Failed</th>
            <th>Ambiguous</th>
            <th>Success rate</th>
            <th>Mean confidence</th>
            <th>Mean latency</th>
            {SCORES.map((score) => (
              <th key={score.key}>{score.name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {agents.map((agent) => (
            <tr key={agent.agent}>
              <td>{agent.agent}</td>
              <td>{agent.decisions}</td>
              <td>{agent.successes}</td>
              <td>{agent.failures}</td>
              <td>{agent.ambiguous}</td>
              <td>{percent(agent.success_rate)}</td>
              <td>{agent.mean_confidence.toFixed(2)}</td>
              <td>{ms(agent.mean_latency_ms)}</td>
              {SCORES.map((score) => (
                <td key={score.key} style={{ background: scoreColor(agent[score.key]), color: '#1a1a1a' }}>
                  {percent(agent[score.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <Plot
        title="Precision, recall and F1 by agent"
        data={scores}
        layout={{
          barmode: 'group',
          xaxis: { title: { text: 'Agent' } },
          yaxis: { title: { text: 'Score' }, range: [0, 1] },
        }}
      />
    </Panel>
  );
}

const OUTCOME_COLORS: Record<string, string> = {
  success: '#27ae60',
  failure: '#e74c3c',
  ambiguous: '#f39c12',
};

function Charts({ decisions }: { decisions: RoutingDecision[] }) {
  const histogram = useMemo(
    () =>
      Object.entries(OUTCOME_COLORS).map(([outcome, color]) => ({
        type: 'histogram',
        name: outcome,
        x: decisions.filter((decision) => decision.outcome === outcome).map((decision) => decision.confidence),
        xbins: { start: 0, end: 1, size: 0.1 },
        marker: { color },
      })),
    [decisions],
  );
  const calibration = useMemo(() => {
    const points = calibrationPoints(decisions);
    return [
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Actual success rate',
        x: points.map((point) => point.confidence),
        y: points.map((point) => point.success_rate),
        text: points.map((point) => `${point.decisions} decisions`),
        marker: { size: points.map((point) => point.decisions * 2), sizemode: 'area', sizemin: 4 },
        hovertemplate: 'Confidence %{x:.2f}<br>Success %{y:.0%}<br>%{text}<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Perfect calibration',
        x: [0, 1],
        y: [0, 1],
        line: { dash: 'dash', color: 'gray' },
      },
    ];
  }, [decisions]);
  const perAgent = useMemo(
    () =>
      decisionsPerHourByAgent(decisions).map((series) => ({
        type: 'scatter',
        mode: 'lines+markers',
        name: series.agent,
        x: series.hours,
        y: series.decisions,
      })),
    [decisions],
  );
  const successRate = useMemo(() => {
    const hours = successRatePerHour(decisions);
    return [
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Success rate',
        x: hours.map((hour) => hour.hour),
        y: hours.map((hour) => hour.success_rate),
      },
    ];
  }, [decisions]);
  return (
    <Panel title="Confidence and outcomes">
      <Plot
        title="Confidence by outcome"
        data={histogram}
        layout={{
          barmode: 'stack',
          xaxis: { title: { text: 'Confidence' }, range: [0, 1] },
          yaxis: { title: { text: 'Decisions' } },
        }}
      />
      <Plot
        title="Confidence calibration"
        data={calibration}
        layout={{
          xaxis: { title: { text: 'Routing confidence' }, range: [0, 1] },
          yaxis: { title: { text: 'Actual success rate' }, range: [0, 1], tickformat: '.0%' },
        }}
      />
      <Plot
        title="Decisions per hour by agent"
        data={perAgent}
        layout={{
          xaxis: { title: { text: 'Hour (UTC)' } },
          yaxis: { title: { text: 'Decisions' } },
        }}
      />
      <Plot
        title="Success rate per hour"
        data={successRate}
        layout={{
          xaxis: { title: { text: 'Hour (UTC)' } },
          yaxis: { title: { text: 'Success rate' }, range: [0, 1], tickformat: '.0%' },
        }}
      />
    </Panel>
  );
}

/** A decision's label as the tables show it: the label and the agent it
 * should have gone to, who gave it with the LLM's confidence and whether it
 * awaits review, and the reasoning. */
function LabelCell({ label }: { label: DecisionLabel | null }) {
  if (!label) return <>Unlabelled</>;
  const llm = label.annotator === 'llm';
  const by = [
    labelledBy(label),
    llm && label.confidence !== null ? `confidence ${label.confidence.toFixed(2)}` : null,
    llm && label.requires_review && !label.human_reviewed ? 'needs review' : null,
  ];
  return (
    <>
      {label.label}
      {label.suggested_agent && ` (should be ${label.suggested_agent})`}
      <br />
      <span className="muted">{by.filter(Boolean).join(' · ')}</span>
      {label.reasoning && (
        <>
          <br />
          <span className="muted">{label.reasoning}</span>
        </>
      )}
    </>
  );
}

function Labelling({
  tenant,
  lookback,
  version,
  decisions,
  relabelling,
  onRelabel,
  onNotice,
}: {
  tenant: string;
  lookback: number;
  /** Changes when the view refreshes, to read the stored labels again. */
  version: number;
  decisions: RoutingDecision[];
  relabelling: string | null;
  onRelabel: (decision: RoutingDecision) => void;
  onNotice: (notice: string) => void;
}) {
  const statistics = useLoad(
    (signal) => runtimeJson<LabelStatistics>(`${decisionsPath(tenant)}/label-statistics`, { signal }),
    [tenant, version],
  );
  const annotate = useAction();
  const find = useAction();
  const [threshold, setThreshold] = useState('0.6');
  const [limit, setLimit] = useState('20');
  const [candidates, setCandidates] = useState<AnnotationCandidate[] | null>(null);
  const [priorities, setPriorities] = useState<Priority[]>(PRIORITIES);
  const [showLlmLabelled, setShowLlmLabelled] = useState(true);
  const bySpan = useMemo(
    () => Object.fromEntries(decisions.map((decision) => [decision.span_id ?? '', decision])),
    [decisions],
  );
  const labels = useMemo(
    () => Object.fromEntries(decisions.map((decision) => [decision.span_id ?? '', decision.label])),
    [decisions],
  );
  const shown = candidates && shownCandidates(candidates, priorities, showLlmLabelled, labels);
  const stats = statistics.data;
  return (
    <Panel title="Labelling">
      <div className="inline-form">
        <button
          disabled={annotate.pending}
          onClick={() =>
            annotate.run(async () => {
              const run = await runtimeJson<{ workflow_name: string }>(`/admin/tenant/${seg(tenant)}/optimize`, {
                method: 'POST',
                body: { mode: 'llm-annotate', lookback_hours: lookback },
              });
              onNotice(`Started LLM labelling run ${run.workflow_name}. Follow it in Optimization runs, then refresh.`);
            })
          }
        >
          {annotate.pending ? 'Starting…' : 'Label with the LLM'}
        </button>
      </div>
      {annotate.error && <Alert>{annotate.error}</Alert>}
      {statistics.error && <Alert>Stored labels are unavailable: {statistics.error}</Alert>}
      {stats && (
        <dl className="facts" aria-label="Stored labels">
          <dt>Stored labels (30 days)</dt>
          <dd>{stats.total}</dd>
          <dt>Reviewed</dt>
          <dd>{stats.human_reviewed}</dd>
          <dt>Pending review</dt>
          <dd>{stats.pending_review}</dd>
          <dt>By label</dt>
          <dd>
            {Object.keys(stats.by_label).length
              ? Object.entries(stats.by_label)
                  .sort(([a], [b]) => a.localeCompare(b))
                  .map(([label, count]) => `${label} ${count}`)
                  .join(', ')
              : '—'}
          </dd>
        </dl>
      )}
      <form
        className="inline-form"
        aria-label="Find decisions needing review"
        onSubmit={(e) => {
          e.preventDefault();
          find.run(async () => {
            const query = new URLSearchParams({
              lookback_hours: String(lookback),
              confidence_threshold: threshold,
              max_annotations: limit,
            });
            const found = await runtimeJson<{ candidates: AnnotationCandidate[] }>(
              `${decisionsPath(tenant)}/annotation-candidates?${query}`,
            );
            setCandidates(found.candidates);
            const count = found.candidates.length;
            onNotice(`Found ${count} decision${count === 1 ? '' : 's'} needing review.`);
          });
        }}
      >
        <label>
          Confidence threshold
          <input
            required
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={threshold}
            onChange={(e) => setThreshold(e.target.value)}
          />
        </label>
        <label>
          Most to show
          <input required type="number" min={1} max={100} value={limit} onChange={(e) => setLimit(e.target.value)} />
        </label>
        <button type="submit" disabled={find.pending}>
          {find.pending ? 'Finding…' : 'Find decisions needing review'}
        </button>
      </form>
      {find.error && <Alert>{find.error}</Alert>}
      {candidates && (
        <>
          <div className="inline-form">
            <fieldset>
              <legend>Priority</legend>
              {PRIORITIES.map((priority) => (
                <label key={priority} className="check">
                  <input
                    type="checkbox"
                    checked={priorities.includes(priority)}
                    onChange={(e) =>
                      setPriorities(
                        PRIORITIES.filter((p) => (p === priority ? e.target.checked : priorities.includes(p))),
                      )
                    }
                  />
                  {priority}
                </label>
              ))}
            </fieldset>
            <label className="check">
              <input type="checkbox" checked={showLlmLabelled} onChange={(e) => setShowLlmLabelled(e.target.checked)} />
              Show LLM-labelled
            </label>
          </div>
          <p className="muted">
            Showing {shown!.length} of {candidates.length} decisions needing review.
          </p>
          {shown!.length > 0 && (
            <table aria-label="Decisions needing review">
              <thead>
                <tr>
                  <th>Priority</th>
                  <th>Time</th>
                  <th>Query</th>
                  <th>Agent</th>
                  <th>Confidence</th>
                  <th>Outcome</th>
                  <th>Reason</th>
                  <th>Label</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {shown!.map((candidate) => {
                  const decision: RoutingDecision = bySpan[candidate.span_id] ?? {
                    span_id: candidate.span_id,
                    trace_id: null,
                    start_time: candidate.start_time,
                    query: candidate.query,
                    chosen_agent: candidate.chosen_agent,
                    confidence: candidate.confidence,
                    outcome: candidate.outcome,
                    reason: '',
                    latency_ms: 0,
                    entity_extraction_failed: false,
                    label: null,
                  };
                  return (
                    <tr key={candidate.span_id} className={relabelling === candidate.span_id ? 'selected' : undefined}>
                      <td>{candidate.priority}</td>
                      <td title={`Span ${candidate.span_id}`}>{when(candidate.start_time)}</td>
                      <td>{candidate.query}</td>
                      <td>{candidate.chosen_agent}</td>
                      <td>{candidate.confidence.toFixed(2)}</td>
                      <td>{candidate.outcome}</td>
                      <td>{candidate.reason}</td>
                      <td>
                        <LabelCell label={decision.label} />
                      </td>
                      <td>
                        <button aria-label={`Label ${candidate.span_id} for review`} onClick={() => onRelabel(decision)}>
                          Label
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </>
      )}
    </Panel>
  );
}

function Decisions({
  tenant,
  decisions,
  reviewer,
  relabelling,
  onRelabel,
  onChanged,
}: {
  tenant: string;
  decisions: RoutingDecision[];
  reviewer: string;
  relabelling: string | null;
  onRelabel: (decision: RoutingDecision) => void;
  onChanged: (notice: string, decision: RoutingDecision) => void;
}) {
  const [filter, setFilter] = useState<LabelState | 'all'>('all');
  const approve = useAction();
  const shown = decisions.filter((decision) => filter === 'all' || labelState(decision) === filter);
  return (
    <Panel title="Decisions">
      <div className="inline-form">
        <label>
          Show
          <select value={filter} onChange={(e) => setFilter(e.target.value as LabelState | 'all')}>
            {LABEL_FILTERS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      {approve.error && <Alert>{approve.error}</Alert>}
      {shown.length === 0 ? (
        <p className="muted">No decisions match.</p>
      ) : (
        <table aria-label="Decisions">
          <thead>
            <tr>
              <th>Time</th>
              <th>Query</th>
              <th>Agent</th>
              <th>Confidence</th>
              <th>Outcome</th>
              <th>Latency</th>
              <th>Label</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {shown.map((decision) => {
              const id = decision.span_id ?? decision.start_time;
              return (
                <tr key={id} className={relabelling === decision.span_id ? 'selected' : undefined}>
                  <td title={decision.trace_id ? `Trace ${decision.trace_id}` : undefined}>
                    {when(decision.start_time)}
                  </td>
                  <td>{decision.query ?? '—'}</td>
                  <td>{decision.chosen_agent}</td>
                  <td>{decision.confidence.toFixed(2)}</td>
                  <td title={decision.reason}>{decision.outcome}</td>
                  <td>{ms(decision.latency_ms)}</td>
                  <td>
                    <LabelCell label={decision.label} />
                  </td>
                  <td>
                    {decision.span_id && (
                      <span className="confirm">
                        {approvable(decision) && (
                          <button
                            aria-label={`Approve the LLM label of ${decision.span_id}`}
                            disabled={approve.pending}
                            onClick={() =>
                              approve.run(async () => {
                                if (!reviewer.trim()) throw new Error('Enter your name as the reviewer first.');
                                const approved = await runtimeJson<RoutingDecision>(
                                  `${decisionsPath(tenant)}/${seg(decision.span_id!)}/approve`,
                                  {
                                    method: 'POST',
                                    body: {
                                      start_time: decision.start_time,
                                      reviewer: reviewer.trim(),
                                    },
                                  },
                                );
                                onChanged(`Approved the LLM label of ${decision.span_id}.`, approved);
                              })
                            }
                          >
                            Approve
                          </button>
                        )}
                        <button aria-label={`Relabel ${decision.span_id}`} onClick={() => onRelabel(decision)}>
                          Relabel
                        </button>
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

function Relabel({
  tenant,
  decision,
  reviewer,
  choices,
  onDone,
}: {
  tenant: string;
  decision: RoutingDecision;
  reviewer: string;
  choices: string[];
  onDone: (notice: string, decision: RoutingDecision) => void;
}) {
  const [label, setLabel] = useState(initialReviewLabel(decision.label, choices));
  const [reasoning, setReasoning] = useState(decision.label?.reasoning ?? '');
  const [suggested, setSuggested] = useState(decision.label?.suggested_agent ?? '');
  const save = useAction();
  const spanId = decision.span_id!;
  return (
    <Panel title={`Label ${spanId}`}>
      <form
        className="inline-form"
        aria-label={`Label ${spanId}`}
        onSubmit={(e) => {
          e.preventDefault();
          save.run(async () => {
            if (!reviewer.trim()) throw new Error('Enter your name as the reviewer first.');
            const labelled = await runtimeJson<RoutingDecision>(`${decisionsPath(tenant)}/${seg(spanId)}/label`, {
              method: 'PUT',
              body: {
                start_time: decision.start_time,
                reviewer: reviewer.trim(),
                label,
                reasoning,
                suggested_agent: suggested.trim() || null,
              },
            });
            onDone(`Labelled ${spanId} ${label}.`, labelled);
          });
        }}
      >
        <span>
          {decision.query ?? '—'} → {decision.chosen_agent} ({decision.confidence.toFixed(2)})
        </span>
        <label>
          Label
          <select required value={label} onChange={(e) => setLabel(e.target.value)}>
            {choices.map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label>
          Reasoning
          <textarea rows={2} value={reasoning} onChange={(e) => setReasoning(e.target.value)} />
        </label>
        <label>
          Should have gone to
          <input value={suggested} onChange={(e) => setSuggested(e.target.value)} placeholder="agent name" />
        </label>
        <button type="submit" disabled={save.pending}>
          {save.pending ? 'Saving…' : 'Save label'}
        </button>
        {save.error && <Alert>{save.error}</Alert>}
      </form>
    </Panel>
  );
}
