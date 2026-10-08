import { useMemo, useState } from 'react';
import { Alert, Panel, useAction, useLoad } from './common';
import { runtimeJson, seg } from './http';
import { LookbackSelect, percent } from './metrics';
import { Plot } from './Plot';
import {
  LABEL_FILTERS,
  REVIEW_LABELS,
  ROUTING_LOOKBACKS,
  calibrationBins,
  decisionsByHour,
  labelState,
  labelledBy,
  withReviews,
  type AgentRouting,
  type LabelState,
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

export function RoutingView() {
  const [tenant, setTenant] = useState('');
  const [lookback, setLookback] = useState(24);
  return (
    <div className="ops-view">
      <TenantChooser action="Show decisions" onChoose={setTenant} />
      {tenant && <Routing key={`${tenant}-${lookback}`} tenant={tenant} lookback={lookback} onLookback={setLookback} />}
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
  // Decisions as the review routes answered them; Phoenix serves a new
  // label to the list only after a short indexing delay.
  const [reviewed, setReviewed] = useState<Record<string, RoutingDecision>>({});
  const routing = useLoad(
    (signal) => runtimeJson<RoutingDecisions>(`${decisionsPath(tenant)}?lookback_hours=${lookback}`, { signal }),
    [tenant, lookback],
  );
  const annotate = useAction();
  const changed = (message: string, decision: RoutingDecision) => {
    setNotice(message);
    setReviewed((previous) => ({ ...previous, [decision.span_id!]: decision }));
  };
  const refresh = () => {
    setReviewed({});
    routing.reload();
  };
  const data = routing.data;
  const decisions = useMemo(() => withReviews(data?.decisions ?? [], reviewed), [data, reviewed]);
  return (
    <>
      <Panel title={`Routing decisions of ${tenant}`} actions={<button onClick={refresh}>Refresh</button>}>
        <div className="inline-form">
          <LookbackSelect value={lookback} onChange={onLookback} options={ROUTING_LOOKBACKS} />
          <label>
            Reviewer
            <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="you@example.com" />
          </label>
          <button
            disabled={annotate.pending}
            onClick={() =>
              annotate.run(async () => {
                const run = await runtimeJson<{ workflow_name: string }>(`/admin/tenant/${seg(tenant)}/optimize`, {
                  method: 'POST',
                  body: { mode: 'llm-annotate' },
                });
                setNotice(
                  `Started LLM labelling run ${run.workflow_name}. Follow it in Optimization runs, then refresh.`,
                );
              })
            }
          >
            {annotate.pending ? 'Starting…' : 'Label with the LLM'}
          </button>
        </div>
        {annotate.error && <Alert>{annotate.error}</Alert>}
        {routing.error && <Alert>{routing.error}</Alert>}
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
          <Decisions tenant={tenant} decisions={decisions} reviewer={reviewer} onChanged={changed} />
        </>
      )}
    </>
  );
}

function Agents({ agents }: { agents: AgentRouting[] }) {
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
            </tr>
          ))}
        </tbody>
      </table>
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
    const bins = calibrationBins(decisions);
    return [
      {
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Success rate',
        x: bins.map((bin) => bin.range),
        y: bins.map((bin) => bin.success_rate),
        text: bins.map((bin) => `${bin.decisions} decisions`),
        hovertemplate: 'Confidence %{x}<br>Success %{y:.0%}<br>%{text}<extra></extra>',
      },
    ];
  }, [decisions]);
  const overTime = useMemo(() => {
    const hours = decisionsByHour(decisions);
    return [
      {
        type: 'bar',
        name: 'Decisions',
        x: hours.map((h) => h.hour),
        y: hours.map((h) => h.decisions),
      },
      {
        type: 'bar',
        name: 'Succeeded',
        x: hours.map((h) => h.hour),
        y: hours.map((h) => h.successes),
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
        title="Success rate by confidence"
        data={calibration}
        layout={{
          xaxis: { title: { text: 'Confidence' } },
          yaxis: {
            title: { text: 'Success rate' },
            range: [0, 1],
            tickformat: '.0%',
          },
        }}
      />
      <Plot
        title="Decisions per hour"
        data={overTime}
        layout={{
          barmode: 'group',
          xaxis: { title: { text: 'Hour (UTC)' } },
          yaxis: { title: { text: 'Decisions' } },
        }}
      />
    </Panel>
  );
}

function Decisions({
  tenant,
  decisions,
  reviewer,
  onChanged,
}: {
  tenant: string;
  decisions: RoutingDecision[];
  reviewer: string;
  onChanged: (notice: string, decision: RoutingDecision) => void;
}) {
  const [filter, setFilter] = useState<LabelState | 'all'>('all');
  const [relabelling, setRelabelling] = useState<string | null>(null);
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
                <tr key={id} className={relabelling === id ? 'selected' : undefined}>
                  <td title={decision.trace_id ? `Trace ${decision.trace_id}` : undefined}>
                    {when(decision.start_time)}
                  </td>
                  <td>{decision.query ?? '—'}</td>
                  <td>{decision.chosen_agent}</td>
                  <td>{decision.confidence.toFixed(2)}</td>
                  <td title={decision.reason}>{decision.outcome}</td>
                  <td>{ms(decision.latency_ms)}</td>
                  <td>
                    {decision.label ? (
                      <span title={decision.label.reasoning ?? undefined}>
                        {decision.label.label}
                        {decision.label.suggested_agent && ` (should be ${decision.label.suggested_agent})`}
                        <br />
                        <span className="muted">{labelledBy(decision.label)}</span>
                      </span>
                    ) : (
                      'Unlabelled'
                    )}
                  </td>
                  <td>
                    {decision.span_id && (
                      <span className="confirm">
                        {labelState(decision) === 'llm' && (
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
                        <button
                          aria-label={`Relabel ${decision.span_id}`}
                          onClick={() => setRelabelling(relabelling === id ? null : id)}
                        >
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
      {(() => {
        const decision = shown.find((row) => (row.span_id ?? row.start_time) === relabelling);
        return (
          decision?.span_id && (
            <Relabel
              key={decision.span_id}
              tenant={tenant}
              decision={decision}
              reviewer={reviewer}
              onDone={(message, labelled) => {
                setRelabelling(null);
                onChanged(message, labelled);
              }}
            />
          )
        );
      })()}
    </Panel>
  );
}

function Relabel({
  tenant,
  decision,
  reviewer,
  onDone,
}: {
  tenant: string;
  decision: RoutingDecision;
  reviewer: string;
  onDone: (notice: string, decision: RoutingDecision) => void;
}) {
  const [label, setLabel] = useState(REVIEW_LABELS[0]);
  const [reasoning, setReasoning] = useState('');
  const [suggested, setSuggested] = useState('');
  const save = useAction();
  const spanId = decision.span_id!;
  return (
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
      <strong>Label {spanId}</strong>
      <label>
        Label
        <select value={label} onChange={(e) => setLabel(e.target.value)}>
          {REVIEW_LABELS.map((value) => (
            <option key={value}>{value}</option>
          ))}
        </select>
      </label>
      <label>
        Reasoning
        <input value={reasoning} onChange={(e) => setReasoning(e.target.value)} />
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
  );
}
