import { useState } from 'react';
import { Alert, Panel, useAction } from './common';
import { runtimeJson, seg } from './http';
import { percent } from './metrics';
import { formatEvidence, formatRange, slowThresholds, type Trace } from './traces';

export interface RootCause {
  hypothesis: string;
  confidence: number;
  category: string;
  evidence: string[];
  affected_traces: string[];
  suggested_action: string;
}

export interface Recommendation {
  priority: string;
  category: string;
  recommendation: string;
  details: string[];
  affected_components: string[];
}

export interface Tally {
  value: string;
  count: number;
}

export interface FailureAnalysis {
  error_types: Tally[];
  operations: Tally[];
  profiles: Tally[];
  strategies: Tally[];
  hours: { hour: number; requests: number; failed: number; failure_rate: number }[];
  bursts: { start_time: string; end_time: string; failures: number; duration_minutes: number; trace_ids: string[] }[];
}

export interface PerformanceAnalysis {
  percentile: number;
  threshold_ms: number;
  operations: { operation: string; count: number; mean_ms: number; min_ms: number; max_ms: number; sample_ms: number[] }[];
  profiles: Tally[];
  strategies: Tally[];
  latency: {
    slow_mean_ms: number;
    slow_std_ms: number;
    normal_mean_ms: number;
    normal_std_ms: number;
    slowdown_factor: number;
  };
}

export interface RootCauseAnalysis {
  traces: number;
  failed: number;
  slow: number;
  failure_rate: number;
  slow_threshold_ms: number | null;
  root_causes: RootCause[];
  recommendations: Recommendation[];
  failure_analysis: FailureAnalysis | null;
  performance_analysis: PerformanceAnalysis | null;
}

export interface PhoenixLinks {
  phoenix_url: string | null;
  project: string;
  project_url: string | null;
}

/** One analysis and the request it answered, so a section switch or a
 * repeated request shows it again without re-running. */
export interface CachedAnalysis {
  signature: string;
  includeSlow: boolean;
  percentile: string;
  analysis: RootCauseAnalysis;
}

const ms = (value: number) => `${value.toFixed(1)} ms`;

/** The Phoenix query selecting ``ids``. */
export function traceIdsQuery(ids: string[]): string {
  return ids.map((id) => `trace_id == "${id}"`).join(' or ');
}

/** The Phoenix query selecting the spans from ``start`` to ``end``. */
export function timeRangeQuery(start: string, end: string): string {
  return `timestamp >= "${start}" and timestamp <= "${end}"`;
}

function PhoenixQuery({ query, links, what }: { query: string; links?: PhoenixLinks; what: string }) {
  return (
    <div className="phoenix-query">
      <code aria-label={`Phoenix query for ${what}`}>{query}</code>
      {links?.project_url ? (
        <a href={links.project_url} target="_blank" rel="noreferrer">
          Open the project in Phoenix
        </a>
      ) : null}
    </div>
  );
}

function TallyTable({ label, name, tallies, total }: { label: string; name: string; tallies: Tally[]; total: number }) {
  if (!tallies.length) return null;
  return (
    <table aria-label={label}>
      <thead>
        <tr>
          <th>{name}</th>
          <th>Count</th>
          <th>Share</th>
        </tr>
      </thead>
      <tbody>
        {tallies.map((tally) => (
          <tr key={tally.value}>
            <td>{tally.value}</td>
            <td>{tally.count}</td>
            <td>{percent(tally.count / total)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const QUERY_REFERENCE: [string, string[]][] = [
  ['Status', ['status_code == "ERROR"', 'status_code == "OK"']],
  ['Latency', ['latency_ms > 1000', 'latency_ms < 100']],
  ['Time range', ['timestamp >= "2026-01-01T00:00:00Z"', 'timestamp >= "2026-01-01" and timestamp <= "2026-01-02"']],
  ['Trace IDs', ['trace_id == "abc123"', 'trace_id == "id1" or trace_id == "id2"']],
];

export function RootCauses({
  tenant,
  query,
  basis,
  traces,
  links,
  cached,
  onAnalysis,
}: {
  tenant: string;
  query: URLSearchParams;
  /** Identifies the traces analyzed, so a cached analysis of the same traces
   * shows again. */
  basis: string;
  traces: Trace[];
  links?: PhoenixLinks;
  cached?: CachedAnalysis;
  onAnalysis: (cached: CachedAnalysis) => void;
}) {
  const [includeSlow, setIncludeSlow] = useState(cached?.includeSlow ?? true);
  const [percentile, setPercentile] = useState(cached?.percentile ?? '95');
  const action = useAction();
  const signature = (slow: boolean, value: string) => `${basis}|${slow}|${value}`;
  const analysis = cached?.signature === signature(includeSlow, percentile) ? cached.analysis : undefined;
  const value = Number(percentile);
  const preview = Number.isInteger(value) && value >= 50 && value <= 99 ? slowThresholds(traces, value) : undefined;
  return (
    <Panel title="Root causes">
      <details className="explainer">
        <summary>Phoenix query reference</summary>
        <p>Paste these into the search bar of the tenant's Phoenix project; combine them with and / or.</p>
        <dl>
          {QUERY_REFERENCE.map(([kind, queries]) => (
            <div key={kind}>
              <dt>{kind}</dt>
              {queries.map((item) => (
                <dd key={item}>
                  <code>{item}</code>
                </dd>
              ))}
            </div>
          ))}
        </dl>
      </details>
      <form
        className="inline-form"
        aria-label="Find root causes"
        onSubmit={(e) => {
          e.preventDefault();
          action.run(async () => {
            if (!Number.isInteger(value) || value < 50 || value > 99)
              throw new Error('The slow percentile must be a whole number from 50 to 99.');
            const params = new URLSearchParams(query);
            params.set('include_slow', String(includeSlow));
            params.set('slow_percentile', String(value));
            onAnalysis({
              signature: signature(includeSlow, percentile),
              includeSlow,
              percentile,
              analysis: await runtimeJson<RootCauseAnalysis>(`/admin/tenant/${seg(tenant)}/telemetry/root-causes?${params}`),
            });
          });
        }}
      >
        <label className="check">
          <input type="checkbox" checked={includeSlow} onChange={(e) => setIncludeSlow(e.target.checked)} />
          Include slow traces
        </label>
        <label>
          Slow percentile
          <input inputMode="numeric" value={percentile} onChange={(e) => setPercentile(e.target.value)} />
        </label>
        <button type="submit" disabled={action.pending}>
          {action.pending ? 'Analyzing…' : 'Find root causes'}
        </button>
        {action.error && <Alert>{action.error}</Alert>}
      </form>
      {includeSlow && preview?.succeeded != null && (
        <p className="caption" aria-label="Slow threshold">
          P{value} of the successful traces: {ms(preview.succeeded)}; successful traces slower than this are flagged.
          {preview.all != null && Math.abs(preview.all - preview.succeeded) > 100
            ? ` P${value} of all traces, failures included, is ${ms(preview.all)}.`
            : ''}
        </p>
      )}
      {!analysis && (
        <p className="muted">
          {traces.length
            ? 'Find root causes runs over the traces the filters above keep.'
            : 'No traces are available for root cause analysis.'}
        </p>
      )}
      {analysis && <Analysis analysis={analysis} percentile={value} traces={traces} links={links} />}
    </Panel>
  );
}

function Analysis({
  analysis,
  percentile,
  traces,
  links,
}: {
  analysis: RootCauseAnalysis;
  percentile: number;
  traces: Trace[];
  links?: PhoenixLinks;
}) {
  const issues = analysis.failed + analysis.slow;
  const all = slowThresholds(traces, percentile).all;
  return (
    <>
      <dl className="facts" aria-label="Root cause summary">
        <dt>Traces analyzed</dt>
        <dd>{analysis.traces}</dd>
        <dt>Total issues</dt>
        <dd>{issues}</dd>
        <dt>Failed</dt>
        <dd>
          {analysis.failed} ({percent(analysis.failure_rate)})
        </dd>
        <dt>Slow</dt>
        <dd>
          {analysis.slow_threshold_ms === null
            ? analysis.slow
            : `${analysis.slow} (slower than ${ms(analysis.slow_threshold_ms)})`}
        </dd>
      </dl>
      {analysis.slow_threshold_ms !== null && all !== null && (
        <p className="caption" aria-label="Threshold comparison">
          P{percentile} of the successful traces is {ms(analysis.slow_threshold_ms)}; P{percentile} of all traces,
          failures included, is {ms(all)}. Failed traces do not count toward the slow threshold.
        </p>
      )}
      {issues === 0 && (
        <p className="ok-line">
          No failures or slow traces among the {analysis.traces} analyzed: the failure rate is 0%.
        </p>
      )}
      {analysis.root_causes.length === 0 ? (
        <p className="muted">No root cause stands out in these traces.</p>
      ) : (
        <ol className="root-causes" aria-label="Hypotheses">
          {analysis.root_causes.map((cause) => (
            <li key={`${cause.category}-${cause.hypothesis}`}>
              <details>
                <summary>
                  {cause.hypothesis} ({percent(cause.confidence)} confidence, {cause.category})
                </summary>
                <ul aria-label="Evidence">
                  {cause.evidence.map((item) => (
                    <li key={item}>{formatEvidence(item)}</li>
                  ))}
                </ul>
                <p>Suggested action: {cause.suggested_action}</p>
                <p>Affected traces: {cause.affected_traces.join(', ')}</p>
                {cause.affected_traces.length > 0 && (
                  <>
                    <p>
                      {cause.affected_traces.length} affected {cause.affected_traces.length === 1 ? 'trace' : 'traces'};
                      sample:{' '}
                      {cause.affected_traces
                        .slice(0, 5)
                        .map((id) => `${id.slice(0, 8)}…`)
                        .join(', ')}
                    </p>
                    <PhoenixQuery
                      query={traceIdsQuery(cause.affected_traces.slice(0, 5))}
                      links={links}
                      what={`the traces of ${cause.hypothesis}`}
                    />
                  </>
                )}
              </details>
            </li>
          ))}
        </ol>
      )}
      {analysis.recommendations.length > 0 && (
        <table aria-label="Recommendations">
          <thead>
            <tr>
              <th>Priority</th>
              <th>Category</th>
              <th>Recommendation</th>
              <th>Details</th>
              <th>Affected components</th>
            </tr>
          </thead>
          <tbody>
            {analysis.recommendations.map((item) => (
              <tr key={`${item.category}-${item.recommendation}`}>
                <td>{item.priority}</td>
                <td>{item.category}</td>
                <td>{item.recommendation}</td>
                <td>{item.details.join('; ')}</td>
                <td>{item.affected_components.join(', ') || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {analysis.failure_analysis && (
        <FailureDetails failures={analysis.failure_analysis} failed={analysis.failed} links={links} />
      )}
      {analysis.performance_analysis && (
        <PerformanceDetails performance={analysis.performance_analysis} slow={analysis.slow} links={links} />
      )}
    </>
  );
}

function FailureDetails({ failures, failed, links }: { failures: FailureAnalysis; failed: number; links?: PhoenixLinks }) {
  return (
    <section className="analysis-detail" aria-label="Failure analysis">
      <h3>Failure analysis</h3>
      <p>
        {links?.project_url ? (
          <a href={links.project_url} target="_blank" rel="noreferrer">
            View all {failed} failed traces in Phoenix
          </a>
        ) : (
          `${failed} failed traces`
        )}
      </p>
      <PhoenixQuery query={'status_code == "ERROR"'} what="the failed traces" />
      <TallyTable label="Error types" name="Error type" tallies={failures.error_types} total={failed} />
      <TallyTable label="Failed operations" name="Operation" tallies={failures.operations} total={failed} />
      <TallyTable label="Failed profiles" name="Profile" tallies={failures.profiles} total={failed} />
      <TallyTable label="Failed strategies" name="Strategy" tallies={failures.strategies} total={failed} />
      {failures.hours.length > 0 && (
        <table aria-label="Hours with failures">
          <thead>
            <tr>
              <th>Hour (UTC)</th>
              <th>Traces</th>
              <th>Failed</th>
              <th>Failure rate</th>
            </tr>
          </thead>
          <tbody>
            {failures.hours.map((hour) => (
              <tr key={hour.hour}>
                <td>{String(hour.hour).padStart(2, '0')}:00</td>
                <td>{hour.requests}</td>
                <td>{hour.failed}</td>
                <td>{percent(hour.failure_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {failures.bursts.length > 0 && (
        <table aria-label="Failure bursts">
          <thead>
            <tr>
              <th>Burst</th>
              <th>Failures</th>
              <th>Duration</th>
              <th>When</th>
              <th>Phoenix query</th>
            </tr>
          </thead>
          <tbody>
            {failures.bursts.map((burst, index) => (
              <tr key={burst.start_time}>
                <td>{index + 1}</td>
                <td>{burst.failures}</td>
                <td>{burst.duration_minutes.toFixed(1)} min</td>
                <td>{formatRange(burst.start_time, burst.end_time)}</td>
                <td>
                  <PhoenixQuery
                    query={timeRangeQuery(burst.start_time, burst.end_time)}
                    links={links}
                    what={`burst ${index + 1}`}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function PerformanceDetails({
  performance,
  slow,
  links,
}: {
  performance: PerformanceAnalysis;
  slow: number;
  links?: PhoenixLinks;
}) {
  const { latency } = performance;
  return (
    <section className="analysis-detail" aria-label="Performance analysis">
      <h3>Performance analysis</h3>
      <p>
        {links?.project_url ? (
          <a href={links.project_url} target="_blank" rel="noreferrer">
            View {slow} slow traces in Phoenix
          </a>
        ) : (
          `${slow} slow traces`
        )}
      </p>
      <PhoenixQuery query={`latency_ms > ${performance.threshold_ms.toFixed(0)}`} what="the slow traces" />
      <p className="caption">
        Successful traces slower than P{performance.percentile} ({ms(performance.threshold_ms)}) of the successful
        traces are flagged.
      </p>
      <table aria-label="Slow operations">
        <thead>
          <tr>
            <th>Operation</th>
            <th>Slow traces</th>
            <th>Range</th>
            <th>Mean</th>
            <th>Sample</th>
          </tr>
        </thead>
        <tbody>
          {performance.operations.map((op) => (
            <tr key={op.operation}>
              <td>{op.operation}</td>
              <td>{op.count}</td>
              <td>{op.min_ms === op.max_ms ? ms(op.min_ms) : `${ms(op.min_ms)} - ${ms(op.max_ms)}`}</td>
              <td>{ms(op.mean_ms)}</td>
              <td>{op.sample_ms.slice(0, 3).map(ms).join(', ')}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <dl className="facts" aria-label="Degradation">
        <dt>Slow mean</dt>
        <dd>
          {ms(latency.slow_mean_ms)} (σ {ms(latency.slow_std_ms)})
        </dd>
        <dt>Other successful mean</dt>
        <dd>
          {ms(latency.normal_mean_ms)} (σ {ms(latency.normal_std_ms)})
        </dd>
        <dt>Slowdown</dt>
        <dd>{latency.slowdown_factor.toFixed(2)}×</dd>
      </dl>
      <TallyTable label="Slow profiles" name="Profile" tallies={performance.profiles} total={slow} />
      <TallyTable label="Slow strategies" name="Strategy" tallies={performance.strategies} total={slow} />
    </section>
  );
}
