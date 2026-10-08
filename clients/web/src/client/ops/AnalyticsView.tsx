import { useMemo, useState } from 'react';
import { Alert, Panel, useAction, useLoad } from './common';
import { runtimeJson, seg } from './http';
import { Bars, LookbackSelect, percent } from './metrics';
import { Plot } from './Plot';
import { TenantChooser } from './tenants';
import {
  GROUPS,
  HEATMAP_COLUMNS,
  HEATMAP_ROWS,
  SORTS,
  WINDOWS,
  durationsBy,
  explore,
  heatmap,
  outliers,
  timeBuckets,
  type Group,
  type HeatmapField,
  type Sort,
  type Trace,
  type TraceAnalytics,
  type Window,
} from './traces';

const SECTIONS = ['Overview', 'Time series', 'Distribution', 'Heatmap', 'Outliers', 'Trace explorer', 'Root causes'] as const;
type Section = (typeof SECTIONS)[number];
const PAGE_SIZE = 20;

const ms = (value: number | null) => (value === null ? '—' : `${value.toFixed(1)} ms`);
const when = (iso: string) => new Date(iso).toLocaleString();

interface Filters {
  operation: string;
  profiles: string[];
  strategies: string[];
}

export function AnalyticsView() {
  const [tenant, setTenant] = useState('');
  const [lookback, setLookback] = useState(24);
  return (
    <div className="ops-view">
      <TenantChooser action="Show traces" onChoose={setTenant} />
      {tenant && <Traces key={`${tenant}-${lookback}`} tenant={tenant} lookback={lookback} onLookback={setLookback} />}
    </div>
  );
}

function Traces({ tenant, lookback, onLookback }: { tenant: string; lookback: number; onLookback: (hours: number) => void }) {
  const [filters, setFilters] = useState<Filters>({ operation: '', profiles: [], strategies: [] });
  const [section, setSection] = useState<Section>('Overview');
  const query = useMemo(() => {
    const params = new URLSearchParams({ lookback_hours: String(lookback), operation: filters.operation });
    filters.profiles.forEach((value) => params.append('profile', value));
    filters.strategies.forEach((value) => params.append('strategy', value));
    return params;
  }, [lookback, filters]);
  const analytics = useLoad(
    (signal) => runtimeJson<TraceAnalytics>(`/admin/tenant/${seg(tenant)}/telemetry/traces?${query}`, { signal }),
    [tenant, query],
  );
  const data = analytics.data;
  const statistics = data?.statistics;
  return (
    <>
      <Panel title={`Traces of ${tenant}`} actions={<button onClick={analytics.reload}>Refresh</button>}>
        <FilterForm
          key={JSON.stringify(data?.facets ?? null)}
          lookback={lookback}
          onLookback={onLookback}
          facets={data?.facets}
          filters={filters}
          onApply={setFilters}
        />
        {analytics.error && <Alert>{analytics.error}</Alert>}
        {statistics && statistics.requests === 0 && <p className="muted">No traces match in this window.</p>}
        {statistics && statistics.requests > 0 && (
          <dl className="facts" aria-label="Trace summary">
            <dt>Traces</dt>
            <dd>{statistics.requests}</dd>
            <dt>Succeeded</dt>
            <dd>
              {statistics.succeeded} ({percent(statistics.success_rate ?? 0)})
            </dd>
            <dt>Mean latency</dt>
            <dd>{ms(statistics.latency_ms.mean)}</dd>
            <dt>P95 latency</dt>
            <dd>{ms(statistics.latency_ms.p95)}</dd>
          </dl>
        )}
      </Panel>
      {data && statistics && statistics.requests > 0 && (
        <>
          <nav className="section-tabs" aria-label="Analytics sections">
            {SECTIONS.map((name) => (
              <button key={name} aria-pressed={section === name} onClick={() => setSection(name)}>
                {name}
              </button>
            ))}
          </nav>
          {section === 'Overview' && <Overview data={data} />}
          {section === 'Time series' && <TimeSeries traces={data.traces} />}
          {section === 'Distribution' && <Distribution traces={data.traces} />}
          {section === 'Heatmap' && <HeatmapSection traces={data.traces} />}
          {section === 'Outliers' && <Outliers data={data} />}
          {section === 'Trace explorer' && <Explorer traces={data.traces} />}
          {section === 'Root causes' && <RootCauses key={query.toString()} tenant={tenant} query={query} />}
        </>
      )}
    </>
  );
}

function FilterForm({
  lookback,
  onLookback,
  facets,
  filters,
  onApply,
}: {
  lookback: number;
  onLookback: (hours: number) => void;
  facets?: TraceAnalytics['facets'];
  filters: Filters;
  onApply: (filters: Filters) => void;
}) {
  const [draft, setDraft] = useState(filters);
  const selected = (select: HTMLSelectElement) => [...select.selectedOptions].map((option) => option.value);
  return (
    <form
      className="inline-form"
      aria-label="Trace filters"
      onSubmit={(e) => {
        e.preventDefault();
        onApply({ ...draft, operation: draft.operation.trim() });
      }}
    >
      <LookbackSelect value={lookback} onChange={onLookback} />
      <label>
        Operation contains
        <input value={draft.operation} onChange={(e) => setDraft({ ...draft, operation: e.target.value })} />
      </label>
      <label>
        Profiles
        <select
          multiple
          value={draft.profiles}
          onChange={(e) => setDraft({ ...draft, profiles: selected(e.target) })}
        >
          {(facets?.profiles ?? draft.profiles).map((value) => (
            <option key={value}>{value}</option>
          ))}
        </select>
      </label>
      <label>
        Strategies
        <select
          multiple
          value={draft.strategies}
          onChange={(e) => setDraft({ ...draft, strategies: selected(e.target) })}
        >
          {(facets?.strategies ?? draft.strategies).map((value) => (
            <option key={value}>{value}</option>
          ))}
        </select>
      </label>
      <button type="submit">Apply filters</button>
    </form>
  );
}

function Overview({ data }: { data: TraceAnalytics }) {
  const { latency_ms: latency, by_operation: operations, requests } = data.statistics;
  const percentiles = (['min', 'p50', 'p75', 'p90', 'p95', 'p99', 'max'] as const).map((key) => ({
    label: key === 'min' ? 'Min' : key === 'max' ? 'Max' : key.toUpperCase(),
    value: latency[key] ?? 0,
  }));
  return (
    <Panel title="Overview">
      <div className="chart-grid">
        <Bars title="Latency percentiles" entries={percentiles} format={ms} />
        <Bars
          title="Traces per operation"
          entries={operations.map((row) => ({ label: row.operation, value: row.count }))}
          format={String}
        />
      </div>
      <table aria-label="Traces by operation">
        <thead>
          <tr>
            <th>Operation</th>
            <th>Traces</th>
            <th>Share</th>
            <th>Mean</th>
            <th>P95</th>
            <th>Failed</th>
          </tr>
        </thead>
        <tbody>
          {operations.map((row) => (
            <tr key={row.operation}>
              <td>{row.operation}</td>
              <td>{row.count}</td>
              <td>{percent(row.count / requests)}</td>
              <td>{ms(row.mean_ms)}</td>
              <td>{ms(row.p95_ms)}</td>
              <td>{percent(row.error_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}

function TimeSeries({ traces }: { traces: Trace[] }) {
  const [window, setWindow] = useState<Window>('5 min');
  const buckets = useMemo(() => timeBuckets(traces, window), [traces, window]);
  const latency = useMemo(() => {
    const x = buckets.map((bucket) => bucket.start);
    return [
      { type: 'scatter', mode: 'lines+markers', name: 'Mean', x, y: buckets.map((bucket) => bucket.mean_ms) },
      { type: 'scatter', mode: 'lines', name: 'P50', x, y: buckets.map((bucket) => bucket.p50_ms), line: { dash: 'dot' } },
      { type: 'scatter', mode: 'lines', name: 'P95', x, y: buckets.map((bucket) => bucket.p95_ms), line: { dash: 'dash' } },
    ];
  }, [buckets]);
  const volume = useMemo(
    () => [{ type: 'bar', name: 'Traces', x: buckets.map((bucket) => bucket.start), y: buckets.map((bucket) => bucket.requests) }],
    [buckets],
  );
  return (
    <Panel title="Time series">
      <div className="inline-form">
        <label>
          Bucket
          <select value={window} onChange={(e) => setWindow(e.target.value as Window)}>
            {Object.keys(WINDOWS).map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </label>
      </div>
      <Plot title="Latency over time" data={latency} layout={LATENCY_OVER_TIME} />
      <Plot title="Traces over time" data={volume} layout={TRACES_OVER_TIME} />
    </Panel>
  );
}

const LATENCY_OVER_TIME = { yaxis: { title: { text: 'Latency (ms)' } }, hovermode: 'x unified' };
const TRACES_OVER_TIME = { yaxis: { title: { text: 'Traces' } } };
const MS_AXIS = { xaxis: { title: { text: 'Latency (ms)' } } };

function Distribution({ traces }: { traces: Trace[] }) {
  const [group, setGroup] = useState<Group | ''>('');
  const groups = useMemo(() => durationsBy(traces, group || null), [traces, group]);
  const histogram = useMemo(
    () => groups.map((entry) => ({ type: 'histogram', name: entry.name, x: entry.durations, opacity: 0.7 })),
    [groups],
  );
  const boxes = useMemo(
    () => groups.map((entry) => ({ type: 'box', name: entry.name, x: entry.durations, boxpoints: 'outliers' })),
    [groups],
  );
  return (
    <Panel title="Distribution">
      <div className="inline-form">
        <label>
          Group by
          <select value={group} onChange={(e) => setGroup(e.target.value as Group | '')}>
            <option value="">Nothing</option>
            {GROUPS.map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </label>
      </div>
      <Plot title="Latency histogram" data={histogram} layout={HISTOGRAM} />
      <Plot title="Latency spread" data={boxes} layout={MS_AXIS} />
    </Panel>
  );
}

const HISTOGRAM = { ...MS_AXIS, barmode: 'overlay', yaxis: { title: { text: 'Traces' } } };

function HeatmapSection({ traces }: { traces: Trace[] }) {
  const [column, setColumn] = useState<HeatmapField>('hour');
  const [row, setRow] = useState<HeatmapField>('operation');
  const grid = useMemo(() => heatmap(traces, column, row), [traces, column, row]);
  const data = useMemo(
    () => [
      {
        type: 'heatmap',
        x: grid.columns,
        y: grid.rows,
        z: grid.cells,
        colorscale: 'Viridis',
        colorbar: { title: { text: 'ms' } },
      },
    ],
    [grid],
  );
  const layout = useMemo(
    () => ({ xaxis: { title: { text: column }, type: 'category' }, yaxis: { title: { text: row }, type: 'category' } }),
    [column, row],
  );
  return (
    <Panel title="Heatmap">
      <div className="inline-form">
        <label>
          Columns
          <select value={column} onChange={(e) => setColumn(e.target.value as HeatmapField)}>
            {HEATMAP_COLUMNS.map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </label>
        <label>
          Rows
          <select value={row} onChange={(e) => setRow(e.target.value as HeatmapField)}>
            {HEATMAP_ROWS.map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </label>
      </div>
      {column === row ? (
        <p className="muted">Pick different fields for columns and rows.</p>
      ) : (
        <Plot title={`Mean latency by ${row} and ${column}`} data={data} layout={layout} />
      )}
    </Panel>
  );
}

function Outliers({ data }: { data: TraceAnalytics }) {
  const bounds = data.statistics.outlier_bounds_ms;
  const found = useMemo(() => outliers(data.traces, bounds), [data.traces, bounds]);
  const scatter = useMemo(() => {
    const flagged = new Set(found);
    const normal = data.traces.filter((trace) => !flagged.has(trace));
    const points = (traces: Trace[]) => ({
      x: traces.map((trace) => trace.start_time),
      y: traces.map((trace) => trace.duration_ms),
      text: traces.map((trace) => trace.operation),
    });
    return [
      { type: 'scatter', mode: 'markers', name: 'Within bounds', ...points(normal) },
      { type: 'scatter', mode: 'markers', name: 'Outliers', marker: { symbol: 'x', size: 10 }, ...points(found) },
    ];
  }, [data.traces, found]);
  const layout = useMemo(
    () => ({
      yaxis: { title: { text: 'Latency (ms)' } },
      shapes: bounds
        ? [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: bounds.upper, y1: bounds.upper, line: { dash: 'dash' } }]
        : [],
    }),
    [bounds],
  );
  if (!bounds) {
    return (
      <Panel title="Outliers">
        <p className="muted">Outliers need at least four traces.</p>
      </Panel>
    );
  }
  return (
    <Panel title="Outliers">
      <p className="muted">
        Traces outside {ms(bounds.lower)} to {ms(bounds.upper)} (1.5 times the interquartile range beyond the
        quartiles).
      </p>
      <Plot title="Latency outliers" data={scatter} layout={layout} />
      {found.length === 0 ? (
        <p className="muted">No trace falls outside the bounds.</p>
      ) : (
        <TraceTable label="Outlier traces" traces={found.slice(0, PAGE_SIZE)} />
      )}
    </Panel>
  );
}

function Explorer({ traces }: { traces: Trace[] }) {
  const [search, setSearch] = useState('');
  const [sort, setSort] = useState<Sort>('Newest first');
  const [page, setPage] = useState(0);
  const shown = useMemo(() => explore(traces, search, sort), [traces, search, sort]);
  const pages = Math.max(1, Math.ceil(shown.length / PAGE_SIZE));
  const current = Math.min(page, pages - 1);
  return (
    <Panel title="Trace explorer">
      <div className="inline-form">
        <label>
          Trace ID or operation
          <input
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(0);
            }}
          />
        </label>
        <label>
          Order
          <select value={sort} onChange={(e) => setSort(e.target.value as Sort)}>
            {Object.keys(SORTS).map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </label>
      </div>
      <p className="muted">
        {shown.length} {shown.length === 1 ? 'trace' : 'traces'}
      </p>
      <TraceTable label="Traces" traces={shown.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE)} />
      <div className="pager">
        <button disabled={current === 0} onClick={() => setPage(current - 1)}>
          Previous
        </button>
        <span>
          Page {current + 1} of {pages}
        </span>
        <button disabled={current >= pages - 1} onClick={() => setPage(current + 1)}>
          Next
        </button>
      </div>
    </Panel>
  );
}

function TraceTable({ label, traces }: { label: string; traces: Trace[] }) {
  return (
    <table aria-label={label}>
      <thead>
        <tr>
          <th>When</th>
          <th>Operation</th>
          <th>Duration</th>
          <th>Outcome</th>
          <th>Profile</th>
          <th>Strategy</th>
          <th>Trace ID</th>
        </tr>
      </thead>
      <tbody>
        {traces.map((trace) => (
          <tr key={trace.span_id ?? `${trace.trace_id}-${trace.start_time}`}>
            <td>{when(trace.start_time)}</td>
            <td>{trace.operation}</td>
            <td>{ms(trace.duration_ms)}</td>
            <td>{trace.succeeded ? 'succeeded' : `failed: ${trace.error ?? 'no message'}`}</td>
            <td>{trace.profile ?? '—'}</td>
            <td>{trace.strategy ?? '—'}</td>
            <td>
              <code>{trace.trace_id ?? '—'}</code>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

interface RootCause {
  hypothesis: string;
  confidence: number;
  category: string;
  evidence: string[];
  affected_traces: string[];
  suggested_action: string;
}

interface Recommendation {
  priority: string;
  category: string;
  recommendation: string;
  details: string[];
  affected_components: string[];
}

interface RootCauseAnalysis {
  traces: number;
  failed: number;
  slow: number;
  failure_rate: number;
  slow_threshold_ms: number | null;
  root_causes: RootCause[];
  recommendations: Recommendation[];
}

function RootCauses({ tenant, query }: { tenant: string; query: URLSearchParams }) {
  const [includeSlow, setIncludeSlow] = useState(true);
  const [percentile, setPercentile] = useState('95');
  const [analysis, setAnalysis] = useState<RootCauseAnalysis>();
  const action = useAction();
  return (
    <Panel title="Root causes">
      <form
        className="inline-form"
        aria-label="Find root causes"
        onSubmit={(e) => {
          e.preventDefault();
          action.run(async () => {
            const value = Number(percentile);
            if (!Number.isInteger(value) || value < 50 || value > 99)
              throw new Error('The slow percentile must be a whole number from 50 to 99.');
            const params = new URLSearchParams(query);
            params.set('include_slow', String(includeSlow));
            params.set('slow_percentile', String(value));
            setAnalysis(
              await runtimeJson<RootCauseAnalysis>(`/admin/tenant/${seg(tenant)}/telemetry/root-causes?${params}`),
            );
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
      {!analysis && <p className="muted">Runs over the traces the filters above keep.</p>}
      {analysis && (
        <>
          <dl className="facts" aria-label="Root cause summary">
            <dt>Traces analyzed</dt>
            <dd>{analysis.traces}</dd>
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
                    <ul>
                      {cause.evidence.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                    <p>Suggested action: {cause.suggested_action}</p>
                    <p>Affected traces: {cause.affected_traces.join(', ')}</p>
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
        </>
      )}
    </Panel>
  );
}
