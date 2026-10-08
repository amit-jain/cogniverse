import { useEffect, useMemo, useRef, useState } from 'react';
import { analyticsHtml, analyticsJson, download, tracesCsv, type AnalyticsExport } from './analyticsExport';
import { Alert, Panel, messageOf, useLoad } from './common';
import { RuntimeRequestError, runtimeJson, seg } from './http';
import { Bars, percent } from './metrics';
import { Plot } from './Plot';
import { RootCauses, type CachedAnalysis, type PhoenixLinks } from './rootCauses';
import { TenantChooser } from './tenants';
import {
  DEFAULT_PRESET,
  PRESETS,
  describeRange,
  toUtcInput,
  windowQuery,
  type Preset,
  type TimeRange,
} from './timeRange';
import {
  GROUPS,
  HEATMAP_COLUMNS,
  HEATMAP_ROWS,
  SEARCH_SCOPES,
  SORTS,
  WINDOWS,
  durationsBy,
  ecdf,
  explore,
  clockTime,
  heatmap,
  hourlyErrorRates,
  iqrBounds,
  outliers,
  quantile,
  timeBuckets,
  type Group,
  type HeatmapField,
  type SearchScope,
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

/** How long a read may take before the view says its figures may be stale. */
export const SLOW_READ_MS = 5_500;
/** The success-rate target the overview measures against. */
export const SUCCESS_TARGET = 0.95;

export function AnalyticsView() {
  const [tenant, setTenant] = useState('');
  return (
    <div className="ops-view">
      <TenantChooser action="Show traces" onChoose={setTenant} />
      {tenant && <Traces key={tenant} tenant={tenant} />}
    </div>
  );
}

interface Loaded {
  data: TraceAnalytics;
  /** The window and filters the data was read with. */
  query: URLSearchParams;
  window: { start: string; end: string };
  at: Date;
}

/** The tenant's traces in ``range`` as ``filters`` keep them, re-read on
 * ``reload``; ``slow`` once a read has taken longer than ``SLOW_READ_MS``. */
function useTraces(tenant: string, range: TimeRange, filters: Filters) {
  const [loaded, setLoaded] = useState<Loaded>();
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [slow, setSlow] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let window: { start: string; end: string };
    try {
      window = windowQuery(range);
    } catch (e) {
      setError(messageOf(e));
      setLoading(false);
      return;
    }
    const query = new URLSearchParams({ ...window, operation: filters.operation });
    filters.profiles.forEach((value) => query.append('profile', value));
    filters.strategies.forEach((value) => query.append('strategy', value));
    setLoading(true);
    setSlow(false);
    const timer = setTimeout(() => setSlow(true), SLOW_READ_MS);
    runtimeJson<TraceAnalytics>(`/admin/tenant/${seg(tenant)}/telemetry/traces?${query}`, {
      signal: controller.signal,
    })
      .then((data) => {
        setLoaded({ data, query, window, at: new Date() });
        setError('');
      })
      .catch((e: unknown) => {
        if (controller.signal.aborted) return;
        setLoaded(undefined);
        // A telemetry backend or runtime that failed the read may answer the
        // next one.
        const transient = !(e instanceof RuntimeRequestError) || e.status >= 500;
        setError(transient ? `${messageOf(e)} Refresh to retry.` : messageOf(e));
      })
      .finally(() => {
        clearTimeout(timer);
        if (!controller.signal.aborted) {
          setLoading(false);
          setSlow(false);
        }
      });
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [tenant, range, filters, attempt]);
  return { loaded, error, loading, slow, reload: () => setAttempt((n) => n + 1) };
}

function Traces({ tenant }: { tenant: string }) {
  const [range, setRange] = useState<TimeRange>({ preset: DEFAULT_PRESET });
  const [filters, setFilters] = useState<Filters>({ operation: '', profiles: [], strategies: [] });
  const [section, setSection] = useState<Section>('Overview');
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [interval, setInterval_] = useState(30);
  const [raw, setRaw] = useState(false);
  const [cached, setCached] = useState<CachedAnalysis>();
  const traces = useTraces(tenant, range, filters);
  const links = useLoad(
    (signal) => runtimeJson<PhoenixLinks>(`/admin/tenant/${seg(tenant)}/telemetry/phoenix`, { signal }),
    [tenant],
  );
  const reload = useRef(traces.reload);
  reload.current = traces.reload;
  const busy = useRef(traces.loading);
  busy.current = traces.loading;
  useEffect(() => {
    if (!autoRefresh) return;
    const timer = setInterval(() => {
      if (!busy.current) reload.current();
    }, interval * 1000);
    return () => clearInterval(timer);
  }, [autoRefresh, interval]);
  const loaded = traces.loaded;
  const data = loaded?.data;
  const statistics = data?.statistics;
  const basis = useMemo(
    () => (data?.traces ?? []).map((trace) => trace.span_id ?? `${trace.trace_id}@${trace.start_time}`).join(','),
    [data],
  );
  const report = (): AnalyticsExport | undefined =>
    loaded && {
      tenant,
      window: loaded.window,
      filters,
      analytics: loaded.data,
      rootCauses: cached?.signature.startsWith(`${basis}|`) ? cached.analysis : undefined,
    };
  const fileName = (extension: string) => `traces-${tenant.replace(/[^A-Za-z0-9_-]/g, '_')}.${extension}`;
  return (
    <>
      <Panel title={`Traces of ${tenant}`} actions={<button onClick={traces.reload}>Refresh</button>}>
        <FilterForm
          key={JSON.stringify(data?.facets ?? null)}
          range={range}
          facets={data?.facets}
          filters={filters}
          onApply={(nextRange, next) => {
            setRange(nextRange);
            setFilters(next);
          }}
        />
        <div className="inline-form" aria-label="Refresh and data">
          <label className="check">
            <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
            Refresh automatically
          </label>
          <label>
            Every (seconds)
            <input
              type="number"
              min={5}
              max={300}
              step={5}
              value={interval}
              onChange={(e) => setInterval_(Math.min(300, Math.max(5, Number(e.target.value) || 5)))}
            />
          </label>
          <label className="check">
            <input type="checkbox" checked={raw} onChange={(e) => setRaw(e.target.checked)} />
            Show raw data
          </label>
          <button type="button" disabled={!loaded} onClick={() => download(fileName('json'), 'application/json', analyticsJson(report()!))}>
            Download JSON
          </button>
          <button type="button" disabled={!loaded} onClick={() => download(fileName('csv'), 'text/csv', tracesCsv(loaded!.data.traces))}>
            Download CSV
          </button>
          <button type="button" disabled={!loaded} onClick={() => download(fileName('html'), 'text/html', analyticsHtml(report()!))}>
            Download HTML
          </button>
        </div>
        {loaded && (
          <p className="caption" aria-label="Last refreshed">
            {describeRange(range)}; last refreshed {clockTime(loaded.at)}.
          </p>
        )}
        {traces.slow && (
          <p className="alert warning" aria-label="Slow read">
            The telemetry backend is slow to answer
            {loaded ? `; the figures below are from ${clockTime(loaded.at)} and may be stale` : ''}. Still
            reading…
          </p>
        )}
        {traces.error && <Alert>{traces.error}</Alert>}
        {links.error && <p className="alert warning">Phoenix links are unavailable: {links.error}</p>}
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
        {statistics && statistics.requests > 0 && (
          <p className="caption" aria-label="Success target">
            {targetLine(statistics.success_rate ?? 0)}
          </p>
        )}
      </Panel>
      {raw && data && <RawData traces={data.traces} />}
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
          {section === 'Root causes' && (
            <RootCauses
              tenant={tenant}
              query={loaded.query}
              basis={basis}
              traces={data.traces}
              links={links.data}
              cached={cached}
              onAnalysis={setCached}
            />
          )}
        </>
      )}
    </>
  );
}

/** "Succeeded 1.7 points below the 95% target." */
export function targetLine(successRate: number): string {
  const points = (successRate - SUCCESS_TARGET) * 100;
  if (Math.abs(points) < 0.05) return `On the ${percent(SUCCESS_TARGET)} success target.`;
  return `${Math.abs(points).toFixed(1)} points ${points < 0 ? 'below' : 'above'} the ${percent(SUCCESS_TARGET)} success target.`;
}

function RawData({ traces }: { traces: Trace[] }) {
  return (
    <Panel title="Raw data">
      <div className="raw-data">
        <table aria-label="Raw traces">
          <thead>
            <tr>
              {RAW_COLUMNS.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {traces.map((trace) => (
              <tr key={trace.span_id ?? `${trace.trace_id}-${trace.start_time}`}>
                {RAW_COLUMNS.map((column) => (
                  <td key={column}>{trace[column] === null ? '' : String(trace[column])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

const RAW_COLUMNS = [
  'start_time',
  'trace_id',
  'span_id',
  'operation',
  'duration_ms',
  'succeeded',
  'profile',
  'strategy',
  'error',
] as const;

function FilterForm({
  range,
  facets,
  filters,
  onApply,
}: {
  range: TimeRange;
  facets?: TraceAnalytics['facets'];
  filters: Filters;
  onApply: (range: TimeRange, filters: Filters) => void;
}) {
  const [draft, setDraft] = useState(filters);
  const [rangeDraft, setRangeDraft] = useState<TimeRange>(range);
  const custom = !('preset' in rangeDraft);
  const selected = (select: HTMLSelectElement) => [...select.selectedOptions].map((option) => option.value);
  return (
    <form
      className="inline-form"
      aria-label="Trace filters"
      onSubmit={(e) => {
        e.preventDefault();
        onApply(rangeDraft, { ...draft, operation: draft.operation.trim() });
      }}
    >
      <label>
        Window
        <select
          value={custom ? 'Custom range' : (rangeDraft as { preset: Preset }).preset}
          onChange={(e) => {
            const value = e.target.value;
            if (value === 'Custom range') {
              const now = new Date();
              setRangeDraft({ start: toUtcInput(new Date(now.getTime() - 3_600_000)), end: toUtcInput(now) });
            } else setRangeDraft({ preset: value as Preset });
          }}
        >
          {Object.keys(PRESETS).map((name) => (
            <option key={name}>{name}</option>
          ))}
          <option>Custom range</option>
        </select>
      </label>
      {custom && (
        <>
          <label>
            Start (UTC)
            <input
              type="datetime-local"
              value={rangeDraft.start}
              onChange={(e) => setRangeDraft({ ...rangeDraft, start: e.target.value })}
            />
          </label>
          <label>
            End (UTC)
            <input
              type="datetime-local"
              value={rangeDraft.end}
              onChange={(e) => setRangeDraft({ ...rangeDraft, end: e.target.value })}
            />
          </label>
        </>
      )}
      <label>
        Operation (regular expression)
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
  const donut = useMemo(
    () => [
      {
        type: 'pie',
        hole: 0.4,
        labels: operations.map((row) => row.operation),
        values: operations.map((row) => row.count),
        textinfo: 'label+percent',
      },
    ],
    [operations],
  );
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
      <Plot title="Operation share" data={donut} />
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
  const violins = useMemo(
    () =>
      groups.map((entry) => ({
        type: 'violin',
        name: entry.name,
        y: entry.durations,
        box: { visible: true },
        meanline: { visible: true },
        points: 'outliers',
      })),
    [groups],
  );
  const cumulative = useMemo(() => {
    const durations = traces.map((trace) => trace.duration_ms);
    const curve = ecdf(durations);
    return {
      data: [{ type: 'scatter', mode: 'lines', name: 'Share of traces', x: curve.x, y: curve.y, line: { shape: 'hv' } }],
      layout: {
        ...MS_AXIS,
        yaxis: { title: { text: 'Share of traces at or below' }, tickformat: '.0%' },
        shapes: ECDF_PERCENTILES.map((p) => ({
          type: 'line',
          yref: 'paper',
          y0: 0,
          y1: 1,
          x0: quantile(durations, p / 100),
          x1: quantile(durations, p / 100),
          line: { dash: 'dash' },
        })),
        annotations: ECDF_PERCENTILES.map((p) => ({
          x: quantile(durations, p / 100),
          yref: 'paper',
          y: 1,
          text: `P${p}`,
          showarrow: false,
        })),
      },
    };
  }, [traces]);
  return (
    <Panel title="Distribution">
      <details className="explainer">
        <summary>Reading these charts</summary>
        <p>
          The histogram counts traces per latency range: peaks are the common latencies, and more than one peak
          means traces fall into separate performance modes.
        </p>
        <p>
          The box spans the middle half of the latencies (first to third quartile) with the median inside; whiskers
          reach 1.5 interquartile ranges, and points beyond them are outliers.
        </p>
        <p>
          The violin's width at a latency is how many traces take that long, with the box and the mean line inside:
          a symmetric violin is steady performance, a long upper tail a few slow traces, several bulges several
          modes.
        </p>
        <p>
          The cumulative curve reads "this share of traces finished in this many milliseconds or less"; the dashed
          lines mark P50, P90, P95 and P99. A steep curve is consistent latency, a gradual one high variability.
        </p>
      </details>
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
      <Plot title="Latency density" data={violins} layout={VIOLIN} />
      <Plot title="Cumulative latency" data={cumulative.data} layout={cumulative.layout} />
    </Panel>
  );
}

const ECDF_PERCENTILES = [50, 90, 95, 99];
const VIOLIN = { yaxis: { title: { text: 'Latency (ms)' } } };
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
      <details className="explainer">
        <summary>About profile and strategy</summary>
        <p>
          Profile is the processing profile a search ran with (for example video_colpali_smol500_mv_frame), and
          strategy its ranking strategy (for example hybrid or bm25). Only search traces record them; other traces
          read "unknown", as do traces whose attributes were not captured.
        </p>
      </details>
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

const OUTLIER_METRICS = ['Latency', 'Hourly error rate'] as const;
type OutlierMetric = (typeof OUTLIER_METRICS)[number];
const REFERENCE_PERCENTILES = [50, 95, 99];

/** Horizontal reference lines across a chart, each labelled. */
function referenceLines(lines: { y: number; text: string; dash: string }[]) {
  return {
    shapes: lines.map((line) => ({
      type: 'line',
      xref: 'paper',
      x0: 0,
      x1: 1,
      y0: line.y,
      y1: line.y,
      line: { dash: line.dash },
    })),
    annotations: lines.map((line) => ({ xref: 'paper', x: 1, y: line.y, text: line.text, showarrow: false, xanchor: 'right' })),
  };
}

function Outliers({ data }: { data: TraceAnalytics }) {
  const [metric, setMetric] = useState<OutlierMetric>('Latency');
  return (
    <Panel title="Outliers">
      <details className="explainer">
        <summary>How outliers are found</summary>
        <p>
          Tukey's rule on the interquartile range (IQR): Q1 and Q3 are the 25th and 75th percentiles, IQR = Q3 - Q1,
          and anything above Q3 + 1.5 × IQR or below Q1 - 1.5 × IQR is an outlier. The 1.5 multiplier flags about 2-3%
          of normally distributed values.
        </p>
        <p>
          Unlike a standard-deviation rule it is robust to the extreme values it looks for, suits skewed latencies and
          assumes no particular distribution.
        </p>
      </details>
      <div className="inline-form">
        <label>
          Metric
          <select value={metric} onChange={(e) => setMetric(e.target.value as OutlierMetric)}>
            {OUTLIER_METRICS.map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </label>
      </div>
      {metric === 'Latency' ? <LatencyOutliers data={data} /> : <ErrorRateOutliers traces={data.traces} />}
    </Panel>
  );
}

function LatencyOutliers({ data }: { data: TraceAnalytics }) {
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
  const layout = useMemo(() => {
    const durations = data.traces.map((trace) => trace.duration_ms);
    return {
      yaxis: { title: { text: 'Latency (ms)' } },
      ...referenceLines(
        bounds
          ? [
              { y: bounds.upper, text: `Outlier bound (${ms(bounds.upper)})`, dash: 'dash' },
              ...REFERENCE_PERCENTILES.map((p) => {
                const value = quantile(durations, p / 100);
                return { y: value, text: `P${p} (${ms(value)})`, dash: 'dot' };
              }),
            ]
          : [],
      ),
    };
  }, [data.traces, bounds]);
  if (!bounds) return <p className="muted">Outliers need at least four traces.</p>;
  return (
    <>
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
    </>
  );
}

function ErrorRateOutliers({ traces }: { traces: Trace[] }) {
  const { hours, bounds, flagged, scatter, layout } = useMemo(() => {
    const hours = hourlyErrorRates(traces);
    const rates = hours.map((hour) => hour.error_rate);
    const bounds = iqrBounds(rates);
    const flagged = (rate: number) => bounds !== null && (rate < bounds.lower || rate > bounds.upper);
    const series = (name: string, kept: typeof hours, marker: Record<string, unknown>) => ({
      type: 'scatter',
      mode: 'markers',
      name,
      x: kept.map((hour) => hour.hour),
      y: kept.map((hour) => hour.error_rate),
      text: kept.map((hour) => `${hour.failed} of ${hour.requests} failed`),
      marker,
    });
    return {
      hours,
      bounds,
      flagged,
      scatter: [
        series('Within bounds', hours.filter((hour) => !flagged(hour.error_rate)), { size: 8 }),
        series('Outliers', hours.filter((hour) => flagged(hour.error_rate)), { symbol: 'x', size: 10 }),
      ],
      layout: {
        yaxis: { title: { text: 'Error rate (%)' } },
        ...referenceLines(
          bounds
            ? [
                { y: bounds.upper, text: `Outlier bound (${bounds.upper.toFixed(1)}%)`, dash: 'dash' },
                ...REFERENCE_PERCENTILES.map((p) => {
                  const value = quantile(rates, p / 100);
                  return { y: value, text: `P${p} (${value.toFixed(1)}%)`, dash: 'dot' };
                }),
              ]
            : [],
        ),
      },
    };
  }, [traces]);
  return (
    <>
      <p className="muted">
        {bounds
          ? `Hours whose error rate falls outside ${bounds.lower.toFixed(1)}% to ${bounds.upper.toFixed(1)}%.`
          : 'Error-rate outliers need at least four hours with traces.'}
      </p>
      <Plot title="Error rate outliers" data={scatter} layout={layout} />
      <table aria-label="Error rate by hour">
        <thead>
          <tr>
            <th>Hour (UTC)</th>
            <th>Traces</th>
            <th>Failed</th>
            <th>Error rate</th>
            <th>Outlier</th>
          </tr>
        </thead>
        <tbody>
          {hours.map((hour) => (
            <tr key={hour.hour}>
              <td>{hour.hour.slice(0, 13).replace('T', ' ')}:00</td>
              <td>{hour.requests}</td>
              <td>{hour.failed}</td>
              <td>{hour.error_rate.toFixed(1)}%</td>
              <td>{flagged(hour.error_rate) ? 'yes' : 'no'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function Explorer({ traces }: { traces: Trace[] }) {
  const [search, setSearch] = useState('');
  const [scope, setScope] = useState<SearchScope>('Trace ID or operation');
  const [sort, setSort] = useState<Sort>('Newest first');
  const [page, setPage] = useState(0);
  const shown = useMemo(() => explore(traces, search, sort, scope), [traces, search, sort, scope]);
  const pages = Math.max(1, Math.ceil(shown.length / PAGE_SIZE));
  const current = Math.min(page, pages - 1);
  return (
    <Panel title="Trace explorer">
      <div className="inline-form">
        <label>
          Search in
          <select
            value={scope}
            onChange={(e) => {
              setScope(e.target.value as SearchScope);
              setPage(0);
            }}
          >
            {SEARCH_SCOPES.map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </label>
        <label>
          {scope}
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
