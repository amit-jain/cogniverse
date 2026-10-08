/** A tenant's traces as the runtime reports them, and the figures the
 * Analytics view draws from them. */

export interface Trace {
  trace_id: string | null;
  span_id: string | null;
  start_time: string;
  duration_ms: number;
  operation: string;
  succeeded: boolean;
  profile: string | null;
  strategy: string | null;
  error: string | null;
}

export interface TraceAnalytics {
  facets: { operations: string[]; profiles: string[]; strategies: string[] };
  statistics: {
    requests: number;
    succeeded: number;
    failed: number;
    success_rate: number | null;
    latency_ms: Record<'mean' | 'min' | 'p50' | 'p75' | 'p90' | 'p95' | 'p99' | 'max', number | null>;
    outlier_bounds_ms: { lower: number; upper: number } | null;
    by_operation: { operation: string; count: number; mean_ms: number; p95_ms: number; error_rate: number }[];
  };
  traces: Trace[];
}

export const WINDOWS = { '1 min': 60_000, '5 min': 300_000, '15 min': 900_000, '1 hour': 3_600_000 } as const;
export type Window = keyof typeof WINDOWS;

export const GROUPS = ['operation', 'profile', 'strategy', 'status'] as const;
export type Group = (typeof GROUPS)[number];

export const HEATMAP_COLUMNS = ['hour', 'weekday', 'operation'] as const;
export const HEATMAP_ROWS = ['operation', 'profile', 'strategy', 'status', 'day'] as const;
export type HeatmapField = (typeof HEATMAP_COLUMNS)[number] | (typeof HEATMAP_ROWS)[number];

const WEEKDAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

/** The linear-interpolation quantile pandas computes by default. */
export function quantile(values: number[], q: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * q;
  const below = Math.floor(position);
  const above = Math.ceil(position);
  return sorted[below] + (sorted[above] - sorted[below]) * (position - below);
}

/** The value of ``field`` for a trace; a trace without one reads "unknown".
 * Times are read in UTC. */
export function fieldOf(trace: Trace, field: Group | HeatmapField): string {
  const start = new Date(trace.start_time);
  switch (field) {
    case 'status':
      return trace.succeeded ? 'succeeded' : 'failed';
    case 'hour':
      return String(start.getUTCHours()).padStart(2, '0');
    case 'weekday':
      return WEEKDAYS[start.getUTCDay()];
    case 'day':
      return start.toISOString().slice(0, 10);
    default:
      return trace[field] ?? 'unknown';
  }
}

export interface Bucket {
  start: string;
  requests: number;
  mean_ms: number;
  p50_ms: number;
  p95_ms: number;
}

/** Traces per ``window``-long bucket (aligned to the epoch), oldest first;
 * buckets without traces are left out. */
export function timeBuckets(traces: Trace[], window: Window): Bucket[] {
  const width = WINDOWS[window];
  const buckets = new Map<number, number[]>();
  for (const trace of traces) {
    const start = Math.floor(Date.parse(trace.start_time) / width) * width;
    buckets.set(start, [...(buckets.get(start) ?? []), trace.duration_ms]);
  }
  return [...buckets.entries()]
    .sort(([a], [b]) => a - b)
    .map(([start, durations]) => ({
      start: new Date(start).toISOString(),
      requests: durations.length,
      mean_ms: durations.reduce((sum, value) => sum + value, 0) / durations.length,
      p50_ms: quantile(durations, 0.5),
      p95_ms: quantile(durations, 0.95),
    }));
}

/** Durations per value of ``group``, in sorted value order. */
export function durationsBy(traces: Trace[], group: Group | null): { name: string; durations: number[] }[] {
  if (!group) return [{ name: 'all traces', durations: traces.map((trace) => trace.duration_ms) }];
  const groups = new Map<string, number[]>();
  for (const trace of traces) {
    const name = fieldOf(trace, group);
    groups.set(name, [...(groups.get(name) ?? []), trace.duration_ms]);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([name, durations]) => ({ name, durations }));
}

export interface Heatmap {
  columns: string[];
  rows: string[];
  /** Mean latency (ms) per row and column; ``null`` where no trace falls. */
  cells: (number | null)[][];
}

/** Mean latency per ``row`` and ``column`` value; weekdays in calendar
 * order, everything else sorted. */
export function heatmap(traces: Trace[], column: HeatmapField, row: HeatmapField): Heatmap {
  const order = (field: HeatmapField, values: Set<string>) =>
    field === 'weekday' ? WEEKDAYS.filter((day) => values.has(day)) : [...values].sort();
  const columns = order(column, new Set(traces.map((trace) => fieldOf(trace, column))));
  const rows = order(row, new Set(traces.map((trace) => fieldOf(trace, row))));
  const cells = rows.map((rowValue) =>
    columns.map((columnValue) => {
      const durations = traces
        .filter((trace) => fieldOf(trace, row) === rowValue && fieldOf(trace, column) === columnValue)
        .map((trace) => trace.duration_ms);
      return durations.length ? durations.reduce((sum, value) => sum + value, 0) / durations.length : null;
    }),
  );
  return { columns, rows, cells };
}

/** The traces outside ``bounds``, slowest first. */
export function outliers(traces: Trace[], bounds: { lower: number; upper: number } | null): Trace[] {
  if (!bounds) return [];
  return traces
    .filter((trace) => trace.duration_ms < bounds.lower || trace.duration_ms > bounds.upper)
    .sort((a, b) => b.duration_ms - a.duration_ms);
}

export const SORTS = {
  'Newest first': 'newest',
  'Oldest first': 'oldest',
  'Slowest first': 'slowest',
  'Fastest first': 'fastest',
  'Operation A-Z': 'operation',
  'Operation Z-A': 'operation-desc',
  'Failed first': 'failed',
  'Succeeded first': 'succeeded',
} as const;
export type Sort = keyof typeof SORTS;

export const SEARCH_SCOPES = ['Trace ID or operation', 'Trace ID', 'Operation'] as const;
export type SearchScope = (typeof SEARCH_SCOPES)[number];

/** The traces whose trace ID and/or operation (per ``scope``) contains
 * ``search`` (any case), in ``sort`` order; ties keep the newest first. */
export function explore(
  traces: Trace[],
  search: string,
  sort: Sort,
  scope: SearchScope = 'Trace ID or operation',
): Trace[] {
  const needle = search.trim().toLowerCase();
  const inId = (trace: Trace) => (trace.trace_id ?? '').toLowerCase().includes(needle);
  const inOperation = (trace: Trace) => trace.operation.toLowerCase().includes(needle);
  const matches =
    scope === 'Trace ID' ? inId : scope === 'Operation' ? inOperation : (trace: Trace) => inId(trace) || inOperation(trace);
  const kept = traces.filter((trace) => !needle || matches(trace));
  const newest = (a: Trace, b: Trace) => b.start_time.localeCompare(a.start_time);
  const by: Record<(typeof SORTS)[Sort], (a: Trace, b: Trace) => number> = {
    newest,
    oldest: (a, b) => a.start_time.localeCompare(b.start_time),
    slowest: (a, b) => b.duration_ms - a.duration_ms,
    fastest: (a, b) => a.duration_ms - b.duration_ms,
    operation: (a, b) => a.operation.localeCompare(b.operation) || newest(a, b),
    'operation-desc': (a, b) => b.operation.localeCompare(a.operation) || newest(a, b),
    failed: (a, b) => Number(a.succeeded) - Number(b.succeeded) || newest(a, b),
    succeeded: (a, b) => Number(b.succeeded) - Number(a.succeeded) || newest(a, b),
  };
  return [...kept].sort(by[SORTS[sort]]);
}

/** Tukey's fences: 1.5 interquartile ranges beyond the quartiles; ``null``
 * for fewer than four values. */
export function iqrBounds(values: number[]): { lower: number; upper: number } | null {
  if (values.length < 4) return null;
  const q1 = quantile(values, 0.25);
  const q3 = quantile(values, 0.75);
  return { lower: q1 - 1.5 * (q3 - q1), upper: q3 + 1.5 * (q3 - q1) };
}

export interface HourlyErrors {
  /** The UTC hour, as an ISO timestamp. */
  hour: string;
  requests: number;
  failed: number;
  /** Failed traces as a percentage of the hour's traces. */
  error_rate: number;
}

/** The failed share of the traces in each UTC hour, oldest first; hours
 * without traces are left out. */
export function hourlyErrorRates(traces: Trace[]): HourlyErrors[] {
  const hours = new Map<number, { requests: number; failed: number }>();
  for (const trace of traces) {
    const hour = Math.floor(Date.parse(trace.start_time) / 3_600_000) * 3_600_000;
    const entry = hours.get(hour) ?? { requests: 0, failed: 0 };
    entry.requests += 1;
    entry.failed += trace.succeeded ? 0 : 1;
    hours.set(hour, entry);
  }
  return [...hours.entries()]
    .sort(([a], [b]) => a - b)
    .map(([hour, { requests, failed }]) => ({
      hour: new Date(hour).toISOString(),
      requests,
      failed,
      error_rate: (failed / requests) * 100,
    }));
}

/** The empirical cumulative distribution of ``values``: each sorted value
 * and the share of values at or below it. */
export function ecdf(values: number[]): { x: number[]; y: number[] } {
  const sorted = [...values].sort((a, b) => a - b);
  return { x: sorted, y: sorted.map((_, index) => (index + 1) / sorted.length) };
}

/** The ``percentile`` (0-100) of the successful traces' latency, and of all
 * traces', the way root-cause analysis flags slow traces; ``null`` where
 * there are no traces to measure. */
export function slowThresholds(
  traces: Trace[],
  percentile: number,
): { succeeded: number | null; all: number | null } {
  const succeeded = traces.filter((trace) => trace.succeeded).map((trace) => trace.duration_ms);
  const all = traces.map((trace) => trace.duration_ms);
  return {
    succeeded: succeeded.length ? quantile(succeeded, percentile / 100) : null,
    all: all.length ? quantile(all, percentile / 100) : null,
  };
}

const ISO_TIMESTAMP = /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?/g;

/** ``evidence`` with the two ISO timestamps after "Time range:" written as
 * one readable range, in ``timeZone`` (the browser's by default). */
export function formatEvidence(evidence: string, timeZone?: string): string {
  const at = evidence.indexOf('Time range:');
  if (at < 0) return evidence;
  const stamps = evidence.slice(at).match(ISO_TIMESTAMP);
  if (!stamps || stamps.length < 2) return evidence;
  return `${evidence.slice(0, at)}Time range: ${formatRange(stamps[0], stamps[1], timeZone)}`;
}

/** ``text`` with the narrow and no-break spaces some locales put before
 * AM/PM written as plain spaces. */
export function plainSpaces(text: string): string {
  return text.replace(/[\u00a0\u202f]/g, ' ');
}

/** "7:45:12 PM": a time of day in the browser's time zone. */
export function clockTime(date: Date): string {
  return plainSpaces(date.toLocaleTimeString('en-US', TIME));
}

const DAY = { month: 'short', day: 'numeric', year: 'numeric' } as const;
const TIME = { hour: 'numeric', minute: '2-digit', second: '2-digit' } as const;

/** "Oct 5, 2026, 10:01:00 AM - 10:07:00 AM", or both dates when the range
 * spans days. */
export function formatRange(start: string, end: string, timeZone?: string): string {
  const parse = (value: string) => new Date(/[zZ]|[+-]\d{2}:\d{2}$/.test(value) ? value : `${value}Z`);
  const from = parse(start);
  const to = parse(end);
  const day = (date: Date) => plainSpaces(date.toLocaleDateString('en-US', { ...DAY, timeZone }));
  const time = (date: Date) => plainSpaces(date.toLocaleTimeString('en-US', { ...TIME, timeZone }));
  return day(from) === day(to)
    ? `${day(from)}, ${time(from)} - ${time(to)}`
    : `${day(from)}, ${time(from)} - ${day(to)}, ${time(to)}`;
}
