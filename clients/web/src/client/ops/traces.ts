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

export const SORTS = { 'Newest first': 'newest', 'Oldest first': 'oldest', 'Slowest first': 'slowest', 'Fastest first': 'fastest' } as const;
export type Sort = keyof typeof SORTS;

/** The traces whose trace ID or operation contains ``search`` (any case),
 * in ``sort`` order. */
export function explore(traces: Trace[], search: string, sort: Sort): Trace[] {
  const needle = search.trim().toLowerCase();
  const kept = traces.filter(
    (trace) =>
      !needle || trace.operation.toLowerCase().includes(needle) || (trace.trace_id ?? '').toLowerCase().includes(needle),
  );
  const by: Record<(typeof SORTS)[Sort], (a: Trace, b: Trace) => number> = {
    newest: (a, b) => b.start_time.localeCompare(a.start_time),
    oldest: (a, b) => a.start_time.localeCompare(b.start_time),
    slowest: (a, b) => b.duration_ms - a.duration_ms,
    fastest: (a, b) => a.duration_ms - b.duration_ms,
  };
  return [...kept].sort(by[SORTS[sort]]);
}
