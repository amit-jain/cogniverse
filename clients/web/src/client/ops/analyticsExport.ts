import type { RootCauseAnalysis } from './rootCauses';
import type { Trace, TraceAnalytics } from './traces';

/** What the Analytics view exports: the tenant's traces in the window as the
 * filters keep them, their statistics and, once run, the root causes. */
export interface AnalyticsExport {
  tenant: string;
  window: { start: string; end: string };
  filters: { operation: string; profiles: string[]; strategies: string[] };
  analytics: TraceAnalytics;
  rootCauses?: RootCauseAnalysis;
}

export const TRACE_COLUMNS = [
  'trace_id',
  'span_id',
  'start_time',
  'duration_ms',
  'operation',
  'succeeded',
  'profile',
  'strategy',
  'error',
] as const satisfies readonly (keyof Trace)[];

function csvCell(value: unknown): string {
  if (value === null || value === undefined) return '';
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** The traces as CSV, one row per trace under a header of ``TRACE_COLUMNS``. */
export function tracesCsv(traces: Trace[]): string {
  const lines = [TRACE_COLUMNS.join(',')];
  for (const trace of traces) lines.push(TRACE_COLUMNS.map((column) => csvCell(trace[column])).join(','));
  return `${lines.join('\r\n')}\r\n`;
}

export function analyticsJson(report: AnalyticsExport): string {
  return `${JSON.stringify(report, null, 2)}\n`;
}

const escape = (value: unknown) =>
  String(value ?? '').replace(/[&<>"']/g, (char) => `&#${char.charCodeAt(0)};`);

function table(caption: string, headers: string[], rows: unknown[][]): string {
  return (
    `<table><caption>${escape(caption)}</caption><thead><tr>${headers.map((h) => `<th>${escape(h)}</th>`).join('')}` +
    `</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${escape(cell)}</td>`).join('')}</tr>`).join('')}` +
    '</tbody></table>'
  );
}

/** A self-contained HTML report of ``report``. */
export function analyticsHtml(report: AnalyticsExport): string {
  const { statistics, traces } = report.analytics;
  const parts = [
    `<h1>Traces of ${escape(report.tenant)}</h1>`,
    `<p>${escape(report.window.start)} to ${escape(report.window.end)}; operation ${escape(
      report.filters.operation || 'any',
    )}; profiles ${escape(report.filters.profiles.join(', ') || 'any')}; strategies ${escape(
      report.filters.strategies.join(', ') || 'any',
    )}.</p>`,
    table(
      'Summary',
      ['Traces', 'Succeeded', 'Failed', 'Mean ms', 'P95 ms'],
      [[statistics.requests, statistics.succeeded, statistics.failed, statistics.latency_ms.mean, statistics.latency_ms.p95]],
    ),
    table(
      'Traces by operation',
      ['Operation', 'Traces', 'Mean ms', 'P95 ms', 'Error rate'],
      statistics.by_operation.map((row) => [row.operation, row.count, row.mean_ms, row.p95_ms, row.error_rate]),
    ),
    table(
      'Traces',
      [...TRACE_COLUMNS],
      traces.map((trace) => TRACE_COLUMNS.map((column) => trace[column])),
    ),
  ];
  const causes = report.rootCauses;
  if (causes) {
    parts.push(
      table(
        'Root causes',
        ['Hypothesis', 'Confidence', 'Category', 'Suggested action', 'Affected traces'],
        causes.root_causes.map((cause) => [
          cause.hypothesis,
          cause.confidence,
          cause.category,
          cause.suggested_action,
          cause.affected_traces.join(' '),
        ]),
      ),
      table(
        'Recommendations',
        ['Priority', 'Category', 'Recommendation', 'Details'],
        causes.recommendations.map((item) => [item.priority, item.category, item.recommendation, item.details.join('; ')]),
      ),
    );
  }
  return (
    '<!doctype html><html><head><meta charset="utf-8"><title>Cogniverse traces of ' +
    `${escape(report.tenant)}</title></head><body>${parts.join('\n')}</body></html>\n`
  );
}

/** Hands ``text`` to the browser as a file named ``name``. */
export function download(name: string, type: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
