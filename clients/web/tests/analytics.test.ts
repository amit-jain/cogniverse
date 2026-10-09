import { describe, expect, it } from 'vitest';
import { analyticsHtml, analyticsJson, tracesCsv } from '../src/client/ops/analyticsExport';
import { targetLine } from '../src/client/ops/AnalyticsView';
import { timeRangeQuery, traceIdsQuery } from '../src/client/ops/rootCauses';
import { describeRange, windowQuery } from '../src/client/ops/timeRange';
import {
  ecdf,
  explore,
  formatEvidence,
  formatRange,
  hourlyErrorRates,
  iqrBounds,
  slowThresholds,
  type Trace,
  type TraceAnalytics,
} from '../src/client/ops/traces';

function trace(start: string, duration: number, overrides: Partial<Trace> = {}): Trace {
  return {
    trace_id: `t-${start}`,
    span_id: `s-${start}`,
    start_time: start,
    duration_ms: duration,
    operation: 'search_service.search',
    succeeded: true,
    profile: 'video',
    strategy: 'hybrid',
    error: null,
    ...overrides,
  };
}

const TRACES = [
  trace('2026-10-05T12:10:00+00:00', 50, { operation: 'agent.dispatch' }),
  trace('2026-10-05T11:40:00+00:00', 400, { succeeded: false, error: 'timed out' }),
  trace('2026-10-05T11:05:00+00:00', 300),
  trace('2026-10-05T10:30:00+00:00', 100, { succeeded: false, error: 'a "quoted", line\nbreak' }),
  trace('2026-10-05T10:01:00+00:00', 200),
];

describe('explore', () => {
  it('searches the trace ID, the operation or both', () => {
    const ids = (found: Trace[]) => found.map((t) => t.start_time.slice(11, 16));
    expect(ids(explore(TRACES, 'DISPATCH', 'Newest first', 'Operation'))).toEqual(['12:10']);
    expect(ids(explore(TRACES, 'dispatch', 'Newest first', 'Trace ID'))).toEqual([]);
    expect(ids(explore(TRACES, '11:', 'Newest first', 'Trace ID'))).toEqual(['11:40', '11:05']);
    expect(ids(explore(TRACES, '11:', 'Newest first', 'Operation'))).toEqual([]);
  });

  it('sorts by operation and by outcome, newest first among equals', () => {
    const ids = (sort: Parameters<typeof explore>[2]) => explore(TRACES, '', sort).map((t) => t.start_time.slice(11, 16));
    expect(ids('Operation A-Z')).toEqual(['12:10', '11:40', '11:05', '10:30', '10:01']);
    expect(ids('Operation Z-A')).toEqual(['11:40', '11:05', '10:30', '10:01', '12:10']);
    expect(ids('Failed first')).toEqual(['11:40', '10:30', '12:10', '11:05', '10:01']);
    expect(ids('Succeeded first')).toEqual(['12:10', '11:05', '10:01', '11:40', '10:30']);
  });
});

describe('outlier figures', () => {
  it('rates failures per UTC hour and fences them with the IQR', () => {
    expect(hourlyErrorRates(TRACES)).toEqual([
      { hour: '2026-10-05T10:00:00.000Z', requests: 2, failed: 1, error_rate: 50 },
      { hour: '2026-10-05T11:00:00.000Z', requests: 2, failed: 1, error_rate: 50 },
      { hour: '2026-10-05T12:00:00.000Z', requests: 1, failed: 0, error_rate: 0 },
    ]);
    expect(iqrBounds([50, 50, 0])).toBe(null);
    expect(iqrBounds([10, 20, 30, 40])).toEqual({ lower: -5, upper: 55 });
  });

  it('builds the cumulative curve and the slow thresholds root-cause analysis uses', () => {
    expect(ecdf([300, 100, 200, 400])).toEqual({ x: [100, 200, 300, 400], y: [0.25, 0.5, 0.75, 1] });
    // Successful 50, 200, 300; all 50, 100, 200, 300, 400.
    expect(slowThresholds(TRACES, 50)).toEqual({ succeeded: 200, all: 200 });
    expect(slowThresholds(TRACES, 75)).toEqual({ succeeded: 250, all: 300 });
    expect(slowThresholds([], 95)).toEqual({ succeeded: null, all: null });
  });
});

describe('evidence', () => {
  it('writes the ISO range after "Time range:" readably, leaving other evidence alone', () => {
    expect(
      formatEvidence('3 failures. Time range: 2026-10-05T10:01:00+00:00 to 2026-10-05T10:07:30.5+00:00', 'UTC'),
    ).toBe('3 failures. Time range: Oct 5, 2026, 10:01:00 AM - 10:07:30 AM');
    expect(formatRange('2026-10-04T23:59:00', '2026-10-05T00:01:00', 'UTC')).toBe(
      'Oct 4, 2026, 11:59:00 PM - Oct 5, 2026, 12:01:00 AM',
    );
    expect(formatEvidence('Average duration: 400.0ms', 'UTC')).toBe('Average duration: 400.0ms');
  });

  it('writes the Phoenix queries for trace ids and a time range', () => {
    expect(traceIdsQuery(['a1', 'b2'])).toBe('trace_id == "a1" or trace_id == "b2"');
    expect(timeRangeQuery('2026-10-05T10:01:00+00:00', '2026-10-05T10:07:00+00:00')).toBe(
      'timestamp >= "2026-10-05T10:01:00+00:00" and timestamp <= "2026-10-05T10:07:00+00:00"',
    );
  });
});

describe('time range', () => {
  const now = new Date('2026-10-05T12:00:00.000Z');

  it('reads a preset back from now, and a custom range as UTC', () => {
    expect(windowQuery({ preset: 'Last 15 minutes' }, now)).toEqual({
      start: '2026-10-05T11:45:00.000Z',
      end: '2026-10-05T12:00:00.000Z',
    });
    expect(windowQuery({ start: '2026-10-01T08:30', end: '2026-10-02T08:30' }, now)).toEqual({
      start: '2026-10-01T08:30:00.000Z',
      end: '2026-10-02T08:30:00.000Z',
    });
    expect(describeRange({ start: '2026-10-01T08:30', end: '2026-10-02T08:30' })).toBe(
      '2026-10-01 08:30 to 2026-10-02 08:30 UTC',
    );
  });

  it('refuses a custom range that is empty, reversed or longer than 30 days', () => {
    expect(() => windowQuery({ start: '', end: '2026-10-02T08:30' })).toThrow(
      'Give both the start and the end of the custom range.',
    );
    expect(() => windowQuery({ start: '2026-10-02T08:30', end: '2026-10-02T08:30' })).toThrow(
      'The custom range must start before it ends.',
    );
    expect(() => windowQuery({ start: '2026-08-01T00:00', end: '2026-10-02T00:00' })).toThrow(
      'The custom range may span at most 30 days.',
    );
  });
});

describe('success target', () => {
  it('measures the success rate against 95%', () => {
    expect([0.833, 0.95, 0.99].map(targetLine)).toEqual([
      '11.7 points below the 95.0% success target.',
      'On the 95.0% success target.',
      '4.0 points above the 95.0% success target.',
    ]);
  });
});

const ANALYTICS: TraceAnalytics = {
  facets: { operations: ['search_service.search'], profiles: ['video'], strategies: ['hybrid'] },
  statistics: {
    requests: 2,
    succeeded: 1,
    failed: 1,
    success_rate: 0.5,
    latency_ms: { mean: 150, min: 100, p50: 150, p75: 175, p90: 190, p95: 195, p99: 199, max: 200 },
    outlier_bounds_ms: null,
    by_operation: [{ operation: 'search_service.search', count: 2, mean_ms: 150, p95_ms: 195, error_rate: 0.5 }],
  },
  traces: [TRACES[3], TRACES[4]],
};

describe('export', () => {
  const report = {
    tenant: 'acme:<prod>',
    window: { start: '2026-10-05T06:00:00.000Z', end: '2026-10-05T12:00:00.000Z' },
    filters: { operation: 'search', profiles: [], strategies: ['hybrid'] },
    analytics: ANALYTICS,
  };

  it('writes the traces as CSV, quoting what CSV requires', () => {
    expect(tracesCsv(ANALYTICS.traces)).toBe(
      'trace_id,span_id,start_time,duration_ms,operation,succeeded,profile,strategy,error\r\n' +
        't-2026-10-05T10:30:00+00:00,s-2026-10-05T10:30:00+00:00,2026-10-05T10:30:00+00:00,100,' +
        'search_service.search,false,video,hybrid,"a ""quoted"", line\nbreak"\r\n' +
        't-2026-10-05T10:01:00+00:00,s-2026-10-05T10:01:00+00:00,2026-10-05T10:01:00+00:00,200,' +
        'search_service.search,true,video,hybrid,\r\n',
    );
  });

  it('writes the whole report as JSON that reads back the same', () => {
    expect(JSON.parse(analyticsJson(report))).toEqual(report);
  });

  it('writes an escaped HTML report naming the tenant, window and every trace', () => {
    const html = analyticsHtml(report);
    expect(html.startsWith('<!doctype html><html><head><meta charset="utf-8"><title>Cogniverse traces of acme:&#60;prod&#62;</title>')).toBe(
      true,
    );
    expect(html.match(/<caption>[^<]*<\/caption>/g)).toEqual([
      '<caption>Summary</caption>',
      '<caption>Traces by operation</caption>',
      '<caption>Traces</caption>',
    ]);
    expect(html).toContain(
      '<p>2026-10-05T06:00:00.000Z to 2026-10-05T12:00:00.000Z; operation search; profiles any; strategies hybrid.</p>',
    );
    expect(html).toContain('<td>a &#34;quoted&#34;, line\nbreak</td>');
  });
});
