import { describe, expect, it } from 'vitest';
import { durationsBy, explore, fieldOf, heatmap, outliers, quantile, timeBuckets, type Trace } from '../src/client/ops/traces';

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

// Sunday 2026-10-04 and Monday 2026-10-05, in UTC.
const TRACES = [
  trace('2026-10-05T10:07:00+00:00', 400, { profile: null }),
  trace('2026-10-05T10:03:00+00:00', 100),
  trace('2026-10-05T10:01:00+00:00', 300, { operation: 'agent.dispatch', succeeded: false, error: 'down' }),
  trace('2026-10-04T23:59:59+00:00', 200, { strategy: 'bm25' }),
];

describe('quantile', () => {
  it('interpolates linearly between ranks', () => {
    expect([0, 0.25, 0.5, 0.95, 1].map((q) => quantile([400, 100, 300, 200], q))).toEqual([100, 175, 250, 385, 400]);
  });
});

describe('fieldOf', () => {
  it('reads times in UTC and a missing value as unknown', () => {
    expect(['hour', 'weekday', 'day', 'status', 'profile', 'strategy'].map((f) => fieldOf(TRACES[3], f as never))).toEqual([
      '23',
      'Sunday',
      '2026-10-04',
      'succeeded',
      'video',
      'bm25',
    ]);
    expect([fieldOf(TRACES[0], 'profile'), fieldOf(TRACES[2], 'status')]).toEqual(['unknown', 'failed']);
  });
});

describe('timeBuckets', () => {
  it('buckets by window aligned to the epoch, oldest first, without empty buckets', () => {
    expect(timeBuckets(TRACES, '5 min')).toEqual([
      { start: '2026-10-04T23:55:00.000Z', requests: 1, mean_ms: 200, p50_ms: 200, p95_ms: 200 },
      { start: '2026-10-05T10:00:00.000Z', requests: 2, mean_ms: 200, p50_ms: 200, p95_ms: 290 },
      { start: '2026-10-05T10:05:00.000Z', requests: 1, mean_ms: 400, p50_ms: 400, p95_ms: 400 },
    ]);
    expect(timeBuckets(TRACES, '1 hour').map((b) => [b.start, b.requests])).toEqual([
      ['2026-10-04T23:00:00.000Z', 1],
      ['2026-10-05T10:00:00.000Z', 3],
    ]);
  });
});

describe('durationsBy', () => {
  it('groups durations by value in sorted order', () => {
    expect(durationsBy(TRACES, 'operation')).toEqual([
      { name: 'agent.dispatch', durations: [300] },
      { name: 'search_service.search', durations: [400, 100, 200] },
    ]);
    expect(durationsBy(TRACES, null)).toEqual([{ name: 'all traces', durations: [400, 100, 300, 200] }]);
  });
});

describe('heatmap', () => {
  it('averages latency per cell, leaving empty cells null and weekdays in calendar order', () => {
    expect(heatmap(TRACES, 'weekday', 'operation')).toEqual({
      columns: ['Sunday', 'Monday'],
      rows: ['agent.dispatch', 'search_service.search'],
      cells: [
        [null, 300],
        [200, 250],
      ],
    });
  });
});

describe('outliers', () => {
  it('keeps traces outside the bounds, slowest first', () => {
    expect(outliers(TRACES, { lower: 150, upper: 350 }).map((t) => t.duration_ms)).toEqual([400, 100]);
    expect(outliers(TRACES, null)).toEqual([]);
  });
});

describe('explore', () => {
  it('matches trace IDs and operations in any case and sorts', () => {
    expect(explore(TRACES, 'DISPATCH', 'Newest first').map((t) => t.duration_ms)).toEqual([300]);
    expect(explore(TRACES, 't-2026-10-04', 'Newest first').map((t) => t.duration_ms)).toEqual([200]);
    expect(explore(TRACES, '', 'Oldest first').map((t) => t.duration_ms)).toEqual([200, 300, 100, 400]);
    expect(explore(TRACES, '', 'Slowest first').map((t) => t.duration_ms)).toEqual([400, 300, 200, 100]);
    expect(explore(TRACES, ' ', 'Fastest first').map((t) => t.duration_ms)).toEqual([100, 200, 300, 400]);
  });
});
