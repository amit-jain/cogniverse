import { describe, expect, it } from 'vitest';
import { markRetrieved, pairName, successMatrix, type QueryScores, type StrategyScores } from '../src/client/ops/evaluation';

function scores(profile: string, strategy: string, success_rate: number): StrategyScores {
  return { profile, strategy, queries: 1, mrr: 0, ndcg: 0, recall_at_1: 0, recall_at_5: 0, precision_at_5: 0, success_rate };
}

describe('successMatrix', () => {
  it('places each success rate by profile and strategy, both sorted', () => {
    expect(
      successMatrix([scores('video', 'hybrid', 0.5), scores('audio', 'bm25', 1), scores('video', 'bm25', 0)]),
    ).toEqual({
      profiles: ['audio', 'video'],
      strategies: ['bm25', 'hybrid'],
      cells: [
        [1, null],
        [0, 0.5],
      ],
    });
  });

  it('is empty without scores', () => {
    expect(successMatrix([])).toEqual({ profiles: [], strategies: [], cells: [] });
  });
});

describe('markRetrieved', () => {
  const query: QueryScores = {
    profile: 'video',
    strategy: 'hybrid',
    query: 'a red car',
    expected: ['red_car', 'garage'],
    retrieved: ['beach', 'red_car', 'x', 'y', 'garage', 'z'],
    searched_at: '2026-10-05T10:00:00+00:00',
    trace_id: null,
    mrr: 0.5,
    ndcg: 0,
    recall_at_1: 0,
    recall_at_5: 1,
    precision_at_5: 0.4,
  };

  it('marks the first five sources expected or not', () => {
    expect(markRetrieved(query)).toEqual([
      { source: 'beach', expected: false },
      { source: 'red_car', expected: true },
      { source: 'x', expected: false },
      { source: 'y', expected: false },
      { source: 'garage', expected: true },
    ]);
    expect(markRetrieved(query, 2).map((item) => item.source)).toEqual(['beach', 'red_car']);
  });

  it('names a profile and strategy pair', () => {
    expect(pairName(query)).toBe('video / hybrid');
  });
});
