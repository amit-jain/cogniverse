import { describe, expect, it } from 'vitest';
import {
  calibrationBins,
  decisionsByHour,
  labelState,
  labelledBy,
  withReviews,
  type DecisionLabel,
  type RoutingDecision,
} from '../src/client/ops/routing';

function decision(confidence: number, outcome: string, start_time = '2026-10-05T10:15:00+00:00'): RoutingDecision {
  return {
    span_id: 's',
    trace_id: null,
    start_time,
    query: 'q',
    chosen_agent: 'search_agent',
    confidence,
    outcome,
    reason: 'r',
    latency_ms: 1,
    entity_extraction_failed: false,
    label: null,
  };
}

const LLM_LABEL: DecisionLabel = {
  label: 'wrong_routing',
  confidence: 0.7,
  reasoning: 'r',
  suggested_agent: null,
  annotator: 'llm',
  human_reviewed: false,
  requires_review: true,
  approved_by: null,
};

describe('calibrationBins', () => {
  it('places each decision by confidence and rates the successes of each range', () => {
    expect(
      calibrationBins([
        decision(0.05, 'failure'),
        decision(0.19, 'success'),
        decision(0.2, 'success'),
        decision(0.9, 'ambiguous'),
        decision(1, 'success'),
      ]),
    ).toEqual([
      { range: '0.0–0.2', decisions: 2, success_rate: 0.5 },
      { range: '0.2–0.4', decisions: 1, success_rate: 1 },
      { range: '0.4–0.6', decisions: 0, success_rate: null },
      { range: '0.6–0.8', decisions: 0, success_rate: null },
      { range: '0.8–1.0', decisions: 2, success_rate: 0.5 },
    ]);
  });
});

describe('decisionsByHour', () => {
  it('counts decisions and successes per UTC hour, oldest first', () => {
    expect(
      decisionsByHour([
        decision(0.9, 'success', '2026-10-05T11:59:59+00:00'),
        decision(0.9, 'failure', '2026-10-05T12:30:00+02:00'),
        decision(0.9, 'success', '2026-10-05T09:00:00+00:00'),
        decision(0.9, 'ambiguous', '2026-10-05T11:00:00+00:00'),
      ]),
    ).toEqual([
      { hour: '2026-10-05T09:00:00Z', decisions: 1, successes: 1 },
      { hour: '2026-10-05T10:00:00Z', decisions: 1, successes: 0 },
      { hour: '2026-10-05T11:00:00Z', decisions: 2, successes: 1 },
    ]);
  });
});

describe('labelState and labelledBy', () => {
  it('tells unlabelled, LLM-labelled and reviewed decisions apart', () => {
    const approved = {
      ...LLM_LABEL,
      human_reviewed: true,
      requires_review: false,
      approved_by: 'dana',
    };
    const reviewer = {
      ...LLM_LABEL,
      label: 'correct',
      annotator: 'sam',
      human_reviewed: true,
    };
    expect(
      [null, LLM_LABEL, approved, reviewer].map((label) => {
        const row = { ...decision(0.5, 'success'), label };
        return [labelState(row), label && labelledBy(label)];
      }),
    ).toEqual([
      ['unlabelled', null],
      ['llm', 'LLM'],
      ['reviewed', 'LLM, approved by dana'],
      ['reviewed', 'sam'],
    ]);
  });
});

describe('withReviews', () => {
  it('replaces each reviewed decision by span ID and keeps the order', () => {
    const first = { ...decision(0.9, 'success'), span_id: 'a' };
    const second = { ...decision(0.3, 'failure'), span_id: 'b' };
    const unidentified = { ...decision(0.5, 'success'), span_id: null };
    const reviewed = { ...second, label: { ...LLM_LABEL, human_reviewed: true, approved_by: 'dana' } };
    expect(withReviews([first, second, unidentified], { b: reviewed, c: first })).toEqual([
      first,
      reviewed,
      unidentified,
    ]);
  });
});
