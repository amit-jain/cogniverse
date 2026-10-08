import { describe, expect, it } from 'vitest';
import {
  calibrationPoints,
  decisionsPerHourByAgent,
  initialReviewLabel,
  labelState,
  labelledBy,
  lookbackHours,
  scoreColor,
  shownCandidates,
  successRatePerHour,
  withReviews,
  type AnnotationCandidate,
  type DecisionLabel,
  type RoutingDecision,
} from '../src/client/ops/routing';

function decision(
  confidence: number,
  outcome: string,
  start_time = '2026-10-05T10:15:00+00:00',
  chosen_agent = 'search_agent',
): RoutingDecision {
  return {
    span_id: 's',
    trace_id: null,
    start_time,
    query: 'q',
    chosen_agent,
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

const rounded = (points: { confidence: number; success_rate: number; decisions: number }[]) =>
  points.map((point) => ({ ...point, confidence: Math.round(point.confidence * 1000) / 1000 }));

describe('calibrationPoints', () => {
  // Expectations are what pandas.cut(bins=10) and a groupby give for the
  // same confidences and outcomes.
  it('bins between the lowest and highest confidence and keeps the non-empty bins', () => {
    expect(
      rounded(
        calibrationPoints([
          decision(0.05, 'failure'),
          decision(0.19, 'success'),
          decision(0.2, 'success'),
          decision(0.9, 'ambiguous'),
          decision(1, 'success'),
        ]),
      ),
    ).toEqual([
      { confidence: 0.05, success_rate: 0, decisions: 1 },
      { confidence: 0.195, success_rate: 1, decisions: 2 },
      { confidence: 0.9, success_rate: 0, decisions: 1 },
      { confidence: 1, success_rate: 1, decisions: 1 },
    ]);
  });

  it('closes each bin on the right', () => {
    expect(
      rounded(
        calibrationPoints([
          decision(0, 'failure'),
          decision(0.45, 'success'),
          decision(0.5, 'failure'),
          decision(1, 'success'),
        ]),
      ),
    ).toEqual([
      { confidence: 0, success_rate: 0, decisions: 1 },
      { confidence: 0.475, success_rate: 0.5, decisions: 2 },
      { confidence: 1, success_rate: 1, decisions: 1 },
    ]);
  });

  it('puts decisions of one confidence in one bin, and none in none', () => {
    expect(calibrationPoints([decision(0.7, 'success'), decision(0.7, 'failure')])).toEqual([
      { confidence: 0.7, success_rate: 0.5, decisions: 2 },
    ]);
    expect(calibrationPoints([])).toEqual([]);
  });
});

describe('decisionsPerHourByAgent', () => {
  it("counts each agent's decisions per UTC hour, agents by name and hours oldest first", () => {
    expect(
      decisionsPerHourByAgent([
        decision(0.9, 'success', '2026-10-05T11:59:59+00:00', 'summarizer_agent'),
        decision(0.9, 'failure', '2026-10-05T12:30:00+02:00'),
        decision(0.9, 'success', '2026-10-05T09:00:00+00:00'),
        decision(0.9, 'ambiguous', '2026-10-05T10:59:00+00:00'),
      ]),
    ).toEqual([
      { agent: 'search_agent', hours: ['2026-10-05T09:00:00Z', '2026-10-05T10:00:00Z'], decisions: [1, 2] },
      { agent: 'summarizer_agent', hours: ['2026-10-05T11:00:00Z'], decisions: [1] },
    ]);
  });
});

describe('successRatePerHour', () => {
  it('rates every hour from the first to the last, with no rate for an hour without decisions', () => {
    expect(
      successRatePerHour([
        decision(0.9, 'success', '2026-10-05T12:10:00+00:00'),
        decision(0.9, 'failure', '2026-10-05T09:59:00+00:00'),
        decision(0.9, 'success', '2026-10-05T09:01:00+00:00'),
        decision(0.9, 'ambiguous', '2026-10-05T09:30:00+00:00'),
        decision(0.9, 'success', '2026-10-05T10:00:00+00:00'),
      ]),
    ).toEqual([
      { hour: '2026-10-05T09:00:00Z', success_rate: 1 / 3 },
      { hour: '2026-10-05T10:00:00Z', success_rate: 1 },
      { hour: '2026-10-05T11:00:00Z', success_rate: null },
      { hour: '2026-10-05T12:00:00Z', success_rate: 1 },
    ]);
    expect(successRatePerHour([])).toEqual([]);
  });
});

describe('lookbackHours', () => {
  it('accepts whole hours from 1 to 720 only', () => {
    expect(['1', ' 36 ', '720', '0', '721', '2.5', '', 'week'].map(lookbackHours)).toEqual([
      1,
      36,
      720,
      null,
      null,
      null,
      null,
      null,
    ]);
  });
});

describe('initialReviewLabel', () => {
  it("starts on the label's reviewer equivalent, or the first choice", () => {
    const choices = ['correct', 'wrong', 'ambiguous', 'insufficient_info'];
    expect(
      [
        null,
        { ...LLM_LABEL, label: 'wrong_routing' },
        { ...LLM_LABEL, label: 'correct_routing' },
        { ...LLM_LABEL, label: 'insufficient_info' },
        { ...LLM_LABEL, label: 'retired' },
      ].map((label) => initialReviewLabel(label, choices)),
    ).toEqual(['correct', 'wrong', 'correct', 'insufficient_info', 'correct']);
    expect(initialReviewLabel(LLM_LABEL, [])).toBe('');
  });
});

describe('scoreColor', () => {
  it('runs red through yellow to green', () => {
    expect([0, 0.25, 0.5, 1, -1, 2].map(scoreColor)).toEqual([
      'rgb(215, 48, 39)',
      'rgb(235, 152, 115)',
      'rgb(255, 255, 191)',
      'rgb(26, 152, 80)',
      'rgb(215, 48, 39)',
      'rgb(26, 152, 80)',
    ]);
  });
});

describe('shownCandidates', () => {
  const candidate = (span_id: string, priority: 'high' | 'medium' | 'low'): AnnotationCandidate => ({
    span_id,
    start_time: '2026-10-05T10:15:00+00:00',
    query: 'q',
    chosen_agent: 'search_agent',
    confidence: 0.4,
    outcome: 'success',
    priority,
    reason: 'r',
  });
  const candidates = [candidate('a', 'high'), candidate('b', 'medium'), candidate('c', 'low'), candidate('d', 'high')];
  const labels = { a: LLM_LABEL, b: { ...LLM_LABEL, annotator: 'sam' }, c: null };

  it('keeps the chosen priorities and drops LLM-labelled ones when asked', () => {
    expect(shownCandidates(candidates, ['high', 'medium', 'low'], true, labels).map((c) => c.span_id)).toEqual([
      'a',
      'b',
      'c',
      'd',
    ]);
    expect(shownCandidates(candidates, ['high', 'low'], true, labels).map((c) => c.span_id)).toEqual([
      'a',
      'c',
      'd',
    ]);
    expect(shownCandidates(candidates, ['high', 'medium'], false, labels).map((c) => c.span_id)).toEqual([
      'b',
      'd',
    ]);
    expect(shownCandidates(candidates, [], true, labels)).toEqual([]);
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
