import { describe, expect, it } from 'vitest';
import { RuntimeRequestError } from '../src/client/ops/http';
import {
  checkTrainingFile,
  describeOutcome,
  REPORT_START,
  reportFilename,
  reportProgress,
  reportSummary,
  runActionError,
  templateFile,
  type ExampleTemplates,
  type ReportEvent,
  type StepOutcome,
} from '../src/client/ops/optimization';

const EXAMPLE = {
  query: 'kiln firing',
  enhanced_query: 'stoneware kiln firing schedule',
  reasoning: 'names the ware',
};

const CATALOG: ExampleTemplates = {
  max_examples: 2,
  templates: {
    query_enhancement: {
      schema: 'QueryEnhancementExampleSchema',
      fields: ['query', 'enhanced_query', 'expansion_terms', 'synonyms', 'context', 'reasoning'],
      required: ['query', 'enhanced_query', 'reasoning'],
      example: EXAMPLE,
    },
    routing: {
      schema: 'RoutingExperienceSchema',
      fields: ['query', 'chosen_agent'],
      required: ['query', 'chosen_agent'],
      example: { query: 'play it', chosen_agent: 'video_search_agent' },
    },
  },
};

const file = (value: unknown) => JSON.stringify(value);

describe('templateFile', () => {
  it('saves the optimizer and its example as an uploadable file', () => {
    const saved = templateFile('query_enhancement', CATALOG.templates.query_enhancement);
    expect(saved.name).toBe('query_enhancement_examples.json');
    expect(JSON.parse(saved.text)).toEqual({ optimizer: 'query_enhancement', examples: [EXAMPLE] });
    expect(checkTrainingFile(saved.text, CATALOG)).toEqual({
      ok: true,
      optimizer: 'query_enhancement',
      examples: [EXAMPLE],
      preview: JSON.stringify({ optimizer: 'query_enhancement', examples: [EXAMPLE] }, null, 2),
      summary: 'Valid query_enhancement examples file (1 example).',
    });
  });
});

describe('checkTrainingFile', () => {
  it('names a file that is not JSON', () => {
    let reason = '';
    try {
      JSON.parse('not json');
    } catch (error) {
      reason = (error as Error).message;
    }
    expect(checkTrainingFile('not json', CATALOG)).toEqual({ ok: false, errors: [`Invalid JSON: ${reason}`] });
  });

  it('refuses a file without the optimizer and examples keys', () => {
    expect(checkTrainingFile(file([EXAMPLE]), CATALOG)).toEqual({
      ok: false,
      preview: JSON.stringify([EXAMPLE], null, 2),
      errors: ['Expected a JSON object with "optimizer" and "examples" keys.'],
    });
    expect(checkTrainingFile(file({ optimizer: 'workflow', examples: [], good_routes: [] }), CATALOG)).toEqual({
      ok: false,
      preview: JSON.stringify({ optimizer: 'workflow', examples: [], good_routes: [] }, null, 2),
      errors: [
        'Unexpected keys: good_routes.',
        '"optimizer" must be one of query_enhancement, routing.',
        '"examples" must be a non-empty list.',
      ],
    });
  });

  it('refuses more examples than one upload holds', () => {
    const check = checkTrainingFile(file({ optimizer: 'routing', examples: [{}, {}, {}] }), CATALOG);
    expect(check.ok ? [] : check.errors).toEqual(['A file holds at most 2 examples, not 3.']);
  });

  it('names each example missing a required field or carrying an unknown one', () => {
    const check = checkTrainingFile(
      file({ optimizer: 'query_enhancement', examples: ['text', { query: 'q', bogus: 1 }] }),
      CATALOG,
    );
    expect(check.ok ? [] : check.errors).toEqual([
      'examples[0] must be a JSON object.',
      'examples[1] is missing enhanced_query, reasoning.',
      'examples[1] has unknown fields bogus.',
    ]);
  });

  it('counts the examples of a valid file', () => {
    const check = checkTrainingFile(file({ optimizer: 'query_enhancement', examples: [EXAMPLE, EXAMPLE] }), CATALOG);
    expect(check.ok && [check.optimizer, check.summary, check.examples]).toEqual([
      'query_enhancement',
      'Valid query_enhancement examples file (2 examples).',
      [EXAMPLE, EXAMPLE],
    ]);
  });
});

describe('runActionError', () => {
  it('reads a missing run as one Argo deleted after its time-to-live', () => {
    expect(runActionError('wf-1', new RuntimeRequestError('Workflow not found', 404))).toBe(
      'Run wf-1 no longer exists; Argo deleted it when its time-to-live expired.',
    );
    expect(runActionError('wf-1', new RuntimeRequestError('The Argo API did not answer; retry.', 502))).toBe(
      'The Argo API did not answer; retry.',
    );
  });
});

describe('reportProgress', () => {
  const run = (events: ReportEvent[]) => events.reduce(reportProgress, REPORT_START);

  it('follows status, streamed text and the final report', () => {
    const report = { executive_summary: 'All good.', recommendations: ['Keep going'] };
    expect(
      run([
        { type: 'status', phase: 'search', message: 'Searching' },
        { type: 'partial', phase: 'token', data: { accumulated: 'All go' } },
        { type: 'partial', phase: 'thinking', data: { themes: ['a', 'b', 'c', 'd'] } },
      ]),
    ).toEqual({ status: 'Themes: a, b, c', text: 'All go' });
    expect(
      run([
        { type: 'status', phase: 'search', message: 'Searching' },
        { type: 'final', data: report },
      ]),
    ).toEqual({ status: '', text: '', report });
  });

  it('ends on the error the agent reports', () => {
    expect(run([{ type: 'error', message: 'DetailedReportAgent streaming failed with TimeoutError.' }])).toEqual({
      status: '',
      text: '',
      error: 'DetailedReportAgent streaming failed with TimeoutError.',
    });
  });
});

describe('reportSummary', () => {
  it('reads a streamed report and an answer envelope alike', () => {
    expect(reportSummary({ executive_summary: 'Streamed.', recommendations: ['a', 3, 'b'] })).toEqual({
      summary: 'Streamed.',
      recommendations: ['a', 'b'],
    });
    expect(reportSummary({ status: 'success', result: { executive_summary: 'Nothing to search.' } })).toEqual({
      summary: 'Nothing to search.',
      recommendations: [],
    });
  });
});

describe('reportFilename', () => {
  it('stamps the local date and time', () => {
    expect(reportFilename(new Date(2026, 9, 8, 7, 5, 3))).toBe('optimization_report_20261008_070503.json');
  });
});

describe('describeOutcome', () => {
  const outcome = (fields: Partial<StepOutcome>): StepOutcome => ({
    name: null,
    status: 'success',
    reason: null,
    decision: null,
    served_score: null,
    skip_above_score: null,
    baseline_score: null,
    current_score: null,
    candidate_score: null,
    rollback_version: null,
    error: null,
    ...fields,
  });
  const atCeiling = {
    status: 'skipped',
    reason: 'served_score_at_ceiling',
    served_score: 0.983,
    skip_above_score: 0.95,
  };

  it('says a compile was skipped because the served score is at the ceiling', () => {
    expect(describeOutcome(outcome(atCeiling))).toBe(
      'Compile skipped: the served score 0.983 is at or above the ceiling 0.95',
    );
    expect(describeOutcome(outcome({ ...atCeiling, served_score: 1, skip_above_score: 1 }))).toBe(
      'Compile skipped: the served score 1 is at or above the ceiling 1',
    );
    expect(describeOutcome(outcome({ ...atCeiling, served_score: 0.95004, name: 'summary' }))).toBe(
      'summary: Compile skipped: the served score 0.95 is at or above the ceiling 0.95',
    );
  });

  it('names the rollback a skipped run still made', () => {
    expect(describeOutcome(outcome({ ...atCeiling, served_score: 1, rollback_version: 7 }))).toBe(
      'Compile skipped: the served score 1 is at or above the ceiling 0.95; rolled back to the base module as v7',
    );
  });

  it('gives any other skip its reason', () => {
    expect(describeOutcome(outcome({ status: 'skipped', reason: 'no_positive_examples' }))).toBe(
      'Skipped: no_positive_examples',
    );
    expect(describeOutcome(outcome({ status: 'skipped' }))).toBe('Skipped: no reason recorded');
    // A ceiling skip without its scores is not dressed up as one.
    expect(describeOutcome(outcome({ status: 'skipped', reason: 'served_score_at_ceiling' }))).toBe(
      'Skipped: served_score_at_ceiling',
    );
  });

  it('gives a decision with the scores it was made on', () => {
    expect(
      describeOutcome(outcome({ decision: 'promote', baseline_score: 0.5, current_score: 0.983, candidate_score: 1 })),
    ).toBe('Promoted the candidate (baseline 0.5, current 0.983, candidate 1)');
    expect(describeOutcome(outcome({ decision: 'keep', baseline_score: 0.9, candidate_score: 0.9049 }))).toBe(
      'Kept the served artifact (baseline 0.9, candidate 0.905)',
    );
    expect(describeOutcome(outcome({ decision: 'rollback', name: 'profile' }))).toBe(
      'profile: Rolled back to the base module',
    );
    expect(describeOutcome(outcome({ decision: 'reject', baseline_score: 0.9288, candidate_score: 0.9303 }))).toBe(
      'Rejected the candidate (baseline 0.929, candidate 0.93)',
    );
    expect(describeOutcome(outcome({ decision: 'insufficient_population' }))).toBe('insufficient_population');
  });

  it('gives a failure its error and anything else its status', () => {
    expect(describeOutcome(outcome({ status: 'failed', error: 'teacher endpoint down', name: 'report' }))).toBe(
      'report: Failed: teacher endpoint down',
    );
    expect(describeOutcome(outcome({ status: 'failed', reason: 'baseline_not_reconstructable' }))).toBe(
      'Failed: baseline_not_reconstructable',
    );
    expect(describeOutcome(outcome({ status: 'error' }))).toBe('Failed: no error recorded');
    expect(describeOutcome(outcome({ status: 'no_eval_material' }))).toBe('no_eval_material');
  });
});
