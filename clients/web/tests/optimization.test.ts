import { describe, expect, it } from 'vitest';
import { RuntimeRequestError } from '../src/client/ops/http';
import {
  checkTrainingFile,
  REPORT_START,
  reportFilename,
  reportProgress,
  reportSummary,
  runActionError,
  templateFile,
  type ExampleTemplates,
  type ReportEvent,
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
