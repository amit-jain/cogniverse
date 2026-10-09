import { describe, expect, it } from 'vitest';
import { queueSummary } from '../src/client/ops/AnnotationsView';
import {
  changedCorrections,
  confidenceBars,
  entityLabels,
  fieldKind,
  fieldText,
  generationMetadata,
  parseField,
  regenerable,
  retryCount,
  selfConsistencyLines,
  type ReviewedItem,
} from '../src/client/ops/ApprovalsView';
import { ingestOutcome, TERMINAL } from '../src/client/ops/IngestionView';
import { jsonText, parseJsonObject, sameJson } from '../src/client/ops/forms';
import { splitList, withSavedReviews } from '../src/client/ops/WorkflowReviewsView';
import { matchingPoints } from '../src/client/ops/EmbeddingAtlasView';
import { errorMessage } from '../src/client/ops/http';
import { delta, percent } from '../src/client/ops/metrics';
import { formatArgoTime } from '../src/client/ops/OptimizationView';
import { parseSse } from '../src/client/ops/sse';
import { parseRoute, routeHash } from '../src/client/route';

describe('errorMessage', () => {
  it('reads each failure shape the runtime and web server answer with', () => {
    expect(errorMessage({ error: 'runtime down' }, 502)).toBe('runtime down');
    expect(
      errorMessage(
        { error: { message: 'Search span 00000000000000ab is not a span of this tenant.', code: 'span_not_found' } },
        404,
      ),
    ).toBe('Search span 00000000000000ab is not a span of this tenant.');
    expect(errorMessage({ detail: 'Tenant acme:prod already exists' }, 409)).toBe(
      'Tenant acme:prod already exists',
    );
    expect(
      errorMessage(
        { detail: { error: 'profile_list_failed', message: 'Listing profiles failed.' } },
        500,
      ),
    ).toBe('Listing profiles failed.');
    expect(
      errorMessage(
        {
          detail: {
            message: 'Profile validation failed',
            errors: ['Profile name too long (101 chars, max 100)', "Strategy 'x' missing 'class' field."],
          },
        },
        400,
      ),
    ).toBe(
      "Profile validation failed: Profile name too long (101 chars, max 100); Strategy 'x' missing 'class' field.",
    );
    expect(
      errorMessage(
        {
          detail: [
            { loc: ['body', 'org_id'], msg: 'Field required', type: 'missing' },
            { loc: ['query', 'tenant_id'], msg: 'Field required', type: 'missing' },
          ],
        },
        422,
      ),
    ).toBe('org_id: Field required; query.tenant_id: Field required');
  });

  it('names the status when the body carries no reason', () => {
    expect(errorMessage(null, 500)).toBe('The runtime answered HTTP 500.');
    expect(errorMessage({ detail: [{ nope: 1 }] }, 422)).toBe('The runtime answered HTTP 422.');
  });
});

describe('routes', () => {
  it('round-trips every route through its hash', () => {
    for (const route of [
      { kind: 'agent' as const, name: 'search_agent' },
      { kind: 'ops' as const, id: 'tenants' },
      { kind: 'agent' as const, name: 'a b/c' },
      { kind: 'agent' as const, name: 'search_agent', thread: '4b5c6839-4307-44eb-a16f-cac030c76898' },
      { kind: 'agent' as const, name: 'a b/c', thread: 'x/y z' },
    ])
      expect(parseRoute(routeHash(route))).toEqual(route);
  });

  it("puts an agent's thread after its name", () => {
    expect(routeHash({ kind: 'agent', name: 'search_agent', thread: 't-1' })).toBe('#/agents/search_agent/t-1');
    expect(parseRoute('#/agents/search_agent/t-1')).toEqual({ kind: 'agent', name: 'search_agent', thread: 't-1' });
  });

  it('treats an unknown or empty hash as the default agent', () => {
    expect(parseRoute('')).toEqual({ kind: 'agent' });
    expect(parseRoute('#/ops')).toEqual({ kind: 'agent' });
    expect(parseRoute('#/elsewhere/x')).toEqual({ kind: 'agent' });
  });
});

describe('JSON form fields', () => {
  it('parses an object, treats blank as unset, and names the field it rejects', () => {
    expect(parseJsonObject('Strategies', '{"embedding": {"class": "X"}}')).toEqual({
      embedding: { class: 'X' },
    });
    expect(parseJsonObject('Strategies', '  \n ')).toBeUndefined();
    expect(() => parseJsonObject('Strategies', '{"a": ')).toThrow(
      new Error('Strategies is not valid JSON.'),
    );
    expect(() => parseJsonObject('Pipeline config', '[1, 2]')).toThrow(
      new Error('Pipeline config must be a JSON object.'),
    );
    expect(() => parseJsonObject('Pipeline config', 'null')).toThrow(
      new Error('Pipeline config must be a JSON object.'),
    );
  });

  it('round-trips a value through its textarea text and compares ignoring key order', () => {
    const value = { b: [1, { d: 2, c: 3 }], a: 'x' };
    expect(parseJsonObject('Value', jsonText(value))).toEqual(value);
    expect(jsonText(null)).toBe('');
    expect(sameJson({ a: 1, b: { c: 2, d: 3 } }, { b: { d: 3, c: 2 }, a: 1 })).toBe(true);
    expect(sameJson({ a: [1, 2] }, { a: [2, 1] })).toBe(false);
    expect(sameJson({ a: 1 }, { a: 1, b: null })).toBe(false);
  });
});

describe('changedCorrections', () => {
  it('keeps only the fields the reviewer changed, comparing nested values by content', () => {
    const template = {
      chosen_agent: 'video_search_agent',
      entities: [{ text: 'gradient descent', type: 'TOPIC' }],
      relationships: [],
    };
    expect(
      changedCorrections(template, {
        chosen_agent: 'summarizer_agent',
        entities: [{ type: 'TOPIC', text: 'gradient descent' }],
        relationships: [],
        topics: ['optimization'],
      }),
    ).toEqual({ chosen_agent: 'summarizer_agent', topics: ['optimization'] });
    expect(changedCorrections(template, template)).toEqual({});
  });
});

describe('queueSummary', () => {
  it('counts every status in a fixed order, zero for one the queue does not report', () => {
    expect(queueSummary({ expired: 2, pending: 7 })).toBe('7 pending, 0 assigned, 2 expired, 0 completed.');
  });
});

describe('splitList', () => {
  it('splits on commas and line breaks, trimming and dropping blanks', () => {
    expect(splitList(' search_agent, report_agent ,\n\nsummarizer_agent\n')).toEqual([
      'search_agent',
      'report_agent',
      'summarizer_agent',
    ]);
    expect(splitList(' , \n ')).toEqual([]);
  });
});

describe('parseSse', () => {
  it('returns complete frames with their ids and keeps the unfinished tail', () => {
    expect(
      parseSse('id: 1-0\ndata: {"state":"queued"}\n\n: keep-alive\n\nid: 2-0\ndata: {"state":"run'),
    ).toEqual({ frames: [{ id: '1-0', data: '{"state":"queued"}' }], rest: 'id: 2-0\ndata: {"state":"run' });
  });

  it('joins multi-line data, accepts CRLF and frames without ids', () => {
    expect(parseSse('data: a\r\ndata: b\r\n\r\ndata:c\n\n')).toEqual({
      frames: [{ data: 'a\nb' }, { data: 'c' }],
      rest: '',
    });
  });

  it('drops heartbeats and data-less frames', () => {
    expect(parseSse(': keep-alive\n\nid: 3-0\n\n')).toEqual({ frames: [], rest: '' });
  });
});

describe('formatArgoTime', () => {
  it('renders an Argo time in UTC to the minute and a missing one as a dash', () => {
    expect(formatArgoTime('2026-09-16T10:05:59Z')).toBe('2026-09-16 10:05 UTC');
    expect(formatArgoTime(null)).toBe('—');
    expect(formatArgoTime('not a time')).toBe('not a time');
  });
});

describe('metric formats', () => {
  it('shows a rate as a percentage and a delta with its sign', () => {
    expect(percent(0.4567)).toBe('45.7%');
    expect(percent(1)).toBe('100.0%');
    expect(delta(400)).toBe('+400.0');
    expect(delta(-0.5, 3)).toBe('-0.500');
    expect(delta(0)).toBe('0.0');
    expect(delta(null)).toBe('—');
  });
});

describe('matchingPoints', () => {
  const points = [
    { id: 'b', x: 0, y: 0, title: 'volcanoes.txt', text: 'Volcanoes build islands' },
    { id: 'a', x: 1, y: 1, title: 'Rivers.txt', text: 'Rivers carve canyons' },
    { id: 'c', x: 2, y: 2, title: null, text: 'Glaciers grind valleys' },
  ];

  it('orders every point by title, then id, when nothing is searched', () => {
    expect(matchingPoints(points, '  ').map((p) => p.id)).toEqual(['c', 'a', 'b']);
  });

  it('keeps the points whose title or text holds the query in any case', () => {
    expect(matchingPoints(points, 'CARVE').map((p) => p.id)).toEqual(['a']);
    expect(matchingPoints(points, 'volcanoes.txt').map((p) => p.id)).toEqual(['b']);
    expect(matchingPoints(points, 'lava')).toEqual([]);
  });
});

describe('ingestOutcome', () => {
  it('treats a cancelled ingest as finished, with its reason', () => {
    expect([...TERMINAL].sort()).toEqual(['cancelled', 'complete', 'failed']);
    expect(ingestOutcome({ state: 'cancelled', reason: 'tenant deleted' }, false)).toEqual({
      text: 'Cancelled: tenant deleted',
      failed: true,
    });
    expect(ingestOutcome({ state: 'cancelled' }, false)).toEqual({ text: 'Cancelled: no reason given', failed: true });
  });

  it('reports a completion that fed no documents as a failure', () => {
    expect(
      ingestOutcome({ state: 'complete', result: { video_id: 'v1', chunks: 2, documents_fed: 0 } }, false),
    ).toEqual({ text: 'v1: completed without feeding any documents.', failed: true });
    expect(ingestOutcome({ state: 'complete', result: { video_id: 'v1', chunks: 2 } }, false)).toEqual({
      text: 'v1: completed without reporting the documents it fed.',
      failed: true,
    });
    expect(
      ingestOutcome({ state: 'complete', result: { video_id: 'v1', documents_fed: '3' as unknown as number } }, false),
    ).toEqual({ text: 'v1: completed with an invalid documents_fed ("3").', failed: true });
  });

  it('reads a re-upload of ingested bytes without counts as already ingested', () => {
    expect(ingestOutcome({ state: 'complete', result: { video_id: 'v1' } }, true)).toEqual({
      text: 'v1: already ingested; nothing was fed again.',
      failed: false,
    });
  });

  it('keeps a fed completion, a failure and a retry as they were', () => {
    expect(
      ingestOutcome(
        { state: 'complete', result: { video_id: 'v1', chunks: 3, documents_fed: 3, graph_nodes: 4, graph_edges: 2 } },
        false,
      ),
    ).toEqual({ text: 'v1: 3 chunks, 3 documents fed. Graph: 4 nodes, 2 edges.', failed: false });
    expect(ingestOutcome({ state: 'failed', error_type: 'IngestPipelineError', error: 'no frames' }, false)).toEqual({
      text: 'IngestPipelineError: no frames',
      failed: true,
    });
    expect(ingestOutcome({ state: 'running' }, false)).toEqual({ text: '', failed: false });
  });
});

describe('selfConsistencyLines', () => {
  it('gives one line per sampled mention with its agreement and review flag', () => {
    expect(
      selfConsistencyLines({
        self_consistency: {
          samples: 5,
          entities: [
            { text: 'gradient descent', type: 'CONCEPT', agreement: 0.6, needs_review: true },
            { text: 'lecture', type: 'MEDIA', agreement: 1, needs_review: false },
          ],
        },
      }),
    ).toEqual([
      'Agreement (5 samples): gradient descent (CONCEPT) 0.60 — needs review',
      'Agreement (5 samples): lecture (MEDIA) 1.00',
    ]);
    expect(selfConsistencyLines({ agent_type: 'routing' })).toEqual([]);
  });
});

describe('review item details', () => {
  const data = {
    query: 'find the lecture',
    entities: [{ text: 'gradient descent', type: 'CONCEPT' }, { text: 'lecture', type: 'MEDIA' }],
    metadata: { _generation_metadata: { retry_count: 2, reasoning: 'two tries' } },
  };

  it('reads entities, retries and the generation record from the example', () => {
    expect(entityLabels(data)).toEqual(['gradient descent (CONCEPT)', 'lecture (MEDIA)']);
    expect(retryCount(data)).toBe(2);
    expect(generationMetadata(data)).toEqual({ retry_count: 2, reasoning: 'two tries' });
  });

  it('reads an example without them as none, zero retries and no record', () => {
    expect(entityLabels({ query: 'q' })).toEqual([]);
    expect(retryCount({ query: 'q', metadata: {} })).toBe(0);
    expect(generationMetadata({ query: 'q' })).toBe(undefined);
  });
});

describe('corrections editor fields', () => {
  it('edits each value as its kind and parses the text back to the same value', () => {
    const template = {
      chosen_agent: 'video_search_agent',
      task_count: 2,
      success: true,
      agent_sequence: ['a', 'b'],
      metadata: { k: 1 },
    };
    expect(Object.values(template).map(fieldKind)).toEqual(['text', 'number', 'boolean', 'json', 'json']);
    for (const [name, value] of Object.entries(template))
      expect(parseField(name, fieldKind(value), fieldText(value))).toEqual(value);
  });

  it('names the field whose text is not of its kind', () => {
    expect(() => parseField('task_count', 'number', 'three')).toThrow(new Error('task_count must be a number.'));
    expect(() => parseField('task_count', 'number', ' ')).toThrow(new Error('task_count must be a number.'));
    expect(() => parseField('agent_sequence', 'json', '[a')).toThrow(new Error('agent_sequence is not valid JSON.'));
  });
});

describe('review history', () => {
  const rejected = (schema: string | null, replacement: string | null): ReviewedItem => ({
    item_id: 'i1',
    batch_id: 'b1',
    status: 'rejected',
    confidence: 0.3,
    query: 'q',
    data: {},
    created_at: null,
    reviewed_at: null,
    schema_name: schema,
    reviewer: 'r',
    feedback: 'f',
    corrections: {},
    replacement_id: replacement,
    replacement_status: replacement ? 'regenerated' : null,
  });

  it('offers regeneration only for a schema item nothing replaced', () => {
    expect(regenerable(rejected('WorkflowExecutionSchema', null))).toBe(true);
    expect(regenerable(rejected('WorkflowExecutionSchema', 'i2'))).toBe(false);
    expect(regenerable(rejected(null, null))).toBe(false);
  });

  it('charts the mean confidence of each group that holds items, in group order', () => {
    expect(
      confidenceBars({
        total: 3,
        pending: 1,
        auto_approved: 0,
        approved: 1,
        rejected: 1,
        approval_rate: 1 / 3,
        average_confidence: { rejected: 0.3, approved: 0.4, pending: 0.7 },
      }),
    ).toEqual([
      { label: 'Awaiting review', value: 0.7 },
      { label: 'Approved', value: 0.4 },
      { label: 'Rejected', value: 0.3 },
    ]);
  });
});

describe('withSavedReviews', () => {
  it('shows a review the page saved over the list it loaded', () => {
    const loaded = [
      { span_id: 's1', review: null },
      { span_id: 's2', review: { label: 'good' } },
    ];
    expect(withSavedReviews(loaded, { s1: { span_id: 's1', review: { label: 'poor' } } })).toEqual([
      { span_id: 's1', review: { label: 'poor' } },
      { span_id: 's2', review: { label: 'good' } },
    ]);
    expect(withSavedReviews(loaded, {})).toEqual(loaded);
  });
});
