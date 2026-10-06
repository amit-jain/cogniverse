import { describe, expect, it } from 'vitest';
import { queueSummary } from '../src/client/ops/AnnotationsView';
import { changedCorrections } from '../src/client/ops/ApprovalsView';
import { jsonText, parseJsonObject, sameJson } from '../src/client/ops/forms';
import { splitList } from '../src/client/ops/WorkflowReviewsView';
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
    ])
      expect(parseRoute(routeHash(route))).toEqual(route);
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
