import { describe, expect, it } from 'vitest';
import { jsonText, parseJsonObject, sameJson } from '../src/client/ops/forms';
import { errorMessage } from '../src/client/ops/http';
import { parseRoute, routeHash } from '../src/client/route';

describe('errorMessage', () => {
  it('reads each failure shape the runtime and web server answer with', () => {
    expect(errorMessage({ error: 'runtime down' }, 502)).toBe('runtime down');
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
