import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { chooseTenant, decideTenant, probeTenant, resetTenant, tenantState } from '../src/client/tenant';

/** A fetch answering each tenant probe from ``answers`` (by path), after
 * ``delays`` ms when given. */
function fakeFetch(
  answers: Record<string, () => Response>,
  delays: Record<string, number> = {},
  seen: string[] = [],
): typeof fetch {
  return (async (input: RequestInfo | URL) => {
    const path = String(input);
    seen.push(path);
    await new Promise((resolve) => setTimeout(resolve, delays[path] ?? 0));
    const answer = answers[path];
    if (!answer) throw new TypeError('fetch failed');
    return answer();
  }) as typeof fetch;
}

const json = (status: number, body: unknown) => () =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });

const PROBE = (tenant: string) => `/ui-api/runtime/admin/tenants/${encodeURIComponent(tenant)}`;

describe('decideTenant', () => {
  it('takes a registered tenant in its canonical form', () => {
    expect(decideTenant({ kind: 'registered', tenant: 'acme:acme' }, 'acme', false)).toEqual({ tenant: 'acme:acme' });
  });

  it('refuses a tenant the runtime does not know, naming how to register it', () => {
    expect(decideTenant({ kind: 'unknown' }, 'acme:typo', true)).toEqual({
      notice: {
        tone: 'error',
        text:
          'Tenant acme:typo is not registered. Register it in the Tenants view (or with POST /admin/tenants) ' +
          'first, or choose a registered tenant.',
      },
    });
  });

  it('refuses a tenant id the runtime rejects', () => {
    expect(decideTenant({ kind: 'invalid', detail: 'HTTP 422' }, 'a-b', false)).toEqual({
      notice: { tone: 'error', text: 'Tenant a-b cannot be used: HTTP 422' },
    });
  });

  it('keeps a confirmed tenant through a probe that could not reach the runtime', () => {
    expect(decideTenant({ kind: 'unreachable', detail: 'HTTP 503' }, 'acme:prod', true)).toEqual({
      tenant: 'acme:prod',
      notice: {
        tone: 'warning',
        text: 'Could not re-check tenant acme:prod (HTTP 503). Continuing with the last successful check.',
      },
    });
    expect(decideTenant({ kind: 'unreachable', detail: 'HTTP 503' }, 'acme:prod', false)).toEqual({
      tenant: 'acme:prod',
      notice: {
        tone: 'warning',
        text:
          'The runtime could not confirm tenant acme:prod is registered (HTTP 503); its views read as ' +
          'empty if it is not.',
      },
    });
  });
});

describe('probeTenant', () => {
  it('tells a registered, an unknown, an invalid and an unconfirmed tenant apart', async () => {
    const fetchFn = fakeFetch({
      [PROBE('acme')]: json(200, { tenant_full_id: 'acme:acme', org_id: 'acme' }),
      [PROBE('acme:typo')]: json(404, { detail: 'Tenant acme:typo not found' }),
      [PROBE('a-b')]: json(422, { detail: 'Invalid tenant_id' }),
      [PROBE('acme:prod')]: json(503, { detail: { message: 'The tenant registry did not answer; retry.' } }),
      [PROBE('beta:dev')]: json(502, { error: 'The Cogniverse runtime at http://rt did not answer (TypeError).' }),
    });
    expect(
      await Promise.all(['acme', 'acme:typo', 'a-b', 'acme:prod', 'beta:dev', 'gone:x'].map((t) => probeTenant(t, fetchFn))),
    ).toEqual([
      { kind: 'registered', tenant: 'acme:acme' },
      { kind: 'unknown' },
      { kind: 'invalid', detail: 'Invalid tenant_id' },
      { kind: 'unreachable', detail: 'The tenant registry did not answer; retry.' },
      { kind: 'unreachable', detail: 'The Cogniverse runtime at http://rt did not answer (TypeError).' },
      { kind: 'unreachable', detail: 'the web server did not answer' },
    ]);
  });
});

describe('chooseTenant', () => {
  beforeEach(() => resetTenant());
  afterEach(() => vi.useRealTimers());

  it('switches to a registered tenant and refuses an unknown one, keeping the last', async () => {
    const fetchFn = fakeFetch({
      [PROBE('acme')]: json(200, { tenant_full_id: 'acme:acme' }),
      [PROBE('acme:typo')]: json(404, { detail: 'Tenant acme:typo not found' }),
    });
    expect(await chooseTenant('acme', fetchFn)).toBe('acme:acme');
    expect(tenantState()).toEqual({ tenant: 'acme:acme', notice: undefined, checking: false });
    expect(await chooseTenant(' acme:typo ', fetchFn)).toBe('acme:acme');
    expect(tenantState()).toEqual({
      tenant: 'acme:acme',
      notice: decideTenant({ kind: 'unknown' }, 'acme:typo', false).notice,
      checking: false,
    });
  });

  it('keeps a confirmed tenant with a warning when the runtime stops answering', async () => {
    let up = true;
    const fetchFn = fakeFetch({
      [PROBE('acme:prod')]: () =>
        up ? json(200, { tenant_full_id: 'acme:prod' })() : json(503, { detail: { message: 'retry' } })(),
      [PROBE('beta:dev')]: json(200, { tenant_full_id: 'beta:dev' }),
    });
    vi.useFakeTimers({ toFake: ['Date'] });
    await chooseTenant('acme:prod', fetchFn);
    await chooseTenant('beta:dev', fetchFn);
    up = false;
    // Past the 30 s a confirmation is taken as current.
    vi.setSystemTime(Date.now() + 31_000);
    await chooseTenant('acme:prod', fetchFn);
    expect(tenantState()).toEqual({
      tenant: 'acme:prod',
      notice: decideTenant({ kind: 'unreachable', detail: 'retry' }, 'acme:prod', true).notice,
      checking: false,
    });
  });

  it('drops an active tenant the runtime no longer knows', async () => {
    resetTenant('acme:gone');
    const fetchFn = fakeFetch({ [PROBE('acme:gone')]: json(404, { detail: 'Tenant acme:gone not found' }) });
    expect(await chooseTenant('acme:gone', fetchFn)).toBe('');
    expect(tenantState().notice).toEqual(decideTenant({ kind: 'unknown' }, 'acme:gone', false).notice);
  });

  it('answers a recently confirmed tenant without asking the runtime again', async () => {
    const seen: string[] = [];
    const fetchFn = fakeFetch({ [PROBE('acme:prod')]: json(200, { tenant_full_id: 'acme:prod' }) }, {}, seen);
    await chooseTenant('acme:prod', fetchFn);
    await chooseTenant('acme:prod', fetchFn);
    expect(seen).toEqual([PROBE('acme:prod')]);
  });

  it('settles on the tenant chosen last when an earlier probe answers after it', async () => {
    const fetchFn = fakeFetch(
      {
        [PROBE('slow:one')]: json(200, { tenant_full_id: 'slow:one' }),
        [PROBE('fast:two')]: json(200, { tenant_full_id: 'fast:two' }),
      },
      { [PROBE('slow:one')]: 60 },
    );
    const [first, second] = await Promise.all([chooseTenant('slow:one', fetchFn), chooseTenant('fast:two', fetchFn)]);
    expect([first, second]).toEqual(['fast:two', 'fast:two']);
    expect(tenantState()).toEqual({ tenant: 'fast:two', notice: undefined, checking: false });
  });
});
