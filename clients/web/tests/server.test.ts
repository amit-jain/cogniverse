import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { describe, expect, it } from 'vitest';
import { createApp } from '../src/server/app';
import { ConfigError, loadConfig, type ServerConfig } from '../src/server/config';
import { RuntimeUnavailableError, agentsStatus, cogniverseAgents, listAgents } from '../src/server/runtime';
import { TENANT_HEADER, TenantAuthUnavailableError, TenantKeys, UnknownTenantError } from '../src/server/tenants';
import { deadUrl, json, runtimeServer, withAdmin } from './fakeRuntime';

function config(runtimeUrl: string): ServerConfig {
  return { runtimeUrl, port: 4000, host: '127.0.0.1', clientDir: '/nonexistent' };
}

describe('loadConfig', () => {
  it('reads and normalises the environment', () => {
    expect(
      loadConfig(
        { COGNIVERSE_RUNTIME_URL: ' http://rt:8000/api// ', PORT: '5173' },
        '/srv/client',
      ),
    ).toEqual({
      runtimeUrl: 'http://rt:8000/api',
      port: 5173,
      host: '127.0.0.1',
      clientDir: '/srv/client',
    });
  });

  it('names the missing runtime URL', () => {
    expect(() => loadConfig({ COGNIVERSE_RUNTIME_URL: '  ' }, '/c')).toThrow(
      new ConfigError('Missing required environment: COGNIVERSE_RUNTIME_URL.'),
    );
  });

  it('rejects a port that is not a port', () => {
    const env = { COGNIVERSE_RUNTIME_URL: 'http://rt' };
    for (const port of ['0', '70000', '80.5', 'web'])
      expect(() => loadConfig({ ...env, PORT: port }, '/c')).toThrow(
        new ConfigError(`PORT must be an integer port number, got ${port}.`),
      );
  });
});

describe('listAgents', () => {
  it('returns the runtime registry names in order', async () => {
    const seen: string[] = [];
    const url = await runtimeServer((req, res) => {
      seen.push(`${req.method} ${req.url}`);
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify({ agents: ['search_agent', 'coding_agent'], count: 2 }));
    });
    expect(await listAgents(config(url))).toEqual(['search_agent', 'coding_agent']);
    expect(seen).toEqual(['GET /agents/']);
  });

  it('raises with the status when the runtime answers an error', async () => {
    const url = await runtimeServer((_req, res) => {
      res.statusCode = 503;
      res.end('{}');
    });
    await expect(listAgents(config(url))).rejects.toThrow(
      new RuntimeUnavailableError('The Cogniverse runtime answered the agent list with HTTP 503.'),
    );
  });

  it('raises when the body has no agents array', async () => {
    const url = await runtimeServer((_req, res) => res.end(JSON.stringify({ agents: [1, 2] })));
    await expect(listAgents(config(url))).rejects.toThrow(
      new RuntimeUnavailableError(
        'The Cogniverse runtime answered the agent list without an agents array.',
      ),
    );
  });

  it('raises naming the runtime when it is down', async () => {
    const url = await deadUrl();
    await expect(listAgents(config(url))).rejects.toThrow(
      new RuntimeUnavailableError(`The Cogniverse runtime at ${url} did not answer (TypeError).`),
    );
  });
});

describe('cogniverseAgents', () => {
  it("points each agent at its /ag-ui route and runs it with the tenant's key", async () => {
    const seen: string[] = [];
    const [route, admin] = withAdmin(['acme:prod'], (req, res) => {
      seen.push(`${req.method} ${req.url} ${req.headers.authorization}`);
      json(res, 200, {});
    });
    const url = await runtimeServer(route);
    const keys = new TenantKeys(config(url), fetch, 'web-test');
    const agents = cogniverseAgents(config(url), ['search_agent', 'a/b'], keys, 'acme:prod');
    expect(Object.keys(agents)).toEqual(['search_agent', 'a/b']);
    expect(agents.search_agent.url).toBe(`${url}/ag-ui/search_agent`);
    expect(agents['a/b'].url).toBe(`${url}/ag-ui/a%2Fb`);
    expect(agents.search_agent.headers).toEqual({ [TENANT_HEADER]: 'acme:prod' });
    await agents.search_agent.fetch(agents.search_agent.url, { method: 'POST' });
    expect(seen).toEqual([`POST /ag-ui/search_agent ${admin.bearer('acme:prod')}`]);
    expect(admin.calls).toEqual(['probe acme:prod', 'mint acme:prod web-test']);
  });

  it('refuses a run that names no tenant without calling the runtime', async () => {
    const [route, admin] = withAdmin([]);
    const url = await runtimeServer(route);
    const agents = cogniverseAgents(config(url), ['search_agent'], new TenantKeys(config(url)), undefined);
    const response = await agents.search_agent.fetch(agents.search_agent.url, { method: 'POST' });
    expect(response.status).toBe(400);
    expect(await response.json()).toEqual({ error: 'Choose a tenant before talking to an agent.' });
    expect(agents.search_agent.headers).toEqual({});
    expect(admin.calls).toEqual([]);
  });
});

describe('TenantKeys', () => {
  it('mints one key per tenant however many first requests race for it', async () => {
    const [route, admin] = withAdmin(['acme:prod', 'beta:dev']);
    const url = await runtimeServer(route);
    const keys = new TenantKeys(config(url), fetch, 'web-test');
    const tenants = Array.from({ length: 12 }, (_, i) => (i % 2 ? 'acme:prod' : 'beta:dev'));
    const issued = await Promise.all(tenants.map((tenant) => keys.keyFor(tenant)));
    expect(new Set(issued.filter((_, i) => tenants[i] === 'acme:prod'))).toEqual(
      new Set([admin.bearer('acme:prod').slice(7)]),
    );
    expect(new Set(issued.filter((_, i) => tenants[i] === 'beta:dev'))).toEqual(
      new Set([admin.bearer('beta:dev').slice(7)]),
    );
    expect(admin.tenantOf(`Bearer ${await keys.keyFor('acme:prod')}`)).toBe('acme:prod');
    expect([...admin.calls].sort()).toEqual([
      'mint acme:prod web-test',
      'mint beta:dev web-test',
      'probe acme:prod',
      'probe beta:dev',
    ]);
  });

  it('refuses an unregistered tenant and mints nothing for it', async () => {
    const [route, admin] = withAdmin(['acme:prod']);
    const url = await runtimeServer(route);
    const keys = new TenantKeys(config(url));
    await expect(keys.keyFor('acme:typo')).rejects.toThrow(
      new UnknownTenantError('Tenant acme:typo is not registered. Register it with POST /admin/tenants first.'),
    );
    expect(admin.calls).toEqual(['probe acme:typo']);
  });

  it('mints a key while the tenant registry cannot be read, and refuses what it rejects', async () => {
    const [route, admin] = withAdmin(['acme:prod']);
    const url = await runtimeServer((req, res) => {
      if (req.url === '/admin/tenants/acme%3Aprod')
        return json(res, 503, { detail: 'Tenant registry temporarily unavailable' });
      if (req.url === '/admin/tenants/bad%20id') return json(res, 400, { detail: 'Invalid tenant_id' });
      if (req.url === '/admin/tenants/gone%3Aroute') return json(res, 404, { detail: 'Not Found' });
      route(req, res);
    });
    const keys = new TenantKeys(config(url), fetch, 'web-test');
    expect(admin.tenantOf(`Bearer ${await keys.keyFor('acme:prod')}`)).toBe('acme:prod');
    await expect(keys.keyFor('bad id')).rejects.toThrow(
      new TenantAuthUnavailableError('The runtime could not confirm tenant bad id (HTTP 400: Invalid tenant_id).'),
    );
    await expect(keys.keyFor('gone:route')).rejects.toThrow(
      new TenantAuthUnavailableError('The runtime could not confirm tenant gone:route (HTTP 404: Not Found).'),
    );
    expect(admin.calls).toEqual(['mint acme:prod web-test']);
  });

  it("names the key store's failure and the runtime that did not answer", async () => {
    const [route] = withAdmin(['acme:prod']);
    const url = await runtimeServer((req, res) => {
      if (req.url === '/admin/harness/keys')
        return json(res, 503, {
          detail: { error: 'harness_key_store_unavailable', message: 'The harness key store did not answer; retry.' },
        });
      route(req, res);
    });
    await expect(new TenantKeys(config(url)).keyFor('acme:prod')).rejects.toThrow(
      new TenantAuthUnavailableError(
        'The runtime did not issue a harness key for tenant acme:prod (HTTP 503: The harness key store did not answer; retry.).',
      ),
    );
    const dead = await deadUrl();
    await expect(new TenantKeys(config(dead)).keyFor('acme:prod')).rejects.toThrow(
      new TenantAuthUnavailableError(`The Cogniverse runtime at ${dead} did not answer (TypeError).`),
    );
  });

  it('retries a request the runtime rejects with a newly minted key, once', async () => {
    const rejected = new Set<string>();
    const [route, admin] = withAdmin(['acme:prod'], (req, res) => {
      if (rejected.has(req.headers.authorization!)) return json(res, 401, { error: { message: 'Invalid API key' } });
      json(res, 200, { auth: admin.tenantOf(req.headers.authorization) });
    });
    const url = await runtimeServer(route);
    const keys = new TenantKeys(config(url), fetch, 'web-test');
    const agent = cogniverseAgents(config(url), ['search_agent'], keys, 'acme:prod').search_agent;
    expect(await (await agent.fetch(agent.url, { method: 'POST', body: '{}' })).json()).toEqual({ auth: 'acme:prod' });
    rejected.add(admin.bearer('acme:prod'));
    const retried = await agent.fetch(agent.url, { method: 'POST', body: '{}' });
    expect([retried.status, await retried.json()]).toEqual([200, { auth: 'acme:prod' }]);
    expect(rejected.has(admin.bearer('acme:prod'))).toBe(false);
    // A key the runtime rejects again is not retried a second time.
    rejected.add(admin.bearer('acme:prod'));
    rejected.add(`Bearer key-acme:prod-3`);
    const refused = await agent.fetch(agent.url, { method: 'POST', body: '{}' });
    expect(refused.status).toBe(401);
    expect(admin.calls.filter((call) => call.startsWith('mint'))).toEqual([
      'mint acme:prod web-test',
      'mint acme:prod web-test',
      'mint acme:prod web-test',
    ]);
  });

  it('revokes every key it minted', async () => {
    const [route, admin] = withAdmin(['acme:prod', 'beta:dev']);
    const url = await runtimeServer(route);
    const keys = new TenantKeys(config(url), fetch, 'web-test');
    await keys.keyFor('acme:prod');
    await keys.keyFor('beta:dev');
    await keys.revokeAll();
    expect(admin.calls.filter((call) => call.startsWith('revoke')).sort()).toEqual(['revoke hash-1', 'revoke hash-2']);
    await keys.keyFor('acme:prod');
    expect(admin.calls.at(-1)).toBe('mint acme:prod web-test');
  });
});

describe('agentsStatus', () => {
  it("reports each agent online when the runtime serves and the registry has it", async () => {
    const url = await runtimeServer((req, res) => {
      if (req.url === '/health') return json(res, 200, { status: 'degraded' });
      if (req.url === '/agents/search_agent')
        return json(res, 200, { name: 'search_agent', health_status: 'healthy' });
      if (req.url === '/agents/gone_agent') return json(res, 404, { detail: "Agent 'gone_agent' not found" });
      json(res, 500, {});
    });
    expect(await agentsStatus(config(url), ['search_agent', 'gone_agent', 'broken_agent'])).toEqual({
      runtime: 'degraded',
      agents: [
        { name: 'search_agent', status: 'online', health: 'healthy' },
        { name: 'gone_agent', status: 'offline', message: 'Not registered.' },
        { name: 'broken_agent', status: 'offline', message: 'HTTP 500.' },
      ],
    });
  });

  it('reports every agent offline while the runtime is unhealthy', async () => {
    const seen: string[] = [];
    const url = await runtimeServer((req, res) => {
      seen.push(req.url!);
      json(res, 503, { status: 'unhealthy', reason: 'Backend unreachable' });
    });
    expect(await agentsStatus(config(url), ['search_agent'])).toEqual({
      runtime: 'unhealthy',
      reason: 'Backend unreachable',
      agents: [{ name: 'search_agent', status: 'offline', message: 'The runtime is unhealthy: Backend unreachable.' }],
    });
    expect(seen).toEqual(['/health']);
  });

  it('answers 502 naming the runtime when it is down', async () => {
    const url = await deadUrl();
    const response = await createApp(config(url)).request('/ui-api/agents/status');
    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({ error: `The Cogniverse runtime at ${url} did not answer (TypeError).` });
  });
});

describe('GET /ui-api/agents', () => {
  it('relays the registry', async () => {
    const url = await runtimeServer((_req, res) =>
      res.end(JSON.stringify({ agents: ['search_agent'] })),
    );
    const response = await createApp(config(url)).request('/ui-api/agents');
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ agents: ['search_agent'] });
  });

  it('answers 502 with the reason when the runtime is down', async () => {
    const url = await deadUrl();
    const response = await createApp(config(url)).request('/ui-api/agents');
    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({
      error: `The Cogniverse runtime at ${url} did not answer (TypeError).`,
    });
  });
});

describe('runtime proxy', () => {
  it('forwards method, path, query and body, and no key, to an admin route', async () => {
    const seen: unknown[] = [];
    const url = await runtimeServer((req, res) => {
      let body = '';
      req.on('data', (chunk) => (body += chunk));
      req.on('end', () => {
        seen.push({
          method: req.method,
          url: req.url,
          auth: req.headers.authorization,
          type: req.headers['content-type'],
          cookie: req.headers.cookie,
          body,
        });
        res.statusCode = 409;
        res.setHeader('content-type', 'application/json');
        res.setHeader('x-internal', 'secret');
        res.end(JSON.stringify({ detail: 'Tenant acme:prod already exists' }));
      });
    });
    const response = await createApp(config(url)).request('/ui-api/runtime/admin/tenants?dry=1', {
      method: 'POST',
      headers: { 'content-type': 'application/json', cookie: 'session=x' },
      body: JSON.stringify({ tenant_id: 'acme:prod' }),
    });
    expect(seen).toEqual([
      {
        method: 'POST',
        url: '/admin/tenants?dry=1',
        auth: undefined,
        type: 'application/json',
        cookie: undefined,
        body: '{"tenant_id":"acme:prod"}',
      },
    ]);
    expect(response.status).toBe(409);
    expect(response.headers.get('x-internal')).toBe(null);
    expect(await response.json()).toEqual({ detail: 'Tenant acme:prod already exists' });
  });

  it('refuses routes outside the operations views without calling the runtime', async () => {
    let calls = 0;
    const url = await runtimeServer((_req, res) => {
      calls += 1;
      res.end('{}');
    });
    const app = createApp(config(url));
    for (const [method, path] of [
      ['POST', '/admin/harness/keys'],
      ['POST', '/admin/debug/memreset'],
      ['GET', '/admin/tenants/acme:prod/../../harness/keys'],
      ['GET', '/admin/tenants/acme:prod%2F..%2F..%2Fharness%2Fkeys'],
      ['POST', '/admin/tenant/acme:prod/optimize/runs/x%2F..%2F..%2F..%2Fharness%2Fkeys'],
      ['GET', '/admin/tenant/acme:prod/jobs'],
      ['GET', '/events/workflows/wf-1'],
      ['POST', '/v1/chat/completions'],
      ['POST', '/ingestion/start'],
      ['GET', '/events/ingestion/job-1'],
      ['DELETE', '/admin/memories/acme:prod'],
      ['DELETE', '/admin/tenant/acme:prod/memories/m1/archive'],
      ['POST', '/agents/register'],
      ['POST', '/admin/tenant/acme:prod/approvals/batch_1'],
      ['POST', '/admin/tenant/acme:prod/approvals/batch_1/item/extra'],
      ['POST', '/agents/annotations/queue/enqueue'],
      ['GET', '/agents/annotations/queue/span-1'],
      ['GET', '/admin/tenant/acme:prod/orchestration-workflows/abc123'],
      ['POST', '/admin/tenant/acme:prod/orchestration-workflows/abc123/annotation/extra'],
      ['POST', '/ag-ui/search_agent'],
      ['POST', '/ag-ui/results/relevance/extra'],
      ['POST', '/ag-ui/coding_agent'],
      ['POST', '/ag-ui/threads/t1/evaluation/extra'],
      ['GET', '/ag-ui/threads/t1'],
      ['GET', '/ingestion/profiles/extra'],
      ['GET', '/admin/tenant/acme:prod/telemetry/spans'],
      ['POST', '/admin/tenant/acme:prod/routing-decisions/abc123/delete'],
      ['GET', '/admin/tenant/acme:prod/routing-decisions/abc123'],
    ]) {
      const response = await app.request(`/ui-api/runtime${path}`, { method });
      expect(response.status).toBe(404);
    }
    expect(calls).toBe(0);
  });

  it('forwards every route the operations views call', async () => {
    const seen: string[] = [];
    const [route] = withAdmin(['acme:prod'], (req, res) => {
      seen.push(`${req.method} ${req.url}`);
      res.end('{}');
    });
    const url = await runtimeServer(route);
    const app = createApp(config(url));
    const calls = [
      ['GET', '/admin/organizations'],
      ['DELETE', '/admin/organizations/acme'],
      ['GET', '/admin/organizations/acme/tenants'],
      ['POST', '/admin/tenants'],
      ['DELETE', '/admin/tenants/acme:prod'],
      ['PUT', '/admin/tenants/acme:prod/tier'],
      ['GET', '/admin/router-tiers'],
      ['GET', '/admin/base-schemas'],
      ['GET', '/admin/profiles?tenant_id=acme:prod'],
      ['PUT', '/admin/profiles/p1'],
      ['POST', '/admin/profiles/p1/deploy'],
      ['GET', '/admin/profile-templates?tenant_id=acme:prod'],
      ['GET', '/admin/config/sections'],
      ['GET', '/admin/config/sections/routing?tenant_id=acme:prod'],
      ['PUT', '/admin/config/sections/agent'],
      ['GET', '/admin/config/entries?tenant_id=acme:prod'],
      ['GET', '/admin/config/history?scope=routing&service=gateway_agent&config_key=routing_config'],
      ['POST', '/admin/config/rollback'],
      ['GET', '/admin/config/export?tenant_id=acme:prod&include_history=false'],
      ['POST', '/admin/config/import'],
      ['GET', '/admin/config/stats'],
      ['GET', '/admin/config/health'],
      ['GET', '/admin/tenant/acme:prod/telemetry/root-causes?lookback_hours=24&include_slow=true&slow_percentile=95'],
      ['GET', '/admin/tenant/acme:prod/embeddings/atlas?profile=document_text_semantic&limit=500'],
      ['POST', '/ingestion/upload?force=true'],
      ['GET', '/ingestion/ingest_1/events?last-event-id=1-0'],
      ['GET', '/ingestion/ingest_1/status'],
      ['GET', '/admin/tenant/optimize-modes'],
      ['POST', '/admin/tenant/acme:prod/optimize'],
      ['GET', '/admin/tenant/acme:prod/optimize/runs'],
      ['GET', '/admin/tenant/acme:prod/optimize/runs/wf-1'],
      ['POST', '/admin/tenant/acme:prod/optimize/runs/wf-1/cancel'],
      ['POST', '/admin/tenant/acme:prod/optimize/runs/wf-1/retry'],
      ['GET', '/agents/'],
      ['GET', '/admin/tenant/acme:prod/memories/stats?agent_name=search_agent'],
      ['GET', '/admin/tenant/acme:prod/memories/health?agent_name=search_agent'],
      ['GET', '/admin/tenant/acme:prod/memories?agent_name=search_agent&limit=200&q=dark'],
      ['POST', '/admin/tenant/acme:prod/memories'],
      ['DELETE', '/admin/tenant/acme:prod/memories/m1?agent_name=search_agent'],
      ['DELETE', '/admin/tenant/acme:prod/memories?agent_name=search_agent'],
      ['GET', '/admin/tenant/acme:prod/approvals'],
      ['POST', '/admin/tenant/acme:prod/approvals/batch_1/batch_1_routing'],
      ['GET', '/agents/annotations/labels'],
      ['GET', '/agents/annotations/queue'],
      ['POST', '/agents/annotations/queue/span-1/assign'],
      ['POST', '/agents/annotations/queue/span-1/complete'],
      ['GET', '/admin/tenant/acme:prod/orchestration-workflows?lookback_hours=24'],
      ['POST', '/admin/tenant/acme:prod/orchestration-workflows/abc123/annotation'],
      ['POST', '/ag-ui/results/relevance'],
      ['POST', '/ag-ui/summarizer_agent'],
      ['POST', '/ag-ui/threads/thread-1/evaluation'],
      ['GET', '/ingestion/profiles?tenant_id=acme:prod'],
      ['GET', '/admin/tenant/acme:prod/telemetry/profile-selection?lookback_hours=24'],
      ['GET', '/admin/tenant/acme:prod/telemetry/rlm-ab?lookback_hours=168'],
      ['GET', '/admin/tenant/acme:prod/telemetry/traces?lookback_hours=24&operation=search&profile=a&profile=b'],
      ['GET', '/admin/tenant/acme:prod/telemetry/traces?start=2026-10-01T00%3A00%3A00.000Z&end=2026-10-02T00%3A00%3A00.000Z'],
      ['GET', '/admin/tenant/acme:prod/telemetry/phoenix'],
      ['GET', '/admin/tenant/acme:prod/evaluation/golden?lookback_hours=168'],
      ['GET', '/admin/tenant/acme:prod/routing-decisions?lookback_hours=24'],
      ['POST', '/admin/tenant/acme:prod/routing-decisions/abc123/approve'],
      ['PUT', '/admin/tenant/acme:prod/routing-decisions/abc123/label'],
    ];
    for (const [method, path] of calls) {
      const response = await app.request(`/ui-api/runtime${path}`, {
        method,
        headers: { [TENANT_HEADER]: 'acme:prod' },
      });
      expect(response.status).toBe(200);
    }
    expect(seen).toEqual(calls.map(([method, path]) => `${method} ${path}`));
  });

  it("sends an /ag-ui route with the named tenant's key, and none without a tenant", async () => {
    const seen: string[] = [];
    const [route, admin] = withAdmin(['acme:prod', 'beta:dev'], (req: IncomingMessage, res: ServerResponse) => {
      seen.push(`${req.method} ${req.url} ${admin.tenantOf(req.headers.authorization)}`);
      json(res, 200, { stored: true });
    });
    const url = await runtimeServer(route);
    const app = createApp(config(url));
    const rate = (headers: Record<string, string>) =>
      app.request('/ui-api/runtime/ag-ui/results/relevance', { method: 'POST', headers, body: '{}' });
    const [acme, beta] = await Promise.all([rate({ [TENANT_HEADER]: 'acme:prod' }), rate({ [TENANT_HEADER]: 'beta:dev' })]);
    expect([acme.status, beta.status]).toEqual([200, 200]);
    expect(seen.sort()).toEqual([
      'POST /ag-ui/results/relevance acme:prod',
      'POST /ag-ui/results/relevance beta:dev',
    ]);
    const anonymous = await rate({});
    expect(anonymous.status).toBe(400);
    expect(await anonymous.json()).toEqual({ error: 'Choose a tenant before talking to an agent.' });
    const unknown = await rate({ [TENANT_HEADER]: 'acme:typo' });
    expect(unknown.status).toBe(404);
    expect(await unknown.json()).toEqual({
      error: 'Tenant acme:typo is not registered. Register it with POST /admin/tenants first.',
    });
    expect(seen).toHaveLength(2);
  });

  it('streams server-sent events as the runtime sends them', async () => {
    let release: () => void = () => {};
    const url = await runtimeServer((_req, res) => {
      res.setHeader('content-type', 'text/event-stream');
      res.write('data: {"state":"running"}\n\n');
      release = () => res.end('data: {"state":"done"}\n\n');
    });
    const response = await createApp(config(url)).request('/ui-api/runtime/ingestion/job-1/events');
    expect(response.headers.get('content-type')).toBe('text/event-stream');
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    // The first event arrives while the runtime still holds the stream open.
    expect(decoder.decode((await reader.read()).value)).toBe('data: {"state":"running"}\n\n');
    release();
    let rest = '';
    for (let part = await reader.read(); !part.done; part = await reader.read())
      rest += decoder.decode(part.value);
    expect(rest).toBe('data: {"state":"done"}\n\n');
  });

  it('answers 502 naming the runtime when it is down', async () => {
    const url = await deadUrl();
    const response = await createApp(config(url)).request('/ui-api/runtime/admin/organizations');
    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({
      error: `The Cogniverse runtime at ${url} did not answer (TypeError).`,
    });
  });
});

describe('GET /healthz', () => {
  it('answers without calling the runtime', async () => {
    const response = await createApp(config(await deadUrl())).request('/healthz');
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ status: 'ok' });
  });

  it('is not shadowed by the built client', async () => {
    const clientDir = mkdtempSync(path.join(tmpdir(), 'cogniverse-web-'));
    writeFileSync(path.join(clientDir, 'index.html'), '<html>client</html>');
    const app = createApp({ ...config(await deadUrl()), clientDir });
    const health = await app.request('/healthz');
    expect(health.headers.get('content-type')).toMatch(/^application\/json/);
    expect(await health.json()).toEqual({ status: 'ok' });
    const page = await app.request('/tenants');
    expect(await page.text()).toBe('<html>client</html>');
  });
});

describe('server routes', () => {
  it('serves nothing under /api, which the ingress gives to the runtime', async () => {
    const url = await runtimeServer((_req, res) => res.end(JSON.stringify({ agents: ['x'] })));
    const app = createApp(config(url));
    for (const route of ['/api/agents', '/api/copilotkit/info', '/api/runtime/agents/'])
      expect((await app.request(route)).status).toBe(404);
  });
});
