import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterEach, describe, expect, it } from 'vitest';
import { createApp } from '../src/server/app';
import { ConfigError, loadConfig, type ServerConfig } from '../src/server/config';
import { RuntimeUnavailableError, cogniverseAgents, listAgents } from '../src/server/runtime';

const servers: Server[] = [];
afterEach(async () => {
  await Promise.all(servers.splice(0).map((s) => new Promise((r) => s.close(r))));
});

/** A real HTTP server standing in for the runtime's agent list. */
async function runtimeServer(
  handler: (req: IncomingMessage, res: ServerResponse) => void,
): Promise<string> {
  const server = createServer(handler);
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

/** A port nothing listens on. */
async function deadUrl(): Promise<string> {
  const server = createServer();
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  await new Promise((r) => server.close(r));
  return `http://127.0.0.1:${port}`;
}

function config(runtimeUrl: string): ServerConfig {
  return { runtimeUrl, apiKey: 'sk-test', port: 4000, host: '127.0.0.1', clientDir: '/nonexistent' };
}

describe('loadConfig', () => {
  it('reads and normalises the environment', () => {
    expect(
      loadConfig(
        { COGNIVERSE_RUNTIME_URL: ' http://rt:8000/api// ', COGNIVERSE_API_KEY: ' sk-1 ', PORT: '5173' },
        '/srv/client',
      ),
    ).toEqual({
      runtimeUrl: 'http://rt:8000/api',
      apiKey: 'sk-1',
      port: 5173,
      host: '127.0.0.1',
      clientDir: '/srv/client',
    });
  });

  it('names every missing variable', () => {
    expect(() => loadConfig({ COGNIVERSE_API_KEY: '  ' }, '/c')).toThrow(
      new ConfigError('Missing required environment: COGNIVERSE_RUNTIME_URL, COGNIVERSE_API_KEY.'),
    );
  });

  it('rejects a port that is not a port', () => {
    const env = { COGNIVERSE_RUNTIME_URL: 'http://rt', COGNIVERSE_API_KEY: 'k' };
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
  it('points each agent at its /ag-ui route with the harness key', () => {
    const agents = cogniverseAgents(config('http://rt:8000'), ['search_agent', 'a/b']);
    expect(Object.keys(agents)).toEqual(['search_agent', 'a/b']);
    expect(agents.search_agent.url).toBe('http://rt:8000/ag-ui/search_agent');
    expect(agents['a/b'].url).toBe('http://rt:8000/ag-ui/a%2Fb');
    expect(agents.search_agent.headers).toEqual({ Authorization: 'Bearer sk-test' });
  });
});

describe('GET /api/agents', () => {
  it('relays the registry', async () => {
    const url = await runtimeServer((_req, res) =>
      res.end(JSON.stringify({ agents: ['search_agent'] })),
    );
    const response = await createApp(config(url)).request('/api/agents');
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ agents: ['search_agent'] });
  });

  it('answers 502 with the reason when the runtime is down', async () => {
    const url = await deadUrl();
    const response = await createApp(config(url)).request('/api/agents');
    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({
      error: `The Cogniverse runtime at ${url} did not answer (TypeError).`,
    });
  });
});

describe('runtime proxy', () => {
  it('forwards method, path, query, body and the harness key', async () => {
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
    const response = await createApp(config(url)).request('/api/runtime/admin/tenants?dry=1', {
      method: 'POST',
      headers: { 'content-type': 'application/json', cookie: 'session=x' },
      body: JSON.stringify({ tenant_id: 'acme:prod' }),
    });
    expect(seen).toEqual([
      {
        method: 'POST',
        url: '/admin/tenants?dry=1',
        auth: 'Bearer sk-test',
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
    ]) {
      const response = await app.request(`/api/runtime${path}`, { method });
      expect(response.status).toBe(404);
    }
    expect(calls).toBe(0);
  });

  it('forwards every route the operations views call', async () => {
    const seen: string[] = [];
    const url = await runtimeServer((req, res) => {
      seen.push(`${req.method} ${req.url}`);
      res.end('{}');
    });
    const app = createApp(config(url));
    const calls = [
      ['GET', '/admin/organizations'],
      ['DELETE', '/admin/organizations/acme'],
      ['GET', '/admin/organizations/acme/tenants'],
      ['POST', '/admin/tenants'],
      ['DELETE', '/admin/tenants/acme:prod'],
      ['PUT', '/admin/tenants/acme:prod/tier'],
      ['GET', '/admin/router-tiers'],
      ['GET', '/admin/profiles?tenant_id=acme:prod'],
      ['PUT', '/admin/profiles/p1'],
      ['POST', '/admin/profiles/p1/deploy'],
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
    ];
    for (const [method, path] of calls) {
      const response = await app.request(`/api/runtime${path}`, { method });
      expect(response.status).toBe(200);
    }
    expect(seen).toEqual(calls.map(([method, path]) => `${method} ${path}`));
  });

  it('streams server-sent events as the runtime sends them', async () => {
    let release: () => void = () => {};
    const url = await runtimeServer((_req, res) => {
      res.setHeader('content-type', 'text/event-stream');
      res.write('data: {"state":"running"}\n\n');
      release = () => res.end('data: {"state":"done"}\n\n');
    });
    const response = await createApp(config(url)).request('/api/runtime/ingestion/job-1/events');
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
    const response = await createApp(config(url)).request('/api/runtime/admin/organizations');
    expect(response.status).toBe(502);
    expect(await response.json()).toEqual({
      error: `The Cogniverse runtime at ${url} did not answer (TypeError).`,
    });
  });
});
