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
