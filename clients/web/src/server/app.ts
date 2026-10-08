import './telemetry.js';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { serveStatic } from '@hono/node-server/serve-static';
import { CopilotRuntime, createCopilotHonoHandler } from '@copilotkit/runtime/v2';
import { Hono } from 'hono';
import type { ServerConfig } from './config.js';
import { forwardToRuntime } from './proxy.js';
import { RuntimeUnavailableError, agentsStatus, cogniverseAgents, listAgents } from './runtime.js';
import { TenantKeys, requestTenant } from './tenants.js';
import { CogniverseThreadRunner } from './threads.js';

/**
 * The web server: the client, CopilotKit's runtime for the agent workspace
 * and the runtime proxy for the operations views. Both act for the tenant a
 * request names in ``x-cogniverse-tenant``, with that tenant's key from
 * ``keys``.
 */
export function createApp(
  config: ServerConfig,
  fetchFn: typeof fetch = fetch,
  keys: TenantKeys = new TenantKeys(config, fetchFn),
) {
  const runtime = new CopilotRuntime({
    agents: async ({ request }) =>
      cogniverseAgents(config, await listAgents(config, fetchFn), keys, requestTenant(request.headers), fetchFn),
    runner: new CogniverseThreadRunner(config, keys, fetchFn),
  });
  const copilotkit = createCopilotHonoHandler({
    runtime,
    basePath: '/ui-api/copilotkit',
    cors: { origin: [] },
  });

  const app = new Hono();
  // Liveness and readiness: answers while the process serves, whatever the
  // runtime's state, so a runtime outage never restarts this pod.
  app.get('/healthz', (c) => c.json({ status: 'ok' }));
  app.get('/ui-api/agents', async (c) => {
    try {
      return c.json({ agents: await listAgents(config, fetchFn) });
    } catch (error) {
      if (error instanceof RuntimeUnavailableError)
        return c.json({ error: error.message }, 502);
      throw error;
    }
  });
  app.get('/ui-api/agents/status', async (c) => {
    try {
      return c.json(await agentsStatus(config, await listAgents(config, fetchFn), fetchFn));
    } catch (error) {
      if (error instanceof RuntimeUnavailableError)
        return c.json({ error: error.message }, 502);
      throw error;
    }
  });
  app.all('/ui-api/copilotkit/*', (c) => copilotkit.fetch(c.req.raw));
  app.all('/ui-api/runtime/*', (c) => forwardToRuntime(config, c.req.raw, '/ui-api/runtime', keys, fetchFn));

  if (existsSync(path.join(config.clientDir, 'index.html'))) {
    app.use('/*', serveStatic({ root: path.relative(process.cwd(), config.clientDir) }));
    app.get('*', async (c) =>
      c.html(await readFile(path.join(config.clientDir, 'index.html'), 'utf8')),
    );
  }
  return app;
}
