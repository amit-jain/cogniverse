import './telemetry.js';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { serveStatic } from '@hono/node-server/serve-static';
import { CopilotRuntime, createCopilotHonoHandler } from '@copilotkit/runtime/v2';
import { Hono } from 'hono';
import type { ServerConfig } from './config.js';
import { forwardToRuntime } from './proxy.js';
import { RuntimeUnavailableError, cogniverseAgents, listAgents } from './runtime.js';
import { CogniverseThreadRunner } from './threads.js';

export function createApp(config: ServerConfig, fetchFn: typeof fetch = fetch) {
  const runtime = new CopilotRuntime({
    agents: async () => cogniverseAgents(config, await listAgents(config, fetchFn)),
    runner: new CogniverseThreadRunner(config, fetchFn),
  });
  const copilotkit = createCopilotHonoHandler({
    runtime,
    basePath: '/api/copilotkit',
    cors: { origin: [] },
  });

  const app = new Hono();
  app.get('/api/agents', async (c) => {
    try {
      return c.json({ agents: await listAgents(config, fetchFn) });
    } catch (error) {
      if (error instanceof RuntimeUnavailableError)
        return c.json({ error: error.message }, 502);
      throw error;
    }
  });
  app.all('/api/copilotkit/*', (c) => copilotkit.fetch(c.req.raw));
  app.all('/api/runtime/*', (c) => forwardToRuntime(config, c.req.raw, '/api/runtime', fetchFn));

  if (existsSync(path.join(config.clientDir, 'index.html'))) {
    app.use('/*', serveStatic({ root: path.relative(process.cwd(), config.clientDir) }));
    app.get('*', async (c) =>
      c.html(await readFile(path.join(config.clientDir, 'index.html'), 'utf8')),
    );
  }
  return app;
}
