import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { serve } from '@hono/node-server';
import { createApp } from './app.js';
import { loadConfig } from './config.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const config = loadConfig(process.env, path.resolve(here, '../client'));
const server = serve({
  fetch: createApp(config).fetch,
  port: config.port,
  hostname: config.host,
});
console.log(`cogniverse-web listening on http://${config.host}:${config.port}`);

for (const signal of ['SIGINT', 'SIGTERM'] as const)
  process.on(signal, () => server.close(() => process.exit(0)));
