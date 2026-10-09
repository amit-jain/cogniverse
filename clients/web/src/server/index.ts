import type { Server } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { serve } from '@hono/node-server';
import { createApp } from './app.js';
import { loadConfig } from './config.js';
import { TenantKeys } from './tenants.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const config = loadConfig(process.env, path.resolve(here, '../client'));
const keys = new TenantKeys(config);
const server = serve(
  {
    fetch: createApp(config, fetch, keys).fetch,
    port: config.port,
    hostname: config.host,
  },
  () => console.log(`cogniverse-web listening on http://${config.host}:${config.port}`),
);

/** How long requests in flight (streams included) may run once asked to stop. */
const SHUTDOWN_GRACE_MS = 5_000;

for (const signal of ['SIGINT', 'SIGTERM'] as const)
  process.on(signal, () => {
    const http = server as Server;
    // Closing drops idle connections; one still serving a request is cut
    // after the grace period. The harness keys this server minted are
    // revoked meanwhile.
    const closed = new Promise<void>((resolve) => http.close(() => resolve()));
    Promise.all([closed, keys.revokeAll()]).then(() => process.exit(0));
    setTimeout(() => http.closeAllConnections(), SHUTDOWN_GRACE_MS).unref();
  });
