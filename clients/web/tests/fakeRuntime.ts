import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterEach } from 'vitest';

const servers: Server[] = [];
afterEach(async () => {
  await Promise.all(servers.splice(0).map((s) => new Promise((r) => s.close(r))));
});

export function json(res: ServerResponse, status: number, body: unknown) {
  res.statusCode = status;
  res.setHeader('content-type', 'application/json');
  res.end(JSON.stringify(body));
}

/** A real HTTP server standing in for the runtime. */
export async function runtimeServer(
  handler: (req: IncomingMessage, res: ServerResponse) => void,
): Promise<string> {
  const server = createServer(handler);
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

/** A port nothing listens on. */
export async function deadUrl(): Promise<string> {
  const server = createServer();
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  await new Promise((r) => server.close(r));
  return `http://127.0.0.1:${port}`;
}

export interface Admin {
  /** Every tenant probe, key mint and key revocation, in arrival order. */
  calls: string[];
  /** The ``ttl_seconds`` each mint asked for, in arrival order. */
  ttls: unknown[];
  /** The bearer each tenant's latest key is sent with. */
  bearer(tenant: string): string;
  /** The tenant a bearer key was minted for. */
  tenantOf(authorization: string | undefined): string | undefined;
}

/**
 * The runtime's tenant registry and harness-key admin over ``tenants``, with
 * every other request handed to ``handler``. Key ``n`` minted for a tenant is
 * ``key-<tenant>-<n>``.
 */
export function withAdmin(
  tenants: string[],
  handler: (req: IncomingMessage, res: ServerResponse, admin: Admin) => void = (_req, res) => json(res, 404, {}),
): [(req: IncomingMessage, res: ServerResponse) => void, Admin] {
  const minted = new Map<string, string>();
  const latest = new Map<string, string>();
  let count = 0;
  const admin: Admin = {
    calls: [],
    ttls: [],
    bearer: (tenant) => `Bearer ${latest.get(tenant)}`,
    tenantOf: (authorization) => minted.get(authorization?.replace(/^Bearer /, '') ?? ''),
  };
  const route = (req: IncomingMessage, res: ServerResponse) => {
    const url = req.url ?? '';
    const tenant = url.match(/^\/admin\/tenants\/([^/?]+)$/);
    if (req.method === 'GET' && tenant) {
      const id = decodeURIComponent(tenant[1]);
      admin.calls.push(`probe ${id}`);
      if (tenants.includes(id)) json(res, 200, { tenant_full_id: id });
      else json(res, 404, { detail: `Tenant ${id} not found` });
      return;
    }
    if (req.method === 'POST' && url === '/admin/harness/keys') {
      let body = '';
      req.on('data', (chunk) => (body += chunk));
      req.on('end', () => {
        const { tenant_id: id, name, ttl_seconds: ttl } = JSON.parse(body) as {
          tenant_id: string;
          name: string;
          ttl_seconds?: unknown;
        };
        count += 1;
        const key = `key-${id}-${count}`;
        minted.set(key, id);
        latest.set(id, key);
        admin.calls.push(`mint ${id} ${name}`);
        admin.ttls.push(ttl);
        const created = new Date();
        json(res, 200, {
          key,
          key_hash: `hash-${count}`,
          tenant_id: id,
          name,
          created_at: created.toISOString(),
          expires_at: typeof ttl === 'number' ? new Date(created.getTime() + ttl * 1000).toISOString() : null,
          revoked: false,
        });
      });
      return;
    }
    const revoke = url.match(/^\/admin\/harness\/keys\/([^/]+)$/);
    if (req.method === 'DELETE' && revoke) {
      admin.calls.push(`revoke ${revoke[1]}`);
      json(res, 200, { revoked: true, key_hash: revoke[1] });
      return;
    }
    handler(req, res, admin);
  };
  return [route, admin];
}
