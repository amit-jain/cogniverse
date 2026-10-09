import { hostname } from 'node:os';
import type { ServerConfig } from './config.js';

/** The request header naming the tenant a browser acts for. */
export const TENANT_HEADER = 'x-cogniverse-tenant';

/** The runtime answered that the tenant is not registered. */
export class UnknownTenantError extends Error {}

/** The runtime could not confirm the tenant or issue its key. */
export class TenantAuthUnavailableError extends Error {}

interface Minted {
  key: string;
  keyHash: string;
  /** When the key stops authenticating, by this server's clock (ms). */
  expiresAt: number;
}

/** How long one call to the runtime's tenant or key admin may take. */
const ADMIN_TIMEOUT_MS = 20_000;
/** How long revoking the minted keys may hold up a shutdown. */
const REVOKE_TIMEOUT_MS = 3_000;

function reason(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === 'string') return `HTTP ${status}: ${detail}`;
  const message = (detail as { message?: unknown } | null)?.message;
  return typeof message === 'string' ? `HTTP ${status}: ${message}` : `HTTP ${status}`;
}

/**
 * The harness key the server acts with for each tenant.
 *
 * A tenant's key is minted through the runtime's ``POST /admin/harness/keys``
 * with ``config.harnessKeyTtlS`` as its ttl, so a key outlives a server that
 * stops without revoking it by at most that long. Once less than half the ttl
 * remains the next request mints a replacement; concurrent requests share one
 * mint, first or renewal, and a failed renewal fails them as a first mint
 * would while the old key stays held. A superseded key is left to expire. A
 * tenant ``GET /admin/tenants/{id}`` answers is not registered gets none; one
 * the registry could not be read for (5xx) does, since the runtime checks the
 * tenant again on every data path it serves. The runtime resolves every AG-UI
 * call's tenant from its key, so a browser acting for one tenant can never
 * read another's runs. A key the runtime rejects is forgotten and the next
 * request mints a new one; ``revokeAll`` revokes every key this server minted
 * that has not expired.
 */
export class TenantKeys {
  /** The mint in flight for each tenant. */
  private readonly pending = new Map<string, Promise<Minted>>();
  /** The key each tenant's latest settled mint issued. */
  private readonly current = new Map<string, Minted>();
  /** Every key this server minted that has not expired. */
  private minted: Minted[] = [];

  constructor(
    private readonly config: ServerConfig,
    private readonly fetchFn: typeof fetch = fetch,
    readonly keyName = `cogniverse-web ${hostname()}`,
    private readonly now: () => number = Date.now,
  ) {}

  /** The key for ``tenant``, minting it on first use and renewing it once
   * less than half its ttl remains. */
  async keyFor(tenant: string): Promise<string> {
    const held = this.current.get(tenant);
    if (held && held.expiresAt - this.now() >= (this.config.harnessKeyTtlS * 1000) / 2) return held.key;
    let pending = this.pending.get(tenant);
    if (!pending) {
      const mint = this.mint(tenant);
      pending = mint;
      this.pending.set(tenant, mint);
      const settle = () => {
        if (this.pending.get(tenant) === mint) this.pending.delete(tenant);
      };
      mint.then(settle, settle);
    }
    return (await pending).key;
  }

  /** Drop ``key`` for ``tenant`` (the runtime rejected it) unless a newer
   * key already replaced it. */
  forget(tenant: string, key: string): void {
    if (this.current.get(tenant)?.key === key) this.current.delete(tenant);
  }

  /** Revoke every key this server minted; failures are logged, not raised. */
  async revokeAll(): Promise<void> {
    const keys = this.unexpired();
    this.minted = [];
    this.pending.clear();
    this.current.clear();
    const results = await Promise.allSettled(
      keys.map(async ({ keyHash }) => {
        const response = await this.fetchFn(`${this.config.runtimeUrl}/admin/harness/keys/${keyHash}`, {
          method: 'DELETE',
          signal: AbortSignal.timeout(REVOKE_TIMEOUT_MS),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
      }),
    );
    results.forEach((result, index) => {
      if (result.status === 'rejected')
        console.error(
          `cogniverse-web could not revoke harness key ${keys[index].keyHash.slice(0, 12)}: ${
            result.reason instanceof Error ? result.reason.message : String(result.reason)
          }`,
        );
    });
  }

  private unexpired(): Minted[] {
    const now = this.now();
    return this.minted.filter((minted) => minted.expiresAt > now);
  }

  private async admin(path: string, init: RequestInit = {}): Promise<{ status: number; body: unknown }> {
    let response: Response;
    try {
      response = await this.fetchFn(`${this.config.runtimeUrl}${path}`, {
        ...init,
        signal: AbortSignal.timeout(ADMIN_TIMEOUT_MS),
      });
    } catch (error) {
      throw new TenantAuthUnavailableError(
        `The Cogniverse runtime at ${this.config.runtimeUrl} did not answer (${
          error instanceof Error ? error.name : 'unknown error'
        }).`,
      );
    }
    return { status: response.status, body: await response.json().catch(() => null) };
  }

  private async mint(tenant: string): Promise<Minted> {
    const probe = await this.admin(`/admin/tenants/${encodeURIComponent(tenant)}`);
    const detail = (probe.body as { detail?: unknown } | null)?.detail;
    if (probe.status === 404 && typeof detail === 'string' && detail.includes(tenant))
      throw new UnknownTenantError(
        `Tenant ${tenant} is not registered. Register it with POST /admin/tenants first.`,
      );
    if (probe.status >= 500)
      console.warn(
        `cogniverse-web: the runtime could not confirm tenant ${tenant} (${reason(probe.body, probe.status)}); ` +
          'minting its key anyway.',
      );
    else if (probe.status !== 200)
      throw new TenantAuthUnavailableError(
        `The runtime could not confirm tenant ${tenant} (${reason(probe.body, probe.status)}).`,
      );
    const ttlS = this.config.harnessKeyTtlS;
    // The expiry is counted from before the request, so it never trails the runtime's.
    const expiresAt = this.now() + ttlS * 1000;
    const issued = await this.admin('/admin/harness/keys', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ tenant_id: tenant, name: this.keyName, ttl_seconds: ttlS }),
    });
    const body = issued.body as { key?: unknown; key_hash?: unknown } | null;
    if (issued.status !== 200 || typeof body?.key !== 'string' || typeof body.key_hash !== 'string')
      throw new TenantAuthUnavailableError(
        `The runtime did not issue a harness key for tenant ${tenant} (${reason(issued.body, issued.status)}).`,
      );
    const minted = { key: body.key, keyHash: body.key_hash, expiresAt };
    this.minted = [...this.unexpired(), minted];
    this.current.set(tenant, minted);
    return minted;
  }
}

/** A JSON error a tenant-key failure answers with. */
export function tenantKeyFailure(error: unknown): Response | undefined {
  if (error instanceof UnknownTenantError) return jsonError(404, error.message);
  if (error instanceof TenantAuthUnavailableError) return jsonError(503, error.message);
  return undefined;
}

export function jsonError(status: number, error: string): Response {
  return new Response(JSON.stringify({ error }), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

/** The tenant a request names in ``TENANT_HEADER``, if any. */
export function requestTenant(headers: Headers | Record<string, string> | undefined): string | undefined {
  const value =
    headers instanceof Headers
      ? headers.get(TENANT_HEADER)
      : Object.entries(headers ?? {}).find(([name]) => name.toLowerCase() === TENANT_HEADER)?.[1];
  return value?.trim() || undefined;
}

/**
 * A fetch that sends ``tenant``'s key as the bearer, answering the key's
 * failure as the response. A key the runtime rejects with 401 (revoked, say,
 * when the tenant was deleted) is forgotten, and a request whose body can be
 * sent again is retried once with a newly minted key.
 */
export function tenantFetch(keys: TenantKeys, tenant: string | undefined, fetchFn: typeof fetch = fetch) {
  const send = async (url: string, init: RequestInit): Promise<{ response: Response; key?: string }> => {
    let key: string;
    try {
      key = await keys.keyFor(tenant!);
    } catch (error) {
      const failure = tenantKeyFailure(error);
      if (failure) return { response: failure };
      throw error;
    }
    const headers = new Headers(init.headers);
    headers.set('authorization', `Bearer ${key}`);
    return { response: await fetchFn(url, { ...init, headers }), key };
  };
  return async (url: string, init: RequestInit = {}): Promise<Response> => {
    if (!tenant) return jsonError(400, 'Choose a tenant before talking to an agent.');
    const first = await send(url, init);
    if (first.response.status !== 401 || first.key === undefined) return first.response;
    keys.forget(tenant, first.key);
    const replayable = init.body === undefined || init.body === null || typeof init.body === 'string';
    if (!replayable) return first.response;
    await first.response.body?.cancel();
    return (await send(url, init)).response;
  };
}
