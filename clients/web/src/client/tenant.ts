import { useSyncExternalStore } from 'react';

/** The header naming the tenant the web server acts for. */
export const TENANT_HEADER = 'x-cogniverse-tenant';

const STORAGE_KEY = 'cogniverse.tenant';
/** How long a confirmed registration is taken as current. */
const PROBE_TTL_MS = 30_000;
/** How long the registration probe may take before the runtime counts as
 * unreachable. */
const PROBE_TIMEOUT_MS = 20_000;

export interface TenantNotice {
  tone: 'warning' | 'error';
  text: string;
}

export interface TenantState {
  /** The active tenant (canonical when the runtime confirmed it), or ''. */
  tenant: string;
  /** What the last choice of a tenant left to say, if anything. */
  notice?: TenantNotice;
  /** A choice is being checked with the runtime. */
  checking: boolean;
}

/** What asking the runtime about a tenant found. */
export type Probe =
  | { kind: 'registered'; tenant: string }
  | { kind: 'unknown' }
  | { kind: 'invalid'; detail: string }
  | { kind: 'unreachable'; detail: string };

/** What to do with a chosen tenant, given the probe and whether the runtime
 * confirmed the same tenant before. */
export function decideTenant(
  probe: Probe,
  chosen: string,
  previouslyConfirmed: boolean,
): { tenant?: string; notice?: TenantNotice } {
  if (probe.kind === 'registered') return { tenant: probe.tenant };
  if (probe.kind === 'unknown')
    return {
      notice: {
        tone: 'error',
        text:
          `Tenant ${chosen} is not registered. Register it in the Tenants view ` +
          '(or with POST /admin/tenants) first, or choose a registered tenant.',
      },
    };
  if (probe.kind === 'invalid')
    return { notice: { tone: 'error', text: `Tenant ${chosen} cannot be used: ${probe.detail}` } };
  if (previouslyConfirmed)
    return {
      tenant: chosen,
      notice: {
        tone: 'warning',
        text: `Could not re-check tenant ${chosen} (${probe.detail}). Continuing with the last successful check.`,
      },
    };
  return {
    tenant: chosen,
    notice: {
      tone: 'warning',
      text:
        `The runtime could not confirm tenant ${chosen} is registered (${probe.detail}); ` +
        'its views read as empty if it is not.',
    },
  };
}

/** Asks the runtime (through the web server) whether ``tenant`` is registered. */
export async function probeTenant(tenant: string, fetchFn: typeof fetch = fetch): Promise<Probe> {
  let response: Response;
  try {
    response = await fetchFn(`/ui-api/runtime/admin/tenants/${encodeURIComponent(tenant)}`, {
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
  } catch (error) {
    const timedOut = error instanceof DOMException && error.name === 'TimeoutError';
    return {
      kind: 'unreachable',
      detail: timedOut
        ? `the runtime did not answer within ${PROBE_TIMEOUT_MS / 1000} s`
        : 'the web server did not answer',
    };
  }
  const body = (await response.json().catch(() => null)) as
    | { tenant_full_id?: unknown; detail?: unknown; error?: unknown }
    | null;
  if (response.status === 200 && typeof body?.tenant_full_id === 'string')
    return { kind: 'registered', tenant: body.tenant_full_id };
  if (response.status === 404 && typeof body?.detail === 'string') return { kind: 'unknown' };
  const reason =
    typeof body?.error === 'string'
      ? body.error
      : typeof (body?.detail as { message?: unknown } | undefined)?.message === 'string'
        ? (body!.detail as { message: string }).message
        : typeof body?.detail === 'string'
          ? body.detail
          : `HTTP ${response.status}`;
  if (response.status >= 400 && response.status < 500) return { kind: 'invalid', detail: reason };
  return { kind: 'unreachable', detail: reason };
}

function stored(): string {
  try {
    return localStorage.getItem(STORAGE_KEY) ?? '';
  } catch {
    return '';
  }
}

function store(tenant: string) {
  try {
    if (tenant) localStorage.setItem(STORAGE_KEY, tenant);
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Without storage the choice lasts until the page reloads.
  }
}

let state: TenantState = { tenant: stored(), checking: false };
const listeners = new Set<() => void>();
/** When the runtime last confirmed each tenant. */
const confirmed = new Map<string, number>();
let latest = 0;

function publish(next: TenantState) {
  state = next;
  for (const listener of listeners) listener();
}

/** The active tenant; for code outside React. */
export function currentTenant(): string {
  return state.tenant;
}

/** The whole tenant state; for code outside React. */
export function tenantState(): TenantState {
  return state;
}

/**
 * Makes ``raw`` the active tenant once the runtime confirms it is registered;
 * a tenant the runtime does not know is refused, and one it could not check
 * is taken with a warning. Answers the tenant now active.
 */
export async function chooseTenant(raw: string, fetchFn: typeof fetch = fetch): Promise<string> {
  const chosen = raw.trim();
  if (!chosen) return state.tenant;
  const attempt = ++latest;
  const fresh = Date.now() - (confirmed.get(chosen) ?? -Infinity) < PROBE_TTL_MS;
  if (fresh) {
    publish({ tenant: chosen, checking: false });
    store(chosen);
    return chosen;
  }
  publish({ ...state, checking: true });
  const probe = await probeTenant(chosen, fetchFn);
  if (attempt !== latest) return state.tenant;
  const decision = decideTenant(probe, chosen, confirmed.has(chosen));
  if (probe.kind === 'registered') confirmed.set(probe.tenant, Date.now());
  if (probe.kind === 'unknown') confirmed.delete(chosen);
  // A refused tenant that was active (one kept from an earlier visit) is
  // dropped rather than kept.
  const tenant = decision.tenant ?? (state.tenant === chosen ? '' : state.tenant);
  publish({ tenant, notice: decision.notice, checking: false });
  store(tenant);
  return tenant;
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** The active tenant, shared by every view. */
export function useTenant(): TenantState {
  return useSyncExternalStore(subscribe, tenantState);
}

/** Forget the store's state; for tests. */
export function resetTenant(tenant = '') {
  confirmed.clear();
  latest = 0;
  publish({ tenant, checking: false });
}
