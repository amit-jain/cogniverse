import { HttpAgent } from '@ag-ui/client';
import type { ServerConfig } from './config.js';
import { TENANT_HEADER, tenantFetch, type TenantKeys } from './tenants.js';

export class RuntimeUnavailableError extends Error {}

/** Names of the agents the Cogniverse runtime serves, in registry order. */
export async function listAgents(
  config: ServerConfig,
  fetchFn: typeof fetch = fetch,
): Promise<string[]> {
  let response: Response;
  try {
    response = await fetchFn(`${config.runtimeUrl}/agents/`, {
      signal: AbortSignal.timeout(10000),
    });
  } catch (error) {
    throw new RuntimeUnavailableError(
      `The Cogniverse runtime at ${config.runtimeUrl} did not answer (${
        error instanceof Error ? error.name : 'unknown error'
      }).`,
    );
  }
  if (!response.ok)
    throw new RuntimeUnavailableError(
      `The Cogniverse runtime answered the agent list with HTTP ${response.status}.`,
    );
  const body = (await response.json()) as { agents?: unknown };
  if (
    !Array.isArray(body.agents) ||
    !body.agents.every((name) => typeof name === 'string')
  )
    throw new RuntimeUnavailableError(
      'The Cogniverse runtime answered the agent list without an agents array.',
    );
  return body.agents;
}

/** One AG-UI agent per Cogniverse agent, each running on the runtime's
 * /ag-ui surface with ``tenant``'s harness key. */
export function cogniverseAgents(
  config: ServerConfig,
  names: string[],
  keys: TenantKeys,
  tenant: string | undefined,
  fetchFn: typeof fetch = fetch,
): Record<string, HttpAgent> {
  return Object.fromEntries(
    names.map((name) => [
      name,
      new HttpAgent({
        agentId: name,
        url: `${config.runtimeUrl}/ag-ui/${encodeURIComponent(name)}`,
        headers: tenant ? { [TENANT_HEADER]: tenant } : {},
        fetch: tenantFetch(keys, tenant, fetchFn),
      }),
    ]),
  );
}

export interface AgentStatus {
  name: string;
  status: 'online' | 'offline';
  /** The registry's last health check of the agent, when it answered. */
  health?: string;
  message?: string;
}

export interface AgentsStatus {
  runtime: string;
  reason?: string;
  agents: AgentStatus[];
}

const STATUS_TIMEOUT_MS = 10_000;

/**
 * Whether the runtime and each of ``names`` can serve: the runtime's
 * ``/health`` answers, and ``/agents/{name}`` finds the agent registered.
 * An agent is online only while the runtime reports itself healthy or
 * degraded.
 */
export async function agentsStatus(
  config: ServerConfig,
  names: string[],
  fetchFn: typeof fetch = fetch,
): Promise<AgentsStatus> {
  let health: Response;
  try {
    health = await fetchFn(`${config.runtimeUrl}/health`, { signal: AbortSignal.timeout(STATUS_TIMEOUT_MS) });
  } catch (error) {
    throw new RuntimeUnavailableError(
      `The Cogniverse runtime at ${config.runtimeUrl} did not answer (${
        error instanceof Error ? error.name : 'unknown error'
      }).`,
    );
  }
  const body = (await health.json().catch(() => null)) as { status?: unknown; reason?: unknown } | null;
  const runtime = typeof body?.status === 'string' ? body.status : `HTTP ${health.status}`;
  const serving = health.ok && (runtime === 'healthy' || runtime === 'degraded');
  const reason = typeof body?.reason === 'string' ? body.reason : undefined;
  const agents = await Promise.all(
    names.map(async (name): Promise<AgentStatus> => {
      if (!serving) return { name, status: 'offline', message: `The runtime is ${runtime}${reason ? `: ${reason}` : ''}.` };
      let response: Response;
      try {
        response = await fetchFn(`${config.runtimeUrl}/agents/${encodeURIComponent(name)}`, {
          signal: AbortSignal.timeout(STATUS_TIMEOUT_MS),
        });
      } catch {
        return { name, status: 'offline', message: 'The agent did not answer.' };
      }
      if (response.status === 404) return { name, status: 'offline', message: 'Not registered.' };
      if (!response.ok) return { name, status: 'offline', message: `HTTP ${response.status}.` };
      const agent = (await response.json().catch(() => null)) as { health_status?: unknown } | null;
      return typeof agent?.health_status === 'string'
        ? { name, status: 'online', health: agent.health_status }
        : { name, status: 'online' };
    }),
  );
  return reason ? { runtime, reason, agents } : { runtime, agents };
}
