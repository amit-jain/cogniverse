import { HttpAgent } from '@ag-ui/client';
import type { ServerConfig } from './config.js';

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

/** One AG-UI agent per Cogniverse agent, each running on the runtime's /ag-ui surface. */
export function cogniverseAgents(
  config: ServerConfig,
  names: string[],
): Record<string, HttpAgent> {
  return Object.fromEntries(
    names.map((name) => [
      name,
      new HttpAgent({
        agentId: name,
        url: `${config.runtimeUrl}/ag-ui/${encodeURIComponent(name)}`,
        headers: { Authorization: `Bearer ${config.apiKey}` },
      }),
    ]),
  );
}
