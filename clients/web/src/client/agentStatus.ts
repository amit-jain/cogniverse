import { useEffect, useState } from 'react';

export interface AgentStatus {
  name: string;
  status: 'online' | 'offline';
  health?: string;
  message?: string;
}

export interface AgentsStatus {
  runtime: string;
  reason?: string;
  agents: AgentStatus[];
}

/** How often the sidebar re-checks the agents. */
export const AGENT_STATUS_INTERVAL_MS = 30_000;

export async function fetchAgentsStatus(signal?: AbortSignal): Promise<AgentsStatus> {
  const response = await fetch('/ui-api/agents/status', { signal });
  const body = (await response.json().catch(() => ({ error: 'The server returned an unreadable response.' }))) as
    | (AgentsStatus & { error?: undefined })
    | { error: string };
  if (!response.ok || body.error !== undefined)
    throw new Error(body.error ?? `Checking the agents failed (HTTP ${response.status}).`);
  return body as AgentsStatus;
}

/** The agents' status, checked now and every ``AGENT_STATUS_INTERVAL_MS``. */
export function useAgentsStatus(): { status?: AgentsStatus; error?: string } {
  const [state, setState] = useState<{ status?: AgentsStatus; error?: string }>({});
  useEffect(() => {
    let controller = new AbortController();
    const check = () => {
      controller.abort();
      controller = new AbortController();
      const signal = controller.signal;
      fetchAgentsStatus(signal)
        .then((status) => setState({ status }))
        .catch((e: unknown) => {
          if (!signal.aborted) setState({ error: e instanceof Error ? e.message : String(e) });
        });
    };
    check();
    const timer = setInterval(check, AGENT_STATUS_INTERVAL_MS);
    return () => {
      clearInterval(timer);
      controller.abort();
    };
  }, []);
  return state;
}
