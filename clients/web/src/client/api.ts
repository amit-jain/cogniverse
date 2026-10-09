export async function fetchAgents(signal?: AbortSignal): Promise<string[]> {
  const response = await fetch('/ui-api/agents', { signal });
  const body = (await response
    .json()
    .catch(() => ({ error: 'The server returned an unreadable response.' }))) as {
    agents?: string[];
    error?: string;
  };
  if (!response.ok || !body.agents)
    throw new Error(body.error ?? `Loading agents failed (HTTP ${response.status}).`);
  return body.agents;
}

/** "detailed_report_agent" -> "Detailed report". */
export function agentLabel(name: string): string {
  const words = name.replace(/_agent$/, '').split('_').filter(Boolean);
  const text = words.join(' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** A stable hue per agent, so each Dot keeps its colour. */
export function agentHue(name: string): number {
  let hash = 0;
  for (const char of name) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return hash % 360;
}
