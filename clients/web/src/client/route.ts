export type Route = { kind: 'agent'; name?: string; thread?: string } | { kind: 'ops'; id: string };

/**
 * ``#/agents/search_agent/<thread>``, ``#/agents/search_agent`` or
 * ``#/ops/tenants``; anything else is the agent default.
 */
export function parseRoute(hash: string): Route {
  const [section, id, thread] = hash.replace(/^#\/?/, '').split('/').map(decodeURIComponent);
  if (section === 'ops' && id) return { kind: 'ops', id };
  if (section === 'agents' && id) return thread ? { kind: 'agent', name: id, thread } : { kind: 'agent', name: id };
  return { kind: 'agent' };
}

export function routeHash(route: Route): string {
  if (route.kind === 'ops') return `#/ops/${encodeURIComponent(route.id)}`;
  if (!route.name) return '#/';
  const agent = `#/agents/${encodeURIComponent(route.name)}`;
  return route.thread ? `${agent}/${encodeURIComponent(route.thread)}` : agent;
}
