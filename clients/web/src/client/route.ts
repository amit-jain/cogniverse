export type Route = { kind: 'agent'; name?: string } | { kind: 'ops'; id: string };

/** ``#/agents/search_agent`` or ``#/ops/tenants``; anything else is the agent default. */
export function parseRoute(hash: string): Route {
  const [section, id] = hash.replace(/^#\/?/, '').split('/').map(decodeURIComponent);
  if (section === 'ops' && id) return { kind: 'ops', id };
  if (section === 'agents' && id) return { kind: 'agent', name: id };
  return { kind: 'agent' };
}

export function routeHash(route: Route): string {
  if (route.kind === 'ops') return `#/ops/${encodeURIComponent(route.id)}`;
  return route.name ? `#/agents/${encodeURIComponent(route.name)}` : '#/';
}
