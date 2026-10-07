import type { ServerConfig } from './config.js';

/** Runtime routes the operations views call, relative to the runtime root. */
const ALLOWED = [
  /^\/admin\/organizations(\/[^/]+)?$/,
  /^\/admin\/organizations\/[^/]+\/tenants$/,
  /^\/admin\/tenants(\/[^/]+(\/tier)?)?$/,
  /^\/admin\/router-tiers$/,
  /^\/admin\/profiles(\/[^/]+(\/deploy)?)?$/,
  /^\/admin\/tenant\/optimize-modes$/,
  /^\/admin\/tenant\/[^/]+\/optimize(\/runs(\/[^/]+(\/(cancel|retry))?)?)?$/,
  /^\/admin\/tenant\/[^/]+\/memories(\/[^/]+)?$/,
  /^\/admin\/tenant\/[^/]+\/approvals(\/[^/]+\/[^/]+)?$/,
  /^\/admin\/tenant\/[^/]+\/orchestration-workflows(\/[^/]+\/annotation)?$/,
  /^\/admin\/tenant\/[^/]+\/telemetry\/(profile-selection|rlm-ab|traces)$/,
  /^\/admin\/tenant\/[^/]+\/evaluation\/golden$/,
  /^\/ag-ui\/results\/relevance$/,
  /^\/agents\/$/,
  /^\/agents\/annotations\/labels$/,
  /^\/agents\/annotations\/queue(\/[^/]+\/(assign|complete))?$/,
  /^\/ingestion\/upload$/,
  /^\/ingestion\/[^/]+\/(events|status)$/,
];

const FORWARDED_REQUEST_HEADERS = ['content-type', 'accept'];
const FORWARDED_RESPONSE_HEADERS = ['content-type', 'cache-control'];

export function isAllowedRuntimePath(path: string): boolean {
  if (path.split('/').some((segment) => segment === '..' || segment === '.')) return false;
  return ALLOWED.some((pattern) => pattern.test(path));
}

function jsonError(status: number, error: string): Response {
  return new Response(JSON.stringify({ error }), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

/**
 * Forwards ``request`` (whose path starts with ``mount``) to the same path on
 * the runtime with the harness key, streaming both bodies, so server-sent
 * event routes stay live.
 */
export async function forwardToRuntime(
  config: ServerConfig,
  request: Request,
  mount: string,
  fetchFn: typeof fetch = fetch,
): Promise<Response> {
  const url = new URL(request.url);
  const path = decodeURIComponent(url.pathname.slice(mount.length));
  if (!isAllowedRuntimePath(path))
    return jsonError(404, `The web server does not forward ${request.method} ${path}.`);

  const headers = new Headers({ Authorization: `Bearer ${config.apiKey}` });
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const hasBody = request.method !== 'GET' && request.method !== 'HEAD';
  let upstream: Response;
  try {
    upstream = await fetchFn(`${config.runtimeUrl}${url.pathname.slice(mount.length)}${url.search}`, {
      method: request.method,
      headers,
      body: hasBody ? request.body : undefined,
      signal: request.signal,
      // Required by Node's fetch to stream a request body.
      ...(hasBody ? { duplex: 'half' } : {}),
    } as RequestInit);
  } catch (error) {
    if (request.signal.aborted) throw error;
    return jsonError(
      502,
      `The Cogniverse runtime at ${config.runtimeUrl} did not answer (${
        error instanceof Error ? error.name : 'unknown error'
      }).`,
    );
  }
  const responseHeaders = new Headers();
  for (const name of FORWARDED_RESPONSE_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
}
