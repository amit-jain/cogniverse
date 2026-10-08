import type { ServerConfig } from './config.js';
import { jsonError, requestTenant, tenantFetch, type TenantKeys } from './tenants.js';

/** Runtime routes the operations views call, relative to the runtime root. */
const ALLOWED = [
  /^\/admin\/organizations(\/[^/]+)?$/,
  /^\/admin\/organizations\/[^/]+\/tenants$/,
  /^\/admin\/tenants(\/[^/]+(\/tier)?)?$/,
  /^\/admin\/router-tiers$/,
  /^\/admin\/base-schemas$/,
  /^\/admin\/profiles(\/[^/]+(\/deploy)?)?$/,
  /^\/admin\/profile-templates$/,
  /^\/admin\/config\/(sections(\/[^/]+)?|entries|history|rollback|export|import|stats|health)$/,
  /^\/admin\/tenant\/optimize-modes$/,
  /^\/admin\/tenant\/[^/]+\/optimize(\/runs(\/[^/]+(\/(cancel|retry))?)?|\/report)?$/,
  /^\/admin\/tenant\/training-example-templates$/,
  /^\/admin\/tenant\/[^/]+\/training-examples$/,
  /^\/admin\/tenant\/[^/]+\/optimize\/runs\/[^/]+\/synthetic$/,
  /^\/admin\/tenant\/[^/]+\/(search-annotations(\/[^/]+)?|golden-dataset|synthetic\/settings|datasets|optimization-metrics)$/,
  /^\/admin\/tenant\/[^/]+\/profile-selection\/(analysis|train|model|predict)$/,
  /^\/admin\/tenant\/[^/]+\/memories(\/[^/]+)?$/,
  /^\/admin\/tenant\/[^/]+\/approvals(\/(history|stats)|\/[^/]+\/[^/]+(\/regenerate)?)?$/,
  /^\/admin\/tenant\/[^/]+\/orchestration-workflows(\/[^/]+\/annotation)?$/,
  /^\/admin\/tenant\/[^/]+\/telemetry\/(profile-selection|rlm-ab|traces|root-causes|phoenix)$/,
  /^\/admin\/tenant\/[^/]+\/evaluation\/(golden|datasets|dataset)$/,
  /^\/admin\/tenant\/[^/]+\/embeddings\/atlas(\/umap)?$/,
  /^\/admin\/tenant\/[^/]+\/routing-decisions(\/[^/]+\/(approve|label))?$/,
  /^\/admin\/tenant\/[^/]+\/routing-decisions\/(annotation-candidates|label-statistics)$/,
  /^\/ag-ui\/results\/relevance$/,
  /^\/ag-ui\/summarizer_agent$/,
  /^\/ag-ui\/threads\/[^/]+\/evaluation$/,
  /^\/agents\/$/,
  /^\/agents\/annotations\/labels$/,
  /^\/agents\/annotations\/queue(\/[^/]+\/(assign|complete))?$/,
  /^\/ingestion\/(upload|profiles)$/,
  /^\/ingestion\/[^/]+\/(events|status)$/,
];

const FORWARDED_REQUEST_HEADERS = ['content-type', 'accept'];
const FORWARDED_RESPONSE_HEADERS = ['content-type', 'cache-control'];

export function isAllowedRuntimePath(path: string): boolean {
  if (path.split('/').some((segment) => segment === '..' || segment === '.')) return false;
  return ALLOWED.some((pattern) => pattern.test(path));
}

/**
 * Forwards ``request`` (whose path starts with ``mount``) to the same path on
 * the runtime, streaming both bodies, so server-sent event routes stay live.
 * An ``/ag-ui`` route goes with the harness key of the tenant the request
 * names; the runtime takes every other route's tenant from its path.
 */
export async function forwardToRuntime(
  config: ServerConfig,
  request: Request,
  mount: string,
  keys: TenantKeys,
  fetchFn: typeof fetch = fetch,
): Promise<Response> {
  const url = new URL(request.url);
  const path = decodeURIComponent(url.pathname.slice(mount.length));
  if (!isAllowedRuntimePath(path))
    return jsonError(404, `The web server does not forward ${request.method} ${path}.`);

  const send = path.startsWith('/ag-ui/') ? tenantFetch(keys, requestTenant(request.headers), fetchFn) : fetchFn;
  const headers = new Headers();
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const hasBody = request.method !== 'GET' && request.method !== 'HEAD';
  let upstream: Response;
  try {
    upstream = await send(`${config.runtimeUrl}${url.pathname.slice(mount.length)}${url.search}`, {
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
