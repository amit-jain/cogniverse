export interface ServerConfig {
  /** Base URL of the Cogniverse runtime, including any ingress prefix. */
  runtimeUrl: string;
  port: number;
  host: string;
  /** Directory of the built client, served when it exists. */
  clientDir: string;
  /** Lifetime of each harness key the server mints; it renews a key at half of it. */
  harnessKeyTtlS: number;
}

/** The longest harness key lifetime the runtime issues. */
const MAX_HARNESS_KEY_TTL_S = 7 * 24 * 3600;

export class ConfigError extends Error {}

export function loadConfig(
  env: Record<string, string | undefined>,
  clientDir: string,
): ServerConfig {
  if (!env.COGNIVERSE_RUNTIME_URL?.trim())
    throw new ConfigError('Missing required environment: COGNIVERSE_RUNTIME_URL.');
  const port = Number(env.PORT ?? '4000');
  if (!Number.isInteger(port) || port < 1 || port > 65535)
    throw new ConfigError(`PORT must be an integer port number, got ${env.PORT}.`);
  const ttl = env.COGNIVERSE_WEB_HARNESS_KEY_TTL_S ?? '3600';
  const harnessKeyTtlS = /^\d+$/.test(ttl) ? Number(ttl) : NaN;
  if (!(harnessKeyTtlS >= 1 && harnessKeyTtlS <= MAX_HARNESS_KEY_TTL_S))
    throw new ConfigError(
      `COGNIVERSE_WEB_HARNESS_KEY_TTL_S must be a whole number of seconds from 1 to ${MAX_HARNESS_KEY_TTL_S}, got ${ttl}.`,
    );
  return {
    runtimeUrl: env.COGNIVERSE_RUNTIME_URL.trim().replace(/\/+$/, ''),
    port,
    host: env.HOST?.trim() || '127.0.0.1',
    clientDir,
    harnessKeyTtlS,
  };
}
