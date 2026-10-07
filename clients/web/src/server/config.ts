export interface ServerConfig {
  /** Base URL of the Cogniverse runtime, including any ingress prefix. */
  runtimeUrl: string;
  /** Harness key sent to the runtime; it decides the tenant. */
  apiKey: string;
  port: number;
  host: string;
  /** Directory of the built client, served when it exists. */
  clientDir: string;
}

export class ConfigError extends Error {}

export function loadConfig(
  env: Record<string, string | undefined>,
  clientDir: string,
): ServerConfig {
  const missing = ['COGNIVERSE_RUNTIME_URL', 'COGNIVERSE_API_KEY'].filter(
    (name) => !env[name]?.trim(),
  );
  if (missing.length)
    throw new ConfigError(`Missing required environment: ${missing.join(', ')}.`);
  const port = Number(env.PORT ?? '4000');
  if (!Number.isInteger(port) || port < 1 || port > 65535)
    throw new ConfigError(`PORT must be an integer port number, got ${env.PORT}.`);
  return {
    runtimeUrl: env.COGNIVERSE_RUNTIME_URL!.trim().replace(/\/+$/, ''),
    apiKey: env.COGNIVERSE_API_KEY!.trim(),
    port,
    host: env.HOST?.trim() || '127.0.0.1',
    clientDir,
  };
}
