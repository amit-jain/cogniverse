export class RuntimeRequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
    /** The response's parsed JSON body. */
    readonly body: unknown = null,
  ) {
    super(message);
  }
}

/** The typed fields a runtime failure carries beside its message (its
 * ``error`` code and ``failure`` type) and its HTTP status, as one line;
 * ``null`` for an error that is not a runtime answer. */
export function failureDetail(error: unknown): string | null {
  if (!(error instanceof RuntimeRequestError)) return null;
  const detail = (error.body as { detail?: unknown } | null)?.detail as
    | { error?: unknown; failure?: unknown }
    | null
    | undefined;
  const parts = [
    typeof detail?.error === 'string' ? `error ${detail.error}` : null,
    typeof detail?.failure === 'string' ? `failure ${detail.failure}` : null,
    `HTTP ${error.status}`,
  ];
  return parts.filter(Boolean).join(', ');
}

/**
 * The human-readable reason in a failed runtime or web-server response:
 * the web server's ``error``, an OpenAI-style ``error.message``, FastAPI's
 * string ``detail``, a structured ``detail.message`` followed by its
 * ``detail.errors``, or each validation error as ``field: message``.
 */
export function errorMessage(body: unknown, status: number): string {
  const value = body as { error?: unknown; detail?: unknown } | null;
  if (typeof value?.error === 'string') return value.error;
  const envelope = value?.error as { message?: unknown } | null | undefined;
  if (typeof envelope?.message === 'string') return envelope.message;
  const detail = value?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const parts = detail.flatMap((item) => {
      const entry = item as { loc?: unknown[]; msg?: unknown };
      if (typeof entry?.msg !== 'string') return [];
      const field = (entry.loc ?? []).filter((part) => part !== 'body').join('.');
      return [field ? `${field}: ${entry.msg}` : entry.msg];
    });
    if (parts.length) return parts.join('; ');
  }
  const structured = detail as { message?: unknown; errors?: unknown } | null;
  if (typeof structured?.message === 'string') {
    const errors = Array.isArray(structured.errors)
      ? structured.errors.filter((item): item is string => typeof item === 'string')
      : [];
    return errors.length ? `${structured.message}: ${errors.join('; ')}` : structured.message;
  }
  return `The runtime answered HTTP ${status}.`;
}

/**
 * Calls a runtime route through the web server and returns its JSON body.
 * A ``FormData`` body goes as multipart; any other body as JSON.
 */
export async function runtimeJson<T>(
  path: string,
  init: { method?: string; body?: unknown; signal?: AbortSignal } = {},
): Promise<T> {
  const form = init.body instanceof FormData;
  const response = await fetch(`/api/runtime${path}`, {
    method: init.method ?? 'GET',
    headers: init.body === undefined || form ? undefined : { 'content-type': 'application/json' },
    body: init.body === undefined ? undefined : form ? (init.body as FormData) : JSON.stringify(init.body),
    signal: init.signal,
  });
  const text = await response.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    throw new RuntimeRequestError(
      `The runtime answered HTTP ${response.status} with a body that is not JSON.`,
      response.status,
    );
  }
  if (!response.ok) throw new RuntimeRequestError(errorMessage(body, response.status), response.status, body);
  return body as T;
}

/** Path segment for an id that may hold characters like ``:``. */
export const seg = encodeURIComponent;
