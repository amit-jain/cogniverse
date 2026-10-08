export type JsonObject = Record<string, unknown>;

/**
 * ``text`` parsed as a JSON object, ``undefined`` when it is blank, or an
 * error naming ``label`` when it is not a JSON object.
 */
export function parseJsonObject(label: string, text: string): JsonObject | undefined {
  if (!text.trim()) return undefined;
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error(`${label} is not valid JSON.`);
  }
  if (value === null || typeof value !== 'object' || Array.isArray(value))
    throw new Error(`${label} must be a JSON object.`);
  return value as JsonObject;
}

/** ``value`` as indented JSON for a textarea; blank for null. */
export function jsonText(value: unknown): string {
  return value === null || value === undefined ? '' : JSON.stringify(value, null, 2);
}

/** Whether two JSON values are equal, ignoring key order. */
export function sameJson(a: unknown, b: unknown): boolean {
  return JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object')
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonical((value as JsonObject)[key])]),
    );
  return value;
}
