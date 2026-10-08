/** The window the Analytics view reads. */
export const PRESETS = {
  'Last 15 minutes': 15 * 60_000,
  'Last hour': 3_600_000,
  'Last 6 hours': 6 * 3_600_000,
  'Last day': 24 * 3_600_000,
  'Last week': 7 * 24 * 3_600_000,
} as const;
export type Preset = keyof typeof PRESETS;
export const DEFAULT_PRESET: Preset = 'Last 6 hours';
/** The longest window the runtime reads. */
export const MAX_WINDOW_MS = 30 * 24 * 3_600_000;

export type TimeRange = { preset: Preset } | { start: string; end: string };

/** ``YYYY-MM-DDTHH:MM`` (a datetime-local value) read as UTC. */
export function utcInput(value: string): Date {
  return new Date(`${value}:00Z`);
}

/** A ``Date`` as a UTC datetime-local value. */
export function toUtcInput(date: Date): string {
  return date.toISOString().slice(0, 16);
}

/**
 * The ``start``/``end`` query of ``range`` at ``now``.
 *
 * @throws Error naming what is wrong with a custom range.
 */
export function windowQuery(range: TimeRange, now: Date = new Date()): { start: string; end: string } {
  if ('preset' in range)
    return { start: new Date(now.getTime() - PRESETS[range.preset]).toISOString(), end: now.toISOString() };
  const start = utcInput(range.start);
  const end = utcInput(range.end);
  const valid = (value: string, date: Date) => /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(value) && !Number.isNaN(date.getTime());
  if (!valid(range.start, start) || !valid(range.end, end))
    throw new Error('Give both the start and the end of the custom range.');
  if (start >= end) throw new Error('The custom range must start before it ends.');
  if (end.getTime() - start.getTime() > MAX_WINDOW_MS) throw new Error('The custom range may span at most 30 days.');
  return { start: start.toISOString(), end: end.toISOString() };
}

/** "Last 6 hours" or "2026-10-01 00:00 to 2026-10-02 00:00 UTC". */
export function describeRange(range: TimeRange): string {
  if ('preset' in range) return range.preset;
  return `${range.start.replace('T', ' ')} to ${range.end.replace('T', ' ')} UTC`;
}
