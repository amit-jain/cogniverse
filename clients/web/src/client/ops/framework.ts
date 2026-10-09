/** Pure helpers of the optimization framework view. */

/** Whole-unit age of a run start: ``not started`` without one, ``unknown``
 * when it does not parse, never negative. */
export function formatRunAge(startedAt: string | null, now: Date): string {
  if (!startedAt) return 'not started';
  const started = Date.parse(/[zZ]|[+-]\d\d:\d\d$/.test(startedAt) ? startedAt : `${startedAt}Z`);
  if (Number.isNaN(started)) return 'unknown';
  const minutes = Math.floor(Math.max(now.getTime() - started, 0) / 60_000);
  if (minutes < 60) return `${minutes}m ago`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)}h ago`;
  return `${Math.floor(minutes / 1440)}d ago`;
}

export type Band = { label: string; tone: 'ok' | 'warn' | 'error' };

/** The review band of an item below the auto-approval threshold. */
export function confidenceBand(confidence: number): Band {
  if (confidence >= 0.75) return { label: 'Medium-low', tone: 'ok' };
  if (confidence >= 0.6) return { label: 'Low', tone: 'warn' };
  return { label: 'Very low', tone: 'error' };
}

/** The share of items a threshold sends to review, as the whole percent a
 * reviewer expects: 0.85 -> "~15%". */
export function expectedReviewRate(threshold: number): string {
  return `~${Math.trunc(Math.round((1 - threshold) * 1e9) / 1e7)}%`;
}

export type RatingKind = 'thumbs' | 'stars' | 'relevance';

export const RATING_KINDS: { kind: RatingKind; label: string }[] = [
  { kind: 'thumbs', label: 'Thumbs up/down' },
  { kind: 'stars', label: 'Star rating (1-5)' },
  { kind: 'relevance', label: 'Relevance score (0-1)' },
];

/** What a saved rating reads as: "Thumbs up", "4 stars", "0.5 relevance". */
export function ratingText(kind: RatingKind, value: number): string {
  if (kind === 'thumbs') return value ? 'Thumbs up' : 'Thumbs down';
  if (kind === 'stars') return `${value} star${value === 1 ? '' : 's'}`;
  return `${value.toFixed(1)} relevance`;
}

export const ANNOTATION_PAGE_SIZE = 10;

/** The 1-based page numbers of ``total`` items. */
export function pageNumbers(total: number, size = ANNOTATION_PAGE_SIZE): number[] {
  return Array.from({ length: Math.max(1, Math.ceil(total / size)) }, (_, index) => index + 1);
}

export function pageOf<T>(items: T[], page: number, size = ANNOTATION_PAGE_SIZE): T[] {
  return items.slice((page - 1) * size, page * size);
}

export interface GoldenEntry {
  expected_videos: string[];
  relevance_scores: Record<string, number>;
  avg_relevance: number;
  profile: string;
  timestamp: string;
}

/** The first ``limit`` golden queries as table rows. */
export function goldenSample(dataset: Record<string, GoldenEntry>, limit = 10) {
  return Object.entries(dataset)
    .slice(0, limit)
    .map(([query, entry]) => ({
      query,
      expectedVideos: entry.expected_videos.length,
      avgRelevance: entry.avg_relevance,
    }));
}

/** ``YYYYMMDD`` of ``date`` in UTC. */
export function compactDate(date: Date): string {
  return date.toISOString().slice(0, 10).replaceAll('-', '');
}

/** ``YYYYMMDD_HHMMSS`` of ``date`` in UTC. */
export function compactTimestamp(date: Date): string {
  const iso = date.toISOString();
  return `${compactDate(date)}_${iso.slice(11, 19).replaceAll(':', '')}`;
}

/** A tenant ID made safe for a file name. */
export function fileSafe(text: string): string {
  return text.replace(/[^A-Za-z0-9_-]/g, '_');
}

/** Saves ``body`` as pretty JSON under ``filename``. */
export function downloadJson(filename: string, body: unknown): void {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([JSON.stringify(body, null, 2)], { type: 'application/json' }));
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

/** The columns of a sample of generated examples: every key any of them
 * carries, in first-seen order. */
export function exampleColumns(rows: Record<string, unknown>[]): string[] {
  const columns: string[] = [];
  for (const row of rows) for (const key of Object.keys(row)) if (!columns.includes(key)) columns.push(key);
  return columns;
}

/** A cell of a generated example: text as is, anything else as JSON. */
export function cellText(value: unknown): string {
  if (value === null || value === undefined) return '';
  return typeof value === 'string' ? value : JSON.stringify(value);
}

/** An entity as the reviewer reads it: ``text (TYPE)``. */
export function entityText(entity: unknown): string {
  if (entity && typeof entity === 'object' && 'text' in entity) {
    const { text, type } = entity as { text: unknown; type?: unknown };
    return type ? `${String(text)} (${String(type)})` : String(text);
  }
  return cellText(entity);
}

/** Milliseconds as "1250ms", "—" when unknown. */
export function millis(value: number | null): string {
  return value === null ? '—' : `${Math.round(value)}ms`;
}

/** The module optimization modes and what each trains. */
export const MODULES = [
  {
    mode: 'routing',
    label: 'Routing',
    description: "The routing gateway's thresholds, its entity extractor and its profile selector.",
  },
  { mode: 'workflow', label: 'Workflow', description: 'Orchestration templates and agent performance profiles.' },
  { mode: 'unified', label: 'Unified', description: 'Routing, then workflow.' },
];
