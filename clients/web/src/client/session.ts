/** What this browser remembers about one conversation (a tenant's thread): each run's
 * question, when it finished, how many hits it found and the search spans
 * they were recorded under, and the relevance ratings stored for its hits.
 * The turns themselves are the runtime's (``GET /ag-ui/threads/{id}``). */

export interface TurnRecord {
  query: string;
  /** ISO time the run finished. */
  at: string;
  /** Hits the run found, when it searched. */
  results?: number;
  /** Spans the run's searches recorded their hits under. */
  spanIds: string[];
  profile?: string;
}

export interface AnnotationRecord {
  query: string;
  spanId: string;
  resultId: string;
  relevance: string;
  score: number;
  /** ISO time the rating was stored. */
  at: string;
}

export interface SearchSettings {
  /** How many hits a search returns. */
  topK: number;
  /** Hits scoring below this are hidden. */
  minScore: number;
}

export const DEFAULT_SETTINGS: SearchSettings = { topK: 10, minScore: 0 };
export const MAX_TOP_K = 20;

const TURNS_KEY = 'cogniverse.turns.';
const ANNOTATIONS_KEY = 'cogniverse.annotations.';
const SETTINGS_KEY = 'cogniverse.search-settings';

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Without storage the records last as long as the page.
  }
}

export const loadTurns = (tenant: string, thread: string) =>
  read<TurnRecord[]>(`${TURNS_KEY}${tenant}/${thread}`, []);
export const saveTurns = (tenant: string, thread: string, turns: TurnRecord[]) =>
  write(`${TURNS_KEY}${tenant}/${thread}`, turns);
export const loadAnnotations = (tenant: string, thread: string) =>
  read<AnnotationRecord[]>(`${ANNOTATIONS_KEY}${tenant}/${thread}`, []);
export const saveAnnotations = (tenant: string, thread: string, annotations: AnnotationRecord[]) =>
  write(`${ANNOTATIONS_KEY}${tenant}/${thread}`, annotations);

/** The stored settings, each kept only when it is a valid value. */
export function loadSettings(): SearchSettings {
  const stored = read<Partial<SearchSettings>>(SETTINGS_KEY, {});
  return {
    topK:
      Number.isInteger(stored.topK) && stored.topK! >= 1 && stored.topK! <= MAX_TOP_K
        ? stored.topK!
        : DEFAULT_SETTINGS.topK,
    minScore:
      typeof stored.minScore === 'number' && Number.isFinite(stored.minScore) && stored.minScore >= 0
        ? stored.minScore
        : DEFAULT_SETTINGS.minScore,
  };
}

export const saveSettings = (settings: SearchSettings) => write(SETTINGS_KEY, settings);

/** A rating replaces an earlier one of the same hit of the same search. */
export function withAnnotation(annotations: AnnotationRecord[], added: AnnotationRecord): AnnotationRecord[] {
  return [
    ...annotations.filter((entry) => !(entry.spanId === added.spanId && entry.resultId === added.resultId)),
    added,
  ];
}

/** Every search span of a conversation, in the order its runs found them. */
export function spanIdsOf(turns: TurnRecord[]): string[] {
  return [...new Set(turns.flatMap((turn) => turn.spanIds))];
}

/** The ratings of a tenant's conversation as the file "Export annotations" saves. */
export function annotationExport(
  tenant: string,
  thread: string,
  agent: string,
  turns: TurnRecord[],
  annotations: AnnotationRecord[],
  exportedAt: string,
) {
  const last = turns[turns.length - 1];
  return {
    search_session: {
      tenant_id: tenant,
      thread_id: thread,
      agent,
      query: last?.query ?? null,
      profile: last?.profile ?? null,
      exported_at: exportedAt,
    },
    annotations: annotations.map((entry) => ({
      query: entry.query,
      span_id: entry.spanId,
      result_id: entry.resultId,
      relevance: entry.relevance,
      score: entry.score,
      rated_at: entry.at,
    })),
  };
}

/** The text of a chat message's content: a string, or its text parts. */
export function messageText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((part) => {
      const entry = part as { type?: unknown; text?: unknown } | null;
      return entry?.type === 'text' && typeof entry.text === 'string' ? [entry.text] : [];
    })
    .join(' ');
}
