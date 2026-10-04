export interface ResultItem {
  id: string;
  score?: number;
  title?: string;
  snippet?: string;
  /** Segment bounds in seconds, when the hit is a video segment. */
  start?: number;
  end?: number;
}

function text(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined;
}

function number(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/**
 * The search hits in a run's final payload (``state.result.results``), in the
 * runtime's public result shape. The score is the value the set is ranked by:
 * ``rrf_score`` for an ensemble, ``score`` otherwise.
 */
export function resultsOf(state: unknown): ResultItem[] {
  const result = (state as { result?: { results?: unknown } } | undefined)?.result;
  if (!Array.isArray(result?.results)) return [];
  return result.results.flatMap((entry: unknown): ResultItem[] => {
    if (!entry || typeof entry !== 'object') return [];
    const hit = entry as Record<string, unknown>;
    const metadata = (hit.metadata ?? {}) as Record<string, unknown>;
    const temporal = (hit.temporal_info ?? {}) as Record<string, unknown>;
    const id =
      text(hit.id) ?? text(hit.document_id) ?? text(hit.video_id) ?? text(metadata.video_id);
    if (!id) return [];
    return [
      {
        id,
        score: number(hit.rrf_score) ?? number(hit.score),
        title: text(hit.title) ?? text(metadata.title) ?? text(metadata.video_title),
        snippet:
          text(hit.content) ??
          text(hit.text) ??
          text(metadata.audio_transcript) ??
          text(metadata.description),
        start: number(temporal.start_time),
        end: number(temporal.end_time),
      },
    ];
  });
}

/** 75.4 -> "1:15". */
export function clock(seconds: number): string {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

export function ResultCards({ results }: { results: ResultItem[] }) {
  return (
    <ol className="result-list">
      {results.map((item, index) => (
        <li key={`${item.id}-${index}`} className="result-card">
          <div className="result-head">
            <span className="result-title">{item.title ?? item.id}</span>
            {item.score !== undefined && (
              <span className="result-score">{item.score.toFixed(3)}</span>
            )}
          </div>
          {(item.title || item.start !== undefined) && (
            <div className="result-id">
              {item.title && item.id}
              {item.start !== undefined && item.end !== undefined && (
                <span className="result-time">{` ${clock(item.start)}–${clock(item.end)}`}</span>
              )}
            </div>
          )}
          {item.snippet && <p className="result-snippet">{item.snippet}</p>}
        </li>
      ))}
    </ol>
  );
}
