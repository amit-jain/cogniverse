import { useState } from 'react';
import { messageOf } from './ops/common';
import { runtimeJson } from './ops/http';

export interface ResultItem {
  id: string;
  /**
   * The id a relevance rating names: the one the search span records the hit
   * under, and that the triplet miner reads back first.
   */
  ratingId?: string;
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
        ratingId:
          text(hit.document_id) ??
          text(hit.documentid) ??
          text(hit.id) ??
          text(hit.source_id) ??
          text(hit.video_id),
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

/** The search's telemetry span id in a run's final payload, when it has one. */
export function searchSpanOf(state: unknown): string | undefined {
  const result = (state as { result?: { span_id?: unknown } } | undefined)?.result;
  return text(result?.span_id);
}

/** The labels a reviewer rates a hit with, as the runtime stores them. */
export const RELEVANCE_LABELS = ['Highly Relevant', 'Somewhat Relevant', 'Not Relevant'] as const;

/** 75.4 -> "1:15". */
export function clock(seconds: number): string {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

export function ResultCards({ results, spanId }: { results: ResultItem[]; spanId?: string }) {
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
          {spanId && item.ratingId && <Relevance spanId={spanId} resultId={item.ratingId} />}
        </li>
      ))}
    </ol>
  );
}

function Relevance({ spanId, resultId }: { spanId: string; resultId: string }) {
  const [rated, setRated] = useState<string>();
  const [pending, setPending] = useState<string>();
  const [error, setError] = useState('');
  const rate = async (relevance: string) => {
    setPending(relevance);
    setError('');
    try {
      const stored = await runtimeJson<{ relevance: string }>('/ag-ui/results/relevance', {
        method: 'POST',
        body: { span_id: spanId, result_id: resultId, relevance },
      });
      setRated(stored.relevance);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setPending(undefined);
    }
  };
  return (
    <div className="relevance" role="group" aria-label={`Relevance of ${resultId}`}>
      {RELEVANCE_LABELS.map((label) => (
        <button
          key={label}
          type="button"
          aria-pressed={rated === label}
          disabled={pending !== undefined}
          onClick={() => rate(label)}
        >
          {pending === label ? 'Saving…' : label}
        </button>
      ))}
      {error && (
        <p className="relevance-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
