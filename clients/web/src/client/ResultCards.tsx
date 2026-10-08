import { Fragment, useState } from 'react';
import { agentLabel } from './api';
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

function record(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

/** One hit in the runtime's public shape for its agent: a video segment, a
 * document, an image or an audio clip. The score is the value the set is
 * ranked by: ``rrf_score`` for an ensemble, ``score`` or ``relevance_score``
 * otherwise. */
function hitOf(entry: unknown): ResultItem[] {
  const hit = record(entry);
  if (!hit) return [];
  const metadata = record(hit.metadata) ?? {};
  const temporal = record(hit.temporal_info) ?? {};
  const id =
    text(hit.id) ??
    text(hit.document_id) ??
    text(hit.image_id) ??
    text(hit.audio_id) ??
    text(hit.video_id) ??
    text(metadata.video_id);
  if (!id) return [];
  return [
    {
      id,
      ratingId:
        text(hit.document_id) ??
        text(hit.documentid) ??
        text(hit.id) ??
        text(hit.source_id) ??
        text(hit.image_id) ??
        text(hit.audio_id) ??
        text(hit.video_id),
      score: number(hit.rrf_score) ?? number(hit.score) ?? number(hit.relevance_score),
      title: text(hit.title) ?? text(metadata.title) ?? text(metadata.video_title),
      snippet:
        text(hit.content_preview) ??
        text(hit.content) ??
        text(hit.text) ??
        text(metadata.segment_description) ??
        text(hit.transcript) ??
        text(hit.description) ??
        text(metadata.audio_transcript) ??
        text(metadata.description),
      start: number(temporal.start_time),
      end: number(temporal.end_time),
    },
  ];
}

/** The hits of a run's final payload (``state.result.results``). */
export function resultsOf(state: unknown): ResultItem[] {
  const results = record(record(state)?.result)?.results;
  return Array.isArray(results) ? results.flatMap(hitOf) : [];
}

/** The payload of each agent a run's final state holds: the run's own and,
 * for an orchestration, each planned agent's in plan order. */
function payloadsOf(state: unknown): { agent: string; payload: Record<string, unknown> }[] {
  const root = record(state);
  const result = record(root?.result);
  if (!result) return [];
  const agent = text(root?.agent) ?? text(result.agent) ?? '';
  const steps = record(record(result.orchestration_result)?.agent_results) ?? {};
  return [
    { agent, payload: result },
    ...Object.entries(steps).flatMap(([name, payload]) => {
      const step = record(payload);
      return step ? [{ agent: name, payload: step }] : [];
    }),
  ];
}

export interface ResultGroup {
  agent: string;
  /** The span the agent's search recorded its hits under, when it has one. */
  spanId?: string;
  items: ResultItem[];
}

/** The hits of a run's final state, grouped by the agent that found them. */
export function resultGroupsOf(state: unknown): ResultGroup[] {
  return payloadsOf(state).flatMap(({ agent, payload }) => {
    const items = resultsOf({ result: payload });
    return items.length ? [{ agent, spanId: text(payload.span_id), items }] : [];
  });
}

export interface CodingResult {
  agent: string;
  summary?: string;
  files: { path: string; content: string; change?: string }[];
  runs: { command?: string; exitCode?: number; stdout: string; stderr: string }[];
}

/** The code a run wrote and the output of running it, from the coding
 * agent's output, whether it arrives as the dispatcher's envelope
 * (``result.result``), as the streamed output itself, or as one step of an
 * orchestration. */
export function codingOf(state: unknown): CodingResult[] {
  return payloadsOf(state).flatMap(({ agent, payload }) => {
    const output = Array.isArray(payload.code_changes) ? payload : record(payload.result);
    if (!output || !Array.isArray(output.code_changes)) return [];
    const runs = Array.isArray(output.execution_results) ? output.execution_results : [];
    return [
      {
        agent,
        summary: text(output.summary),
        files: output.code_changes.flatMap((entry) => {
          const change = record(entry);
          const path = text(change?.file_path);
          return change && path
            ? [{ path, content: typeof change.content === 'string' ? change.content : '', change: text(change.change_type) }]
            : [];
        }),
        runs: runs.flatMap((entry) => {
          const run = record(entry);
          return run
            ? [
                {
                  command: text(run.command),
                  exitCode: number(run.exit_code),
                  stdout: typeof run.stdout === 'string' ? run.stdout : '',
                  stderr: typeof run.stderr === 'string' ? run.stderr : '',
                },
              ]
            : [];
        }),
      },
    ];
  });
}

/** The search's telemetry span id in a run's final payload, when it has one. */
export function searchSpanOf(state: unknown): string | undefined {
  return text(record(record(state)?.result)?.span_id);
}

/** The labels a reviewer rates a hit with, as the runtime stores them. */
export const RELEVANCE_LABELS = ['Highly Relevant', 'Somewhat Relevant', 'Not Relevant'] as const;

/** 75.4 -> "1:15". */
export function clock(seconds: number): string {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

/** The hits and code of a run's final state, beside the chat. */
export function ResultPanel({ state }: { state: unknown }) {
  const groups = resultGroupsOf(state);
  const coding = codingOf(state);
  if (!groups.length && !coding.length) return null;
  const labelled = groups.length > 1;
  return (
    <aside className="results" aria-label="Results">
      {coding.map((code) => (
        <CodePanel key={code.agent} code={code} />
      ))}
      {groups.map((group) => (
        <Fragment key={`${group.agent}-${group.spanId ?? ''}`}>
          {labelled && <h2 className="result-group">{agentLabel(group.agent)}</h2>}
          <ResultCards results={group.items} spanId={group.spanId} />
        </Fragment>
      ))}
    </aside>
  );
}

function CodePanel({ code }: { code: CodingResult }) {
  return (
    <section className="code-result" aria-label={`Code from ${agentLabel(code.agent)}`}>
      {code.summary && <p className="code-summary">{code.summary}</p>}
      {code.files.map((file) => (
        <figure key={file.path} className="code-file">
          <figcaption>{file.change ? `${file.path} (${file.change})` : file.path}</figcaption>
          <pre>
            <code>{file.content}</code>
          </pre>
        </figure>
      ))}
      {code.runs.map((run, index) => (
        <figure key={index} className="code-run">
          <figcaption>
            {[run.command, run.exitCode === undefined ? undefined : `exit code ${run.exitCode}`]
              .filter(Boolean)
              .join(' — ')}
          </figcaption>
          {run.stdout && <pre aria-label="Output">{run.stdout}</pre>}
          {run.stderr && (
            <pre className="code-stderr" aria-label="Errors">
              {run.stderr}
            </pre>
          )}
        </figure>
      ))}
    </section>
  );
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
