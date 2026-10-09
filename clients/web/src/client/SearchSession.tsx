import { useState } from 'react';
import { agentLabel } from './api';
import { newId } from './ids';
import { messageOf } from './ops/common';
import { errorMessage, runtimeJson, seg } from './ops/http';
import { TENANT_HEADER } from './tenant';
import { parseSse } from './ops/sse';
import {
  MAX_TOP_K,
  annotationExport,
  type AnnotationRecord,
  type SearchSettings,
  type TurnRecord,
} from './session';

/** The agent "Summarize results" runs, grounded in the hits on screen. */
export const SUMMARIZER = 'summarizer_agent';
/** The most hits the runtime grounds one answer in. */
const MAX_GROUNDING_HITS = 50;

/** The conversation's id, its turn count and the search settings every run sends. */
export function SessionBar({
  threadId,
  turns,
  settings,
  onSettings,
}: {
  threadId: string;
  turns: number;
  settings: SearchSettings;
  onSettings: (settings: SearchSettings) => void;
}) {
  return (
    <div className="session-bar" role="group" aria-label="Session">
      <span>
        Session <code title={threadId}>{threadId.slice(0, 8)}</code>
      </span>
      <span>{`${turns} turn${turns === 1 ? '' : 's'}`}</span>
      <label>
        Results per search
        <input
          type="number"
          min={1}
          max={MAX_TOP_K}
          value={settings.topK}
          onChange={(e) => {
            const topK = Number(e.target.value);
            if (Number.isInteger(topK) && topK >= 1 && topK <= MAX_TOP_K) onSettings({ ...settings, topK });
          }}
        />
      </label>
      <label>
        Minimum score
        <input
          type="number"
          min={0}
          step={0.01}
          value={settings.minScore}
          onChange={(e) => {
            const minScore = Number(e.target.value);
            if (Number.isFinite(minScore) && minScore >= 0) onSettings({ ...settings, minScore });
          }}
        />
      </label>
    </div>
  );
}

function time(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour12: false });
}

/** Each run of the conversation: its question, what it found, and when. */
export function History({ turns }: { turns: TurnRecord[] }) {
  const [open, setOpen] = useState(false);
  if (!turns.length) return null;
  return (
    <section className="history" aria-label="History">
      <button type="button" aria-expanded={open} onClick={() => setOpen(!open)}>
        {`History (${turns.length} run${turns.length === 1 ? '' : 's'})`}
      </button>
      {open && (
        <ol>
          {turns.map((turn, index) => (
            <li key={index}>
              <span className="history-query">{turn.query}</span>
              <span className="muted">
                {` — ${turn.results === undefined ? 'no search' : `${turn.results} result${turn.results === 1 ? '' : 's'}`} at ${time(turn.at)}`}
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

interface SummaryRun {
  text: string;
  status: string;
  keyPoints: string[];
  error: string;
  running: boolean;
}

/** One summarizer run for ``tenant`` over ``hits`` on the runtime's AG-UI
 * surface; ``onUpdate`` sees the reply grow, each status, the key points and
 * a failure. */
export async function summarize(
  tenant: string,
  query: string,
  hits: Record<string, unknown>[],
  onUpdate: (update: Partial<SummaryRun>) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`/ui-api/runtime/ag-ui/${SUMMARIZER}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', accept: 'text/event-stream', [TENANT_HEADER]: tenant },
    signal,
    body: JSON.stringify({
      threadId: `summary-${newId()}`,
      runId: newId(),
      state: {},
      messages: [
        { id: newId(), role: 'user', content: `Summarize the search results for '${query}'` },
      ],
      tools: [],
      context: [],
      forwardedProps: { cogniverse: { search_results: hits.slice(0, MAX_GROUNDING_HITS) } },
    }),
  });
  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => null);
    throw new Error(errorMessage(body, response.status));
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  let text = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    const parsed = parseSse(buffer + value);
    buffer = parsed.rest;
    for (const frame of parsed.frames) {
      const event = JSON.parse(frame.data) as Record<string, unknown>;
      if (event.type === 'TEXT_MESSAGE_CONTENT' && typeof event.delta === 'string') {
        text += event.delta;
        onUpdate({ text });
      } else if (event.type === 'CUSTOM' && event.name === 'cogniverse.status') {
        const message = (event.value as { message?: unknown } | null)?.message;
        if (typeof message === 'string') onUpdate({ status: message });
      } else if (event.type === 'STATE_SNAPSHOT') {
        const points = (event.snapshot as { result?: { key_points?: unknown } } | null)?.result?.key_points;
        if (Array.isArray(points))
          onUpdate({ keyPoints: points.filter((point): point is string => typeof point === 'string') });
      } else if (event.type === 'RUN_ERROR') {
        throw new Error(typeof event.message === 'string' ? event.message : 'The summary failed.');
      }
    }
  }
}

/** "Summarize results": the summarizer's streamed summary of the hits on
 * screen and its key points. */
export function SummarizeResults({
  tenant,
  query,
  hits,
}: {
  tenant: string;
  query: string;
  hits: Record<string, unknown>[];
}) {
  const [run, setRun] = useState<SummaryRun>();
  const start = () => {
    let current: SummaryRun = { text: '', status: '', keyPoints: [], error: '', running: true };
    setRun(current);
    const update = (change: Partial<SummaryRun>) => {
      current = { ...current, ...change };
      setRun(current);
    };
    summarize(tenant, query, hits, update)
      .catch((e: unknown) => update({ error: messageOf(e) }))
      .finally(() => update({ running: false, status: '' }));
  };
  return (
    <section className="summary" aria-label="Summary">
      <button type="button" disabled={run?.running} onClick={start}>
        {run?.running ? 'Summarizing…' : `Summarize results with ${agentLabel(SUMMARIZER)}`}
      </button>
      {run?.status && <p className="status">{run.status}</p>}
      {run?.text && <p className="summary-text">{run.text}</p>}
      {run && run.keyPoints.length > 0 && (
        <>
          <h3>Key points</h3>
          <ul aria-label="Summary key points">
            {run.keyPoints.map((point, index) => (
              <li key={index}>{point}</li>
            ))}
          </ul>
        </>
      )}
      {run?.error && (
        <p className="alert error" role="alert">
          {`The summary failed: ${run.error}`}
        </p>
      )}
    </section>
  );
}

/** The ratings stored in this conversation, and a download of them. */
export function Annotations({
  tenant,
  threadId,
  agent,
  turns,
  annotations,
}: {
  tenant: string;
  threadId: string;
  agent: string;
  turns: TurnRecord[];
  annotations: AnnotationRecord[];
}) {
  if (!annotations.length) return null;
  const download = () => {
    const now = new Date().toISOString();
    const blob = new Blob([JSON.stringify(annotationExport(tenant, threadId, agent, turns, annotations, now), null, 2)], {
      type: 'application/json',
    });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `search_annotations_${threadId.slice(0, 8)}_${now.replace(/[:.]/g, '-')}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  };
  return (
    <section className="annotations" aria-label="Annotations">
      <p>{`${annotations.length} rating${annotations.length === 1 ? '' : 's'} saved in this conversation.`}</p>
      <button type="button" onClick={download}>
        Export annotations
      </button>
    </section>
  );
}

const OUTCOMES = [
  ['success', 'Success'],
  ['partial', 'Partial'],
  ['failure', 'Failure'],
] as const;

/** A reviewer's verdict on the whole conversation, stored on its searches. */
export function SessionEvaluation({
  tenant,
  threadId,
  spanIds,
}: {
  tenant: string;
  threadId: string;
  spanIds: string[];
}) {
  const [outcome, setOutcome] = useState('');
  const [score, setScore] = useState(0.5);
  const [saved, setSaved] = useState('');
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  if (!spanIds.length) return null;
  const save = async () => {
    setPending(true);
    setError('');
    setSaved('');
    try {
      const stored = await runtimeJson<{ outcome: string; score: number; span_ids: string[] }>(
        `/ag-ui/threads/${seg(threadId)}/evaluation`,
        { method: 'POST', body: { outcome, score, span_ids: spanIds }, tenant },
      );
      const label = OUTCOMES.find(([value]) => value === stored.outcome)?.[1] ?? stored.outcome;
      const searches = stored.span_ids.length;
      setSaved(`Saved: ${label} (${stored.score.toFixed(1)}) on ${searches} search${searches === 1 ? '' : 'es'}.`);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setPending(false);
    }
  };
  return (
    <form
      className="evaluation"
      aria-label="Evaluate this conversation"
      onSubmit={(e) => {
        e.preventDefault();
        if (outcome) void save();
      }}
    >
      <h2 className="results-heading">Evaluate this conversation</h2>
      <label>
        Outcome
        <select value={outcome} onChange={(e) => setOutcome(e.target.value)}>
          <option value="">Not rated</option>
          {OUTCOMES.map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <label>
        Quality
        <input
          type="range"
          min={0}
          max={1}
          step={0.1}
          value={score}
          onChange={(e) => setScore(Number(e.target.value))}
        />
        <span>{score.toFixed(1)}</span>
      </label>
      <button type="submit" disabled={!outcome || pending}>
        {pending ? 'Saving…' : 'Save evaluation'}
      </button>
      {saved && (
        <p className="alert ok" role="status">
          {saved}
        </p>
      )}
      {error && (
        <p className="alert error" role="alert">
          {error}
        </p>
      )}
    </form>
  );
}
