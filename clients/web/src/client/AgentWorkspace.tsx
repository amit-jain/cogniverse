import type { ActivityMessage } from '@ag-ui/client';
import { CopilotChat, useAgent, useCopilotKit } from '@copilotkit/react-core/v2';
import { useEffect, useRef, useState } from 'react';
import { agentLabel } from './api';
import { NOTICE_ACTIVITY, type Notice } from './Notice';
import { ResultPanel, resultGroupsOf, type Rating } from './ResultCards';
import { Annotations, History, SessionBar, SessionEvaluation, SUMMARIZER, SummarizeResults } from './SearchSession';
import {
  DEFAULT_SETTINGS,
  loadAnnotations,
  loadTurns,
  messageText,
  saveAnnotations,
  saveTurns,
  spanIdsOf,
  withAnnotation,
  type AnnotationRecord,
  type SearchSettings,
  type TurnRecord,
} from './session';

/** The CUSTOM event the runtime emits for each agent progress update. */
export const STATUS_EVENT = 'cogniverse.status';

/** The codes a run reports when it was aborted rather than failed. */
const ABORT_CODES = new Set(['abort', 'STOPPED']);

/** What the conversation says about a run that did not finish: why it
 * failed, or that it was cancelled. */
export function runNotice(failure: { message?: string; code?: string } | undefined, stopped: boolean): Notice | undefined {
  if (stopped) return { tone: 'cancelled', text: 'You cancelled this run.' };
  if (!failure) return undefined;
  if (failure.code && ABORT_CODES.has(failure.code)) return { tone: 'cancelled', text: 'The run was cancelled.' };
  return { tone: 'error', text: `The run failed: ${failure.message || 'no reason given'}` };
}

/** A notice as a message of the conversation; activity messages are never
 * sent to an agent. */
export function noticeMessage(id: string, notice: Notice): ActivityMessage {
  return { id, role: 'activity', activityType: NOTICE_ACTIVITY, content: { ...notice } };
}

/** "1 message", "3 messages": the user and assistant turns of a conversation. */
export function messageCount(messages: { role: string }[]): string {
  const count = messages.filter((message) => message.role === 'user' || message.role === 'assistant').length;
  return `${count} ${count === 1 ? 'message' : 'messages'}`;
}

/** The question of the conversation's latest user message. */
function lastQuery(messages: readonly { role: string; content?: unknown }[]): string {
  for (let index = messages.length - 1; index >= 0; index--)
    if (messages[index].role === 'user') return messageText(messages[index].content);
  return '';
}

/** The record of a finished run: its question and what its searches found. */
export function turnRecord(query: string, snapshot: unknown, at: string): TurnRecord {
  const groups = snapshot === undefined ? [] : resultGroupsOf(snapshot);
  const searched = groups.length > 0;
  const first = groups[0]?.facts;
  return {
    query,
    at,
    results: searched ? groups.reduce((total, group) => total + group.items.length, 0) : undefined,
    spanIds: groups.flatMap((group) => (group.spanId ? [group.spanId] : [])),
    profile: first?.profile ?? (first?.profiles.length ? first.profiles.join(', ') : undefined),
  };
}

/** What a status event shows beside its message. */
interface Progress {
  message: string;
  themes: string[];
  draft: string;
}

const NO_PROGRESS: Progress = { message: '', themes: [], draft: '' };

export function AgentWorkspace({
  tenant,
  agentName,
  threadId,
  onNewThread,
  agents = [],
  settings = DEFAULT_SETTINGS,
  onSettings = () => undefined,
}: {
  tenant: string;
  agentName: string;
  threadId: string;
  onNewThread: () => void;
  /** Every registered agent; the summarizer is offered when it is one. */
  agents?: string[];
  settings?: SearchSettings;
  onSettings?: (settings: SearchSettings) => void;
}) {
  const { agent } = useAgent({ agentId: agentName });
  const { copilotkit } = useCopilotKit();
  const [progress, setProgress] = useState<Progress>(NO_PROGRESS);
  const [turns, setTurns] = useState<TurnRecord[]>(() => loadTurns(threadId));
  const [annotations, setAnnotations] = useState<AnnotationRecord[]>(() => loadAnnotations(threadId));
  const [latencyMs, setLatencyMs] = useState<number>();
  const failure = useRef<{ message?: string; code?: string }>(undefined);
  const stopped = useRef(false);
  const startedAt = useRef(0);
  const snapshot = useRef<unknown>(undefined);
  const restoring = useRef(false);

  useEffect(() => {
    const subscription = agent.subscribe({
      onRunStartedEvent: ({ event }) => {
        setProgress(NO_PROGRESS);
        restoring.current = event.runId.startsWith('restore-');
      },
      onRunInitialized: () => {
        failure.current = undefined;
        stopped.current = false;
        snapshot.current = undefined;
        startedAt.current = performance.now();
      },
      onCustomEvent: ({ event }) => {
        if (event.name !== STATUS_EVENT) return;
        const value = event.value as { message?: unknown; themes?: unknown; summary?: unknown };
        if (typeof value?.message !== 'string') return;
        const message = value.message;
        setProgress((previous) => ({
          message,
          themes: Array.isArray(value.themes)
            ? value.themes.filter((theme): theme is string => typeof theme === 'string')
            : previous.themes,
          draft: typeof value.summary === 'string' ? value.summary : previous.draft,
        }));
      },
      onStateSnapshotEvent: ({ event }) => {
        snapshot.current = event.snapshot;
      },
      onRunErrorEvent: ({ event }) => {
        failure.current = { message: event.message, code: event.code };
      },
      onRunFailed: ({ error }) => {
        failure.current ??= { message: error.message };
      },
      onRunFinalized: ({ messages }) => {
        setProgress(NO_PROGRESS);
        // A restored conversation is not a run of this page.
        if (!restoring.current) {
          setLatencyMs(performance.now() - startedAt.current);
          const record = turnRecord(lastQuery(messages), snapshot.current, new Date().toISOString());
          setTurns((previous) => {
            const next = [...previous, record];
            saveTurns(threadId, next);
            return next;
          });
        }
        restoring.current = false;
        const notice = runNotice(failure.current, stopped.current);
        failure.current = undefined;
        stopped.current = false;
        if (!notice) return;
        return { messages: [...messages, noticeMessage(crypto.randomUUID(), notice)] };
      },
    });
    return () => subscription.unsubscribe();
  }, [agent, threadId]);

  const stop = () => {
    stopped.current = true;
    copilotkit.stopAgent({ agent });
  };

  const rated = (rating: Rating) =>
    setAnnotations((previous) => {
      const next = withAnnotation(previous, {
        ...rating,
        query: lastQuery(agent.messages),
        at: new Date().toISOString(),
      });
      saveAnnotations(threadId, next);
      return next;
    });

  const query = lastQuery(agent.messages);
  const groups = resultGroupsOf(agent.state);
  const hits = groups.flatMap((group) => group.hits);
  const userTurns = agent.messages.filter((message) => message.role === 'user').length;

  return (
    <div className="workspace">
      <header className="workspace-header">
        <h1>{agentLabel(agentName)}</h1>
        <p className="workspace-facts" aria-label="Conversation">
          {tenant} · {messageCount(agent.messages)}
        </p>
        {progress.message && (
          <p className="status" role="status">
            {progress.message}
          </p>
        )}
        <button className="new-thread" onClick={onNewThread}>
          New conversation
        </button>
      </header>
      <SessionBar threadId={threadId} turns={userTurns} settings={settings} onSettings={onSettings} />
      {(progress.themes.length > 0 || progress.draft) && (
        <div className="progress-detail" aria-label="Progress">
          {progress.themes.length > 0 && <p className="status">{`Themes: ${progress.themes.slice(0, 3).join(', ')}`}</p>}
          {progress.draft && <p className="draft-summary">{progress.draft}</p>}
        </div>
      )}
      <div className="workspace-body">
        <section className="chat">
          <CopilotChat
            agentId={agentName}
            threadId={threadId}
            onStop={agent.isRunning ? stop : undefined}
            labels={{ chatInputPlaceholder: `Ask ${agentLabel(agentName)}…` }}
          />
        </section>
        <ResultPanel
          state={agent.state}
          tenant={tenant}
          run={{ query, latencyMs, minScore: settings.minScore }}
          onRated={rated}
        >
          {hits.length > 0 && agents.includes(SUMMARIZER) && agentName !== SUMMARIZER && (
            <SummarizeResults key={`${threadId}-${turns.length}`} query={query} hits={hits} />
          )}
          <Annotations threadId={threadId} agent={agentName} turns={turns} annotations={annotations} />
          <SessionEvaluation threadId={threadId} spanIds={spanIdsOf(turns)} />
          <History turns={turns} />
        </ResultPanel>
      </div>
    </div>
  );
}
