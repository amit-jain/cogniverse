import type { ActivityMessage } from '@ag-ui/client';
import { CopilotChat, useAgent, useCopilotKit } from '@copilotkit/react-core/v2';
import { useEffect, useRef, useState } from 'react';
import { agentLabel } from './api';
import { NOTICE_ACTIVITY, type Notice } from './Notice';
import { ResultPanel } from './ResultCards';

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

export function AgentWorkspace({
  agentName,
  threadId,
  onNewThread,
}: {
  agentName: string;
  threadId: string;
  onNewThread: () => void;
}) {
  const { agent } = useAgent({ agentId: agentName });
  const { copilotkit } = useCopilotKit();
  const [status, setStatus] = useState('');
  const failure = useRef<{ message?: string; code?: string }>(undefined);
  const stopped = useRef(false);

  useEffect(() => {
    const subscription = agent.subscribe({
      onRunStartedEvent: () => setStatus(''),
      onRunInitialized: () => {
        failure.current = undefined;
        stopped.current = false;
      },
      onCustomEvent: ({ event }) => {
        if (event.name !== STATUS_EVENT) return;
        const value = event.value as { message?: unknown };
        if (typeof value?.message === 'string') setStatus(value.message);
      },
      onRunErrorEvent: ({ event }) => {
        failure.current = { message: event.message, code: event.code };
      },
      onRunFailed: ({ error }) => {
        failure.current ??= { message: error.message };
      },
      onRunFinalized: ({ messages }) => {
        setStatus('');
        const notice = runNotice(failure.current, stopped.current);
        failure.current = undefined;
        stopped.current = false;
        if (!notice) return;
        return { messages: [...messages, noticeMessage(crypto.randomUUID(), notice)] };
      },
    });
    return () => subscription.unsubscribe();
  }, [agent]);

  const stop = () => {
    stopped.current = true;
    copilotkit.stopAgent({ agent });
  };

  return (
    <div className="workspace">
      <header className="workspace-header">
        <h1>{agentLabel(agentName)}</h1>
        {status && (
          <p className="status" role="status">
            {status}
          </p>
        )}
        <button className="new-thread" onClick={onNewThread}>
          New conversation
        </button>
      </header>
      <div className="workspace-body">
        <section className="chat">
          <CopilotChat
            agentId={agentName}
            threadId={threadId}
            onStop={agent.isRunning ? stop : undefined}
            labels={{ chatInputPlaceholder: `Ask ${agentLabel(agentName)}…` }}
          />
        </section>
        <ResultPanel state={agent.state} />
      </div>
    </div>
  );
}
