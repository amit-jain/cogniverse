import { CopilotChat, useAgent } from '@copilotkit/react-core/v2';
import { useEffect, useState } from 'react';
import { agentLabel } from './api';
import { ResultCards, resultsOf, searchSpanOf } from './ResultCards';

/** The CUSTOM event the runtime emits for each agent progress update. */
export const STATUS_EVENT = 'cogniverse.status';

export function AgentWorkspace({ agentName }: { agentName: string }) {
  const { agent } = useAgent({ agentId: agentName });
  const [status, setStatus] = useState('');

  useEffect(() => {
    const subscription = agent.subscribe({
      onRunStartedEvent: () => setStatus(''),
      onCustomEvent: ({ event }) => {
        if (event.name !== STATUS_EVENT) return;
        const value = event.value as { message?: unknown };
        if (typeof value?.message === 'string') setStatus(value.message);
      },
      onRunFinalized: () => setStatus(''),
      onRunFailed: () => setStatus(''),
    });
    return () => subscription.unsubscribe();
  }, [agent]);

  const results = resultsOf(agent.state);
  const spanId = searchSpanOf(agent.state);
  return (
    <div className="workspace">
      <header className="workspace-header">
        <h1>{agentLabel(agentName)}</h1>
        {status && (
          <p className="status" role="status">
            {status}
          </p>
        )}
      </header>
      <div className="workspace-body">
        <section className="chat">
          <CopilotChat
            agentId={agentName}
            labels={{ chatInputPlaceholder: `Ask ${agentLabel(agentName)}…` }}
          />
        </section>
        {results.length > 0 && (
          <aside className="results" aria-label="Results">
            <ResultCards key={spanId} results={results} spanId={spanId} />
          </aside>
        )}
      </div>
    </div>
  );
}
