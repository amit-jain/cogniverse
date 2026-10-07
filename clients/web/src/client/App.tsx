import { CopilotKitProvider } from '@copilotkit/react-core/v2';
import { useEffect, useState } from 'react';
import { AgentList } from './AgentList';
import { AgentWorkspace } from './AgentWorkspace';
import { fetchAgents } from './api';

export function App() {
  const [agents, setAgents] = useState<string[]>();
  const [selected, setSelected] = useState<string>();
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setError('');
    fetchAgents(controller.signal)
      .then((names) => {
        setAgents(names);
        setSelected((current) =>
          current && names.includes(current) ? current : names[0],
        );
      })
      .catch((e: unknown) => {
        if (!controller.signal.aborted)
          setError(e instanceof Error ? e.message : 'Loading agents failed.');
      });
    return () => controller.abort();
  }, [attempt]);

  return (
    <CopilotKitProvider runtimeUrl="/api/copilotkit">
      <div className="shell">
        <AgentList
          agents={agents ?? []}
          selected={selected}
          onSelect={setSelected}
        />
        <main className="main">
          {error ? (
            <div className="notice error">
              <p>{error}</p>
              <button onClick={() => setAttempt((n) => n + 1)}>Retry</button>
            </div>
          ) : !agents ? (
            <div className="notice">Loading agents…</div>
          ) : !selected ? (
            <div className="notice">The runtime has no agents registered.</div>
          ) : (
            <AgentWorkspace key={selected} agentName={selected} />
          )}
        </main>
      </div>
    </CopilotKitProvider>
  );
}
