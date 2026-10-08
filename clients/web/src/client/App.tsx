import { CopilotKitProvider } from '@copilotkit/react-core/v2';
import { useEffect, useState } from 'react';
import { AgentWorkspace } from './AgentWorkspace';
import { fetchAgents } from './api';
import { OPS_VIEWS } from './ops/views';
import { parseRoute } from './route';
import { Sidebar } from './Sidebar';

function useHashRoute() {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash));
  useEffect(() => {
    const update = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener('hashchange', update);
    return () => window.removeEventListener('hashchange', update);
  }, []);
  return route;
}

export function App() {
  const route = useHashRoute();
  const [agents, setAgents] = useState<string[]>();
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setError('');
    fetchAgents(controller.signal)
      .then(setAgents)
      .catch((e: unknown) => {
        if (!controller.signal.aborted)
          setError(e instanceof Error ? e.message : 'Loading agents failed.');
      });
    return () => controller.abort();
  }, [attempt]);

  const agentName =
    route.kind === 'agent' ? (route.name && agents?.includes(route.name) ? route.name : agents?.[0]) : undefined;
  const opsView = route.kind === 'ops' ? OPS_VIEWS.find((view) => view.id === route.id) : undefined;

  let main;
  if (route.kind === 'ops') {
    main = opsView ? (
      <div className="workspace">
        <header className="workspace-header">
          <h1>{opsView.label}</h1>
        </header>
        <div className="ops-scroll">
          <opsView.component />
        </div>
      </div>
    ) : (
      <div className="notice">There is no view named {route.id}.</div>
    );
  } else if (error) {
    main = (
      <div className="notice error">
        <p>{error}</p>
        <button onClick={() => setAttempt((n) => n + 1)}>Retry</button>
      </div>
    );
  } else if (!agents) {
    main = <div className="notice">Loading agents…</div>;
  } else if (!agentName) {
    main = <div className="notice">The runtime has no agents registered.</div>;
  } else {
    main = <AgentWorkspace key={agentName} agentName={agentName} />;
  }

  return (
    <CopilotKitProvider runtimeUrl="/api/copilotkit">
      <div className="shell">
        <Sidebar agents={agents ?? []} route={route.kind === 'agent' ? { kind: 'agent', name: agentName } : route} />
        <main className="main">{main}</main>
      </div>
    </CopilotKitProvider>
  );
}
