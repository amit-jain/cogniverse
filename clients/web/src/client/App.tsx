import { CopilotKitProvider } from '@copilotkit/react-core/v2';
import { useEffect, useState } from 'react';
import { AgentWorkspace } from './AgentWorkspace';
import { fetchAgents } from './api';
import { noticeRenderer } from './Notice';
import { OPS_VIEWS } from './ops/views';
import { parseRoute, routeHash } from './route';
import { Sidebar } from './Sidebar';

const THREAD_KEY = 'cogniverse.thread.';

/** The thread this browser last had open with ``agent``, if any. */
function rememberedThread(agent: string): string | undefined {
  try {
    return localStorage.getItem(THREAD_KEY + agent) ?? undefined;
  } catch {
    return undefined;
  }
}

function rememberThread(agent: string, thread: string) {
  try {
    localStorage.setItem(THREAD_KEY + agent, thread);
  } catch {
    // Without storage the thread lives in the address alone.
  }
}

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
  const [threads] = useState(() => new Map<string, string>());
  let threadId: string | undefined;
  if (agentName) {
    threadId =
      (route.kind === 'agent' && route.name === agentName ? route.thread : undefined) ??
      threads.get(agentName) ??
      rememberedThread(agentName) ??
      crypto.randomUUID();
    threads.set(agentName, threadId);
  }
  useEffect(() => {
    if (!agentName || !threadId) return;
    rememberThread(agentName, threadId);
    const hash = routeHash({ kind: 'agent', name: agentName, thread: threadId });
    if (window.location.hash !== hash) window.history.replaceState(null, '', hash);
  }, [agentName, threadId]);
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
    main = (
      <AgentWorkspace
        key={`${agentName}/${threadId}`}
        agentName={agentName}
        threadId={threadId!}
        onNewThread={() => {
          window.location.hash = routeHash({ kind: 'agent', name: agentName, thread: crypto.randomUUID() });
        }}
      />
    );
  }

  return (
    <CopilotKitProvider runtimeUrl="/api/copilotkit" renderActivityMessages={[noticeRenderer]}>
      <div className="shell">
        <Sidebar agents={agents ?? []} route={route.kind === 'agent' ? { kind: 'agent', name: agentName } : route} />
        <main className="main">{main}</main>
      </div>
    </CopilotKitProvider>
  );
}
