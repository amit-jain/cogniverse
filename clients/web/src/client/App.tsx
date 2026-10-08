import { CopilotKitProvider } from '@copilotkit/react-core/v2';
import { useEffect, useMemo, useState } from 'react';
import { AgentWorkspace } from './AgentWorkspace';
import { fetchAgents } from './api';
import { noticeRenderer } from './Notice';
import { OPS_VIEWS } from './ops/views';
import { parseRoute, routeHash } from './route';
import { Sidebar } from './Sidebar';
import { loadSettings, saveSettings, type SearchSettings } from './session';

/** The agent a conversation opens with when the address names none: the
 * gateway, which routes a question to the agents that answer it. */
export const DEFAULT_AGENT = 'gateway_agent';

/** The agent the address names when it is registered, else the default. */
export function chosenAgent(named: string | undefined, agents: string[]): string | undefined {
  if (named && agents.includes(named)) return named;
  return agents.includes(DEFAULT_AGENT) ? DEFAULT_AGENT : agents[0];
}

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

  const [settings, setSettings] = useState<SearchSettings>(loadSettings);
  // Every run carries the search settings as its per-run parameters.
  const properties = useMemo(() => ({ cogniverse: { top_k: settings.topK } }), [settings.topK]);
  const agentName = route.kind === 'agent' && agents ? chosenAgent(route.name, agents) : undefined;
  // An address naming an agent the runtime does not serve opens another; the
  // warning stays while that substitute is open.
  const [missing, setMissing] = useState<{ named: string; shown: string }>();
  useEffect(() => {
    if (route.kind === 'agent' && route.name && agentName && route.name !== agentName)
      setMissing({ named: route.name, shown: agentName });
  }, [route, agentName]);
  const unregistered =
    missing &&
    route.kind === 'agent' &&
    agentName === missing.shown &&
    (!route.name || route.name === missing.named || route.name === missing.shown)
      ? missing.named
      : undefined;
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
      <>
        {unregistered && (
          <p className="alert warning" role="alert">
            {`Agent '${unregistered}' is not registered with the runtime, so this is ${agentName} instead.`}
          </p>
        )}
        <AgentWorkspace
          key={`${agentName}/${threadId}`}
          agentName={agentName}
          threadId={threadId!}
          agents={agents}
          settings={settings}
          onSettings={(next) => {
            setSettings(next);
            saveSettings(next);
          }}
          onNewThread={() => {
            window.location.hash = routeHash({ kind: 'agent', name: agentName, thread: crypto.randomUUID() });
          }}
        />
      </>
    );
  }

  return (
    <CopilotKitProvider
      runtimeUrl="/ui-api/copilotkit"
      renderActivityMessages={[noticeRenderer]}
      properties={properties}
    >
      <div className="shell">
        <Sidebar agents={agents ?? []} route={route.kind === 'agent' ? { kind: 'agent', name: agentName } : route} />
        <main className="main">{main}</main>
      </div>
    </CopilotKitProvider>
  );
}
