import { CopilotKitProvider } from '@copilotkit/react-core/v2';
import { useEffect, useState } from 'react';
import { AgentWorkspace } from './AgentWorkspace';
import { fetchAgents } from './api';
import { noticeRenderer } from './Notice';
import { OPS_VIEWS } from './ops/views';
import { parseRoute, routeHash } from './route';
import { Sidebar } from './Sidebar';
import { TENANT_HEADER, chooseTenant, currentTenant, useTenant } from './tenant';

const THREAD_KEY = 'cogniverse.thread.';

/** The thread this browser last had open with ``agent`` for ``tenant``, if any. */
function rememberedThread(tenant: string, agent: string): string | undefined {
  try {
    return localStorage.getItem(`${THREAD_KEY}${tenant}/${agent}`) ?? undefined;
  } catch {
    return undefined;
  }
}

function rememberThread(tenant: string, agent: string, thread: string) {
  try {
    localStorage.setItem(`${THREAD_KEY}${tenant}/${agent}`, thread);
  } catch {
    // Without storage the thread lives in the address alone.
  }
}

/** Every request CopilotKit sends names the active tenant. */
const tenantHeaders = (): Record<string, string> => {
  const tenant = currentTenant();
  return tenant ? { [TENANT_HEADER]: tenant } : {};
};

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
  const { tenant } = useTenant();
  // A tenant kept from an earlier visit is checked with the runtime again.
  useEffect(() => {
    if (currentTenant()) chooseTenant(currentTenant());
  }, []);
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
  // A thread belongs to one tenant: one taken from the address is used only
  // while the tenant it was opened for is still active.
  const [threadTenants] = useState(() => new Map<string, string>());
  let threadId: string | undefined;
  if (agentName && tenant) {
    const routed = route.kind === 'agent' && route.name === agentName ? route.thread : undefined;
    const owner = routed ? threadTenants.get(routed) : undefined;
    threadId =
      (routed && (owner === undefined || owner === tenant) ? routed : undefined) ??
      threads.get(`${tenant}/${agentName}`) ??
      rememberedThread(tenant, agentName) ??
      crypto.randomUUID();
    threads.set(`${tenant}/${agentName}`, threadId);
    threadTenants.set(threadId, tenant);
  }
  useEffect(() => {
    if (!agentName || !threadId || !tenant) return;
    rememberThread(tenant, agentName, threadId);
    const hash = routeHash({ kind: 'agent', name: agentName, thread: threadId });
    if (window.location.hash !== hash) window.history.replaceState(null, '', hash);
  }, [agentName, threadId, tenant]);
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
  } else if (!tenant) {
    main = (
      <div className="notice">
        Choose the active tenant in the sidebar before talking to an agent. Agents run for that tenant only.
      </div>
    );
  } else {
    main = (
      <AgentWorkspace
        key={`${tenant}/${agentName}/${threadId}`}
        tenant={tenant}
        agentName={agentName}
        threadId={threadId!}
        onNewThread={() => {
          window.location.hash = routeHash({ kind: 'agent', name: agentName, thread: crypto.randomUUID() });
        }}
      />
    );
  }

  return (
    <CopilotKitProvider runtimeUrl="/ui-api/copilotkit" headers={tenantHeaders} renderActivityMessages={[noticeRenderer]}>
      <div className="shell">
        <Sidebar agents={agents ?? []} route={route.kind === 'agent' ? { kind: 'agent', name: agentName } : route} />
        <main className="main">{main}</main>
      </div>
    </CopilotKitProvider>
  );
}
