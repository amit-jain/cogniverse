import { ActiveTenant } from './ActiveTenant';
import { useAgentsStatus } from './agentStatus';
import { agentHue, agentLabel } from './api';
import { OPS_VIEWS } from './ops/views';
import { routeHash, type Route } from './route';

export function Sidebar({ agents, route }: { agents: string[]; route: Route & { name?: string } }) {
  const health = useAgentsStatus();
  const statusOf = (name: string) => health.status?.agents.find((agent) => agent.name === name);
  return (
    <nav className="sidebar" aria-label="Navigation">
      <div className="brand">Cogniverse</div>
      <ActiveTenant />
      <h2 className="nav-heading">Agents</h2>
      {health.error && (
        <p className="agent-status-error" role="alert">
          Agent status is unavailable: {health.error}
        </p>
      )}
      <ul>
        {agents.map((name) => {
          const active = route.kind === 'agent' && route.name === name;
          const status = statusOf(name);
          return (
            <li key={name}>
              <a
                className={active ? 'dot-item active' : 'dot-item'}
                aria-current={active ? 'page' : undefined}
                href={routeHash({ kind: 'agent', name })}
              >
                <span className="dot" style={{ background: `hsl(${agentHue(name)} 65% 55%)` }} aria-hidden />
                {agentLabel(name)}
              </a>
              {status && (
                <span
                  className={`agent-status ${status.status}`}
                  aria-label={`${agentLabel(name)} is ${status.status}`}
                  title={status.message ?? (status.health ? `Registry health: ${status.health}` : undefined)}
                >
                  {status.status}
                </span>
              )}
            </li>
          );
        })}
      </ul>
      <h2 className="nav-heading">Operations</h2>
      <ul>
        {OPS_VIEWS.map((view) => {
          const active = route.kind === 'ops' && route.id === view.id;
          return (
            <li key={view.id}>
              <a
                className={active ? 'dot-item active' : 'dot-item'}
                aria-current={active ? 'page' : undefined}
                href={routeHash({ kind: 'ops', id: view.id })}
              >
                {view.label}
              </a>
            </li>
          );
        })}
      </ul>
      <p className="sidebar-footer">Cogniverse operations and agents</p>
    </nav>
  );
}
