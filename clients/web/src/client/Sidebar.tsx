import { agentHue, agentLabel } from './api';
import { OPS_VIEWS } from './ops/views';
import { routeHash, type Route } from './route';

export function Sidebar({ agents, route }: { agents: string[]; route: Route & { name?: string } }) {
  return (
    <nav className="sidebar" aria-label="Navigation">
      <div className="brand">Cogniverse</div>
      <h2 className="nav-heading">Agents</h2>
      <ul>
        {agents.map((name) => {
          const active = route.kind === 'agent' && route.name === name;
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
    </nav>
  );
}
