import { agentHue, agentLabel } from './api';

export function AgentList({
  agents,
  selected,
  onSelect,
}: {
  agents: string[];
  selected?: string;
  onSelect: (name: string) => void;
}) {
  return (
    <nav className="sidebar" aria-label="Agents">
      <div className="brand">Cogniverse</div>
      <ul>
        {agents.map((name) => (
          <li key={name}>
            <button
              className={name === selected ? 'dot-item active' : 'dot-item'}
              aria-current={name === selected ? 'page' : undefined}
              onClick={() => onSelect(name)}
            >
              <span
                className="dot"
                style={{ background: `hsl(${agentHue(name)} 65% 55%)` }}
                aria-hidden
              />
              {agentLabel(name)}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
