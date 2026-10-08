import { useState } from 'react';
import { Alert, Panel, useAction, useLoad } from './common';
import { jsonText, parseJsonObject } from './forms';
import { runtimeJson, seg } from './http';
import { TenantChooser } from './tenants';

interface Memory {
  id: string;
  memory: string;
  category: string | null;
  metadata: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
  score: number | null;
}

interface MemoryStats {
  agent_name: string;
  user_id: string;
  total: number;
  archived: number;
  writable: boolean;
}

interface MemoryHealth {
  healthy: boolean;
  problem: string | null;
}

const USER_MEMORIES = '_user_memories';
const SYSTEM_SUGGESTIONS = [USER_MEMORIES, '_strategy_store'];
/** The most memories the list route returns in one read. */
const LIST_LIMIT = 200;
const DEFAULT_LIMIT = 20;

function memoriesPath(tenant: string): string {
  return `/admin/tenant/${seg(tenant)}/memories`;
}

export function MemoryView() {
  const [tenant, setTenant] = useState('');
  const [namespace, setNamespace] = useState(USER_MEMORIES);
  const [version, setVersion] = useState(0);
  const [notice, setNotice] = useState('');
  const [saved, setSaved] = useState('');
  const changed = (message: string, savedJson = '') => {
    setNotice(message);
    setSaved(savedJson);
    setVersion((n) => n + 1);
  };
  return (
    <div className="ops-view">
      <TenantChooser
        action="Show memories"
        onChoose={(chosen) => {
          setTenant(chosen);
          setNotice('');
          setSaved('');
        }}
      />
      {tenant && (
        <NamespaceChooser
          namespace={namespace}
          onChoose={(chosen) => {
            setNamespace(chosen);
            setNotice('');
            setSaved('');
          }}
        />
      )}
      {notice && <Alert tone="ok">{notice}</Alert>}
      {saved && <pre aria-label="Saved memory">{saved}</pre>}
      {tenant && (
        <Memories key={`${tenant}-${namespace}-${version}`} tenant={tenant} namespace={namespace} onChanged={changed} />
      )}
    </div>
  );
}

function NamespaceChooser({ namespace, onChoose }: { namespace: string; onChoose: (namespace: string) => void }) {
  const agents = useLoad(
    (signal) => runtimeJson<{ agents: string[] }>('/agents/', { signal }).then((body) => body.agents),
    [],
  );
  const [draft, setDraft] = useState(namespace);
  return (
    <Panel title="Namespace">
      <form
        className="inline-form"
        aria-label="Choose namespace"
        onSubmit={(e) => {
          e.preventDefault();
          onChoose(draft.trim());
        }}
      >
        <label>
          Namespace
          <input required list="memory-namespaces" value={draft} onChange={(e) => setDraft(e.target.value)} />
          <datalist id="memory-namespaces">
            {[...SYSTEM_SUGGESTIONS, ...(agents.data ?? [])].map((name) => (
              <option key={name} value={name} />
            ))}
          </datalist>
        </label>
        <button type="submit">Show</button>
        {agents.error && <Alert>Agent suggestions are unavailable: {agents.error}</Alert>}
      </form>
    </Panel>
  );
}

function Memories({
  tenant,
  namespace,
  onChanged,
}: {
  tenant: string;
  namespace: string;
  onChanged: (notice: string) => void;
}) {
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(DEFAULT_LIMIT);
  const stats = useLoad(
    (signal) =>
      runtimeJson<MemoryStats>(`${memoriesPath(tenant)}/stats?agent_name=${seg(namespace)}`, { signal }),
    [tenant, namespace],
  );
  const health = useLoad(
    (signal) =>
      runtimeJson<MemoryHealth>(`${memoriesPath(tenant)}/health?agent_name=${seg(namespace)}`, { signal }),
    [tenant, namespace],
  );
  const memories = useLoad(
    (signal) => {
      const params = new URLSearchParams({ agent_name: namespace, limit: String(limit) });
      if (query) params.set('q', query);
      return runtimeJson<{ memories: Memory[] }>(`${memoriesPath(tenant)}?${params}`, { signal }).then(
        (body) => body.memories,
      );
    },
    [tenant, namespace, query, limit],
  );
  const writable = stats.data?.writable ?? false;
  return (
    <>
      <Panel
        title={`Memories of ${namespace} in ${tenant}`}
        actions={
          <>
            <button onClick={health.reload}>Check health</button>
            <button
              onClick={() => {
                stats.reload();
                memories.reload();
              }}
            >
              Refresh
            </button>
          </>
        }
      >
        {health.error && <Alert>{health.error}</Alert>}
        {stats.error && <Alert>{stats.error}</Alert>}
        {stats.data && (
          <p className="muted">
            {stats.data.total} live, {stats.data.archived} archived.
            {!writable && ' A system namespace: the runtime manages these memories, so they are read-only here.'}
          </p>
        )}
        <dl className="facts" aria-label="Memory store facts">
          <dt>User ID</dt>
          <dd>{stats.data?.user_id ?? '…'}</dd>
          <dt>Agent ID</dt>
          <dd>{namespace}</dd>
          <dt>Health</dt>
          <dd className={health.data && !health.data.healthy ? 'alert error' : undefined}>
            {health.loading
              ? 'checking…'
              : health.data
                ? health.data.healthy
                  ? 'healthy: the store answers reads'
                  : `unhealthy: ${health.data.problem}`
                : '—'}
          </dd>
        </dl>
        <SearchForm query={query} limit={limit} onSearch={(q, n) => (setQuery(q), setLimit(n))} />
        {memories.error && <Alert>{memories.error}</Alert>}
        {memories.data && memories.data.length === 0 && (
          <p className="muted">{query ? `No memories match “${query}”.` : `No memories in ${namespace}.`}</p>
        )}
        {memories.data && memories.data.length > 0 && (
          <table aria-label="Memories">
            <thead>
              <tr>
                <th>Memory</th>
                {query && <th>Score</th>}
                <th>Category</th>
                <th>Created</th>
                <th>Updated</th>
                <th>ID</th>
                <th>Details</th>
                {writable && <th />}
              </tr>
            </thead>
            <tbody>
              {memories.data.map((memory) => (
                <tr key={memory.id}>
                  <td>{memory.memory}</td>
                  {query && <td>{memory.score === null ? '—' : memory.score.toFixed(3)}</td>}
                  <td>{memory.category ?? '—'}</td>
                  <td>{memory.created_at ?? '—'}</td>
                  <td>{memory.updated_at ?? '—'}</td>
                  <td>{memory.id}</td>
                  <td>
                    <MemoryDetails memory={memory} />
                  </td>
                  {writable && (
                    <td>
                      <DeleteMemory tenant={tenant} namespace={namespace} id={memory.id} onDeleted={onChanged} />
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {writable && <ClearNamespace tenant={tenant} namespace={namespace} onCleared={onChanged} />}
      </Panel>
      {writable && <AddMemory tenant={tenant} namespace={namespace} onAdded={onChanged} />}
    </>
  );
}

/** A memory's every field, metadata included, shown once opened. */
function MemoryDetails({ memory }: { memory: Memory }) {
  const [open, setOpen] = useState(false);
  return (
    <details aria-label={`Details of memory ${memory.id}`} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>Details</summary>
      {open && <pre>{jsonText(memory)}</pre>}
    </details>
  );
}

function SearchForm({
  query,
  limit,
  onSearch,
}: {
  query: string;
  limit: number;
  onSearch: (query: string, limit: number) => void;
}) {
  const [draft, setDraft] = useState(query);
  const [count, setCount] = useState(String(limit));
  const action = useAction();
  const run = (text: string) =>
    action.run(async () => {
      const n = Number(count);
      if (!Number.isInteger(n) || n < 1 || n > LIST_LIMIT)
        throw new Error(`Results must be a whole number from 1 to ${LIST_LIMIT}.`);
      onSearch(text, n);
    });
  return (
    <form
      className="inline-form"
      aria-label="Search memories"
      onSubmit={(e) => {
        e.preventDefault();
        run(draft.trim());
      }}
    >
      <label>
        Query
        <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="semantic search" />
      </label>
      <label>
        Results
        <input
          inputMode="numeric"
          value={count}
          onChange={(e) => setCount(e.target.value)}
          placeholder={`1–${LIST_LIMIT}`}
        />
      </label>
      <button type="submit">Search</button>
      {query && (
        <button type="button" onClick={() => (setDraft(''), run(''))}>
          Show all
        </button>
      )}
      {action.error && <Alert>{action.error}</Alert>}
    </form>
  );
}

function DeleteMemory({
  tenant,
  namespace,
  id,
  onDeleted,
}: {
  tenant: string;
  namespace: string;
  id: string;
  onDeleted: (notice: string) => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const action = useAction();
  if (!confirming)
    return (
      <button className="danger" aria-label={`Delete memory ${id}`} onClick={() => setConfirming(true)}>
        Delete
      </button>
    );
  return (
    <span className="confirm">
      <button
        className="danger"
        disabled={action.pending}
        onClick={() =>
          action.run(async () => {
            await runtimeJson(`${memoriesPath(tenant)}/${seg(id)}?agent_name=${seg(namespace)}`, {
              method: 'DELETE',
            });
            onDeleted(`Deleted memory ${id}.`);
          })
        }
      >
        {action.pending ? 'Deleting…' : `Confirm delete of ${id}`}
      </button>
      <button onClick={() => (setConfirming(false), action.clear())}>Cancel</button>
      {action.error && <Alert>{action.error}</Alert>}
    </span>
  );
}

function ClearNamespace({
  tenant,
  namespace,
  onCleared,
}: {
  tenant: string;
  namespace: string;
  onCleared: (notice: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [typed, setTyped] = useState('');
  const action = useAction();
  return (
    <div className="inline-form" role="group" aria-label="Clear namespace">
      {!open ? (
        <button className="danger" onClick={() => setOpen(true)}>
          Clear every memory of {namespace}
        </button>
      ) : (
        <span className="confirm">
          <input
            aria-label={`Type ${namespace} to clear its memories`}
            placeholder={namespace}
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
          />
          <button
            className="danger"
            disabled={typed !== namespace || action.pending}
            onClick={() =>
              action.run(async () => {
                await runtimeJson(`${memoriesPath(tenant)}?agent_name=${seg(namespace)}`, { method: 'DELETE' });
                onCleared(`Cleared every memory of ${namespace}.`);
              })
            }
          >
            {action.pending ? 'Clearing…' : `Clear ${namespace}`}
          </button>
          <button onClick={() => (setOpen(false), setTyped(''), action.clear())}>Cancel</button>
        </span>
      )}
      {action.error && <Alert>{action.error}</Alert>}
    </div>
  );
}

function AddMemory({
  tenant,
  namespace,
  onAdded,
}: {
  tenant: string;
  namespace: string;
  onAdded: (notice: string, savedJson: string) => void;
}) {
  const [text, setText] = useState('');
  const [category, setCategory] = useState('');
  const [metadata, setMetadata] = useState('');
  const action = useAction();
  return (
    <Panel title={`Add a memory to ${namespace}`}>
      <form
        className="stacked-form"
        aria-label="Add memory"
        onSubmit={(e) => {
          e.preventDefault();
          action.run(async () => {
            const body: Record<string, unknown> = { text, agent_name: namespace };
            if (category.trim()) body.category = category.trim();
            if (metadata.trim()) body.metadata = parseJsonObject('Metadata', metadata);
            const saved = await runtimeJson<{ id: string }>(memoriesPath(tenant), { method: 'POST', body });
            onAdded(`Saved memory ${saved.id} to ${namespace}.`, jsonText(saved));
          });
        }}
      >
        <label>
          Memory
          <textarea required rows={3} value={text} onChange={(e) => setText(e.target.value)} />
        </label>
        <label>
          Category
          <input value={category} onChange={(e) => setCategory(e.target.value)} placeholder="optional" />
        </label>
        <label>
          Metadata (JSON)
          <textarea rows={3} value={metadata} onChange={(e) => setMetadata(e.target.value)} placeholder="{}" />
        </label>
        <button type="submit" disabled={action.pending}>
          {action.pending ? 'Saving…' : 'Save memory'}
        </button>
        {action.error && <Alert>{action.error}</Alert>}
      </form>
    </Panel>
  );
}
