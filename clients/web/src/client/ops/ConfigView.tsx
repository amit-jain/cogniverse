import { useState } from 'react';
import { Alert, Panel, useAction, useLoad } from './common';
import { jsonText, type JsonObject } from './forms';
import { runtimeJson, seg } from './http';
import {
  formFields,
  fromFormState,
  toFormState,
  type FormField,
  type FormState,
  type JsonSchema,
  type SecretInput,
} from './schemaForm';
import { TenantChooser } from './tenants';

interface Section {
  name: string;
  title: string;
  tenant_scoped: boolean;
  service: string | null;
  schema: JsonSchema;
}

interface SectionValue {
  tenant_id: string;
  service: string;
  version: number;
  updated_at: string | null;
  value: JsonObject;
  secrets: Record<string, boolean>;
}

interface Entry {
  scope: string;
  service: string;
  config_key: string;
  version: number;
  updated_at: string;
  section: string | null;
}

interface HistoryVersion {
  version: number;
  updated_at: string;
  value: JsonObject;
}

const query = (params: Record<string, string | undefined>) =>
  Object.entries(params)
    .filter(([, value]) => value !== undefined)
    .map(([key, value]) => `${key}=${seg(value as string)}`)
    .join('&');

export function ConfigView() {
  const sections = useLoad(
    (signal) => runtimeJson<{ sections: Section[] }>('/admin/config/sections', { signal }).then((b) => b.sections),
    [],
  );
  const [tenant, setTenant] = useState('');
  const [version, setVersion] = useState(0);
  const [notice, setNotice] = useState('');
  const changed = (message: string) => {
    setNotice(message);
    setVersion((n) => n + 1);
  };
  const all = sections.data ?? [];
  return (
    <div className="ops-view">
      {notice && <Alert tone="ok">{notice}</Alert>}
      {sections.error && <Alert>{sections.error}</Alert>}
      {all
        .filter((s) => !s.tenant_scoped)
        .map((s) => (
          <SectionPanel key={`${s.name}-${version}`} section={s} onSaved={changed} />
        ))}
      <StoredConfigs key={`system-${version}`} onChanged={changed} />
      <TenantChooser
        action="Show configs"
        onChoose={(chosen) => {
          setTenant(chosen);
          setNotice('');
        }}
      />
      {tenant && (
        <>
          {all
            .filter((s) => s.tenant_scoped && s.service !== null)
            .map((s) => (
              <SectionPanel key={`${tenant}-${s.name}-${version}`} section={s} tenant={tenant} onSaved={changed} />
            ))}
          {all
            .filter((s) => s.tenant_scoped && s.service === null)
            .map((s) => (
              <AgentConfigs key={`${tenant}-${s.name}-${version}`} section={s} tenant={tenant} onSaved={changed} />
            ))}
          <StoredConfigs key={`${tenant}-${version}`} tenant={tenant} onChanged={changed} />
          <ExportImport key={`${tenant}-transfer`} tenant={tenant} onImported={changed} />
        </>
      )}
      <StoreStats key={`stats-${version}`} />
    </div>
  );
}

function SectionPanel({
  section,
  tenant,
  service,
  title,
  onSaved,
}: {
  section: Section;
  tenant?: string;
  service?: string;
  title?: string;
  onSaved: (notice: string) => void;
}) {
  const loaded = useLoad(
    (signal) =>
      runtimeJson<SectionValue>(
        `/admin/config/sections/${seg(section.name)}?${query({ tenant_id: tenant, service })}`,
        { signal },
      ),
    [section.name, tenant, service],
  );
  const heading = title ?? (tenant ? `${section.title} config of ${tenant}` : `${section.title} config`);
  return (
    <Panel title={heading} actions={<button onClick={loaded.reload}>Reload</button>}>
      {loaded.error && <Alert>{loaded.error}</Alert>}
      {loaded.data && (
        <SectionForm
          key={loaded.data.version}
          section={section}
          loaded={loaded.data}
          tenant={tenant}
          heading={heading}
          onSaved={onSaved}
        />
      )}
    </Panel>
  );
}

function SectionForm({
  section,
  loaded,
  tenant,
  heading,
  onSaved,
}: {
  section: Section;
  loaded: SectionValue;
  tenant?: string;
  heading: string;
  onSaved: (notice: string) => void;
}) {
  const fields = formFields(section.schema);
  const [state, setState] = useState<FormState>(() => toFormState(fields, loaded.value));
  const action = useAction();
  return (
    <form
      className="stacked-form"
      aria-label={`Edit ${heading}`}
      onSubmit={(e) => {
        e.preventDefault();
        action.run(async () => {
          const saved = await runtimeJson<SectionValue>(`/admin/config/sections/${seg(section.name)}`, {
            method: 'PUT',
            body: {
              tenant_id: tenant,
              service: section.service === null ? loaded.service : undefined,
              value: fromFormState(fields, state),
              version: loaded.version,
            },
          });
          onSaved(`Saved ${heading} as version ${saved.version}.`);
        });
      }}
    >
      <p className="muted">
        {loaded.version
          ? `Version ${loaded.version}, saved ${new Date(loaded.updated_at ?? '').toLocaleString()}.`
          : 'Not saved yet; showing the defaults.'}
      </p>
      <Fields fields={fields} state={state} secrets={loaded.secrets} onChange={setState} />
      <button type="submit" disabled={action.pending}>
        {action.pending ? 'Saving…' : 'Save'}
      </button>
      {action.error && <Alert>{action.error}</Alert>}
    </form>
  );
}

function Fields({
  fields,
  state,
  secrets = {},
  onChange,
}: {
  fields: FormField[];
  state: FormState;
  secrets?: Record<string, boolean>;
  onChange: (next: FormState) => void;
}) {
  const set = (name: string, value: FormState[string]) => onChange({ ...state, [name]: value });
  return (
    <>
      {fields.map((f) => {
        const value = state[f.name];
        switch (f.kind) {
          case 'object':
            return (
              <fieldset key={f.name}>
                <legend>{f.name}</legend>
                <Fields fields={f.fields ?? []} state={value as FormState} onChange={(next) => set(f.name, next)} />
              </fieldset>
            );
          case 'boolean':
            return (
              <label key={f.name} className="check">
                <input type="checkbox" checked={value as boolean} onChange={(e) => set(f.name, e.target.checked)} />
                {f.name}
              </label>
            );
          case 'choice':
            return (
              <label key={f.name}>
                {f.name}
                <select aria-label={f.name} value={value as string} onChange={(e) => set(f.name, e.target.value)}>
                  {f.nullable && <option value="">(none)</option>}
                  {(f.options ?? []).map((option) => (
                    <option key={option}>{option}</option>
                  ))}
                </select>
              </label>
            );
          case 'json':
            return (
              <label key={f.name}>
                {f.name} (JSON)
                <textarea
                  aria-label={f.name}
                  rows={3}
                  spellCheck={false}
                  value={value as string}
                  onChange={(e) => set(f.name, e.target.value)}
                />
              </label>
            );
          case 'secret': {
            const secret = value as SecretInput;
            return (
              <div key={f.name} className="inline-form">
                <label>
                  {f.name}
                  <input
                    type="password"
                    autoComplete="off"
                    value={secret.text}
                    disabled={secret.clear}
                    placeholder={secrets[f.name] ? 'set; leave blank to keep' : 'not set'}
                    onChange={(e) => set(f.name, { ...secret, text: e.target.value })}
                  />
                </label>
                {secrets[f.name] && (
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={secret.clear}
                      onChange={(e) => set(f.name, { text: '', clear: e.target.checked })}
                    />
                    Clear {f.name}
                  </label>
                )}
              </div>
            );
          }
          default:
            return (
              <label key={f.name}>
                {f.name}
                <input
                  inputMode={f.kind === 'text' ? undefined : 'decimal'}
                  value={value as string}
                  placeholder={f.nullable ? 'unset' : undefined}
                  onChange={(e) => set(f.name, e.target.value)}
                />
              </label>
            );
        }
      })}
    </>
  );
}

function AgentConfigs({
  section,
  tenant,
  onSaved,
}: {
  section: Section;
  tenant: string;
  onSaved: (notice: string) => void;
}) {
  const entries = useLoad(
    (signal) =>
      runtimeJson<{ entries: Entry[] }>(`/admin/config/entries?${query({ tenant_id: tenant })}`, { signal }).then(
        (body) => body.entries.filter((e) => e.section === section.name).map((e) => e.service),
      ),
    [tenant, section.name],
  );
  const [agent, setAgent] = useState('');
  const [draft, setDraft] = useState('');
  return (
    <Panel title={`${section.title} of ${tenant}`}>
      {entries.error && <Alert>{entries.error}</Alert>}
      <form
        className="inline-form"
        aria-label="Choose agent"
        onSubmit={(e) => {
          e.preventDefault();
          setAgent(draft.trim());
        }}
      >
        <label>
          Agent
          <input
            required
            list="configured-agents"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="search_agent"
          />
          <datalist id="configured-agents">
            {(entries.data ?? []).map((name) => (
              <option key={name} value={name} />
            ))}
          </datalist>
        </label>
        <button type="submit">Edit agent config</button>
      </form>
      {entries.data && (
        <p className="muted">
          {entries.data.length ? `Configured: ${entries.data.join(', ')}.` : 'No agent configs saved for this tenant.'}
        </p>
      )}
      {agent && (
        <SectionPanel
          key={agent}
          section={section}
          tenant={tenant}
          service={agent}
          title={`Agent ${agent} config of ${tenant}`}
          onSaved={onSaved}
        />
      )}
    </Panel>
  );
}

function StoredConfigs({ tenant, onChanged }: { tenant?: string; onChanged: (notice: string) => void }) {
  const entries = useLoad(
    (signal) =>
      runtimeJson<{ entries: Entry[] }>(`/admin/config/entries?${query({ tenant_id: tenant })}`, { signal }).then(
        (body) => body.entries,
      ),
    [tenant],
  );
  const [open, setOpen] = useState<Entry>();
  const owner = tenant ?? 'the system';
  return (
    <Panel title={`Stored configs of ${owner}`} actions={<button onClick={entries.reload}>Refresh</button>}>
      {entries.error && <Alert>{entries.error}</Alert>}
      {entries.data && entries.data.length === 0 && <p className="muted">No configs stored for {owner}.</p>}
      {entries.data && entries.data.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Scope</th>
              <th>Service</th>
              <th>Key</th>
              <th>Version</th>
              <th>Updated</th>
              <th>History</th>
            </tr>
          </thead>
          <tbody>
            {entries.data.map((e) => (
              <tr key={`${e.scope}/${e.service}/${e.config_key}`}>
                <td>{e.scope}</td>
                <td>{e.service}</td>
                <td>{e.config_key}</td>
                <td>{e.version}</td>
                <td>{new Date(e.updated_at).toLocaleString()}</td>
                <td>
                  <button
                    className="link"
                    aria-label={`History of ${e.scope}/${e.service}/${e.config_key}`}
                    onClick={() => setOpen(e)}
                  >
                    History
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {open && (
        <ConfigHistory
          key={`${open.scope}/${open.service}/${open.config_key}/${open.version}`}
          tenant={tenant}
          entry={open}
          onRestored={onChanged}
        />
      )}
    </Panel>
  );
}

function ConfigHistory({
  tenant,
  entry,
  onRestored,
}: {
  tenant?: string;
  entry: Entry;
  onRestored: (notice: string) => void;
}) {
  const where = { tenant_id: tenant, scope: entry.scope, service: entry.service, config_key: entry.config_key };
  const history = useLoad(
    (signal) =>
      runtimeJson<{ versions: HistoryVersion[] }>(`/admin/config/history?${query(where)}`, { signal }).then(
        (body) => body.versions,
      ),
    [tenant, entry.scope, entry.service, entry.config_key],
  );
  const action = useAction();
  const latest = history.data?.[0]?.version;
  return (
    <div role="region" aria-label={`History of ${entry.scope}/${entry.service}/${entry.config_key}`}>
      <h3>
        History of {entry.scope}/{entry.service}/{entry.config_key}
      </h3>
      {history.error && <Alert>{history.error}</Alert>}
      {action.error && <Alert>{action.error}</Alert>}
      {(history.data ?? []).map((v) => (
        <details key={v.version}>
          <summary>
            Version {v.version}, {new Date(v.updated_at).toLocaleString()}
            {v.version === latest ? ' (current)' : ''}
          </summary>
          <pre>{jsonText(v.value)}</pre>
          {v.version !== latest && (
            <button
              disabled={action.pending}
              onClick={() =>
                action.run(async () => {
                  const written = await runtimeJson<{ version: number }>('/admin/config/rollback', {
                    method: 'POST',
                    body: { ...where, version: v.version, expected_version: latest },
                  });
                  onRestored(
                    `Restored version ${v.version} of ${entry.scope}/${entry.service}/${entry.config_key} as version ${written.version}.`,
                  );
                })
              }
            >
              Restore version {v.version}
            </button>
          )}
        </details>
      ))}
    </div>
  );
}

function ExportImport({ tenant, onImported }: { tenant: string; onImported: (notice: string) => void }) {
  const [history, setHistory] = useState(false);
  const [file, setFile] = useState<File>();
  const exporting = useAction();
  const importing = useAction();
  return (
    <Panel title={`Export and import for ${tenant}`}>
      <div className="inline-form" role="group" aria-label="Export configs">
        <label className="check">
          <input type="checkbox" checked={history} onChange={(e) => setHistory(e.target.checked)} />
          Include every version
        </label>
        <button
          disabled={exporting.pending}
          onClick={() =>
            exporting.run(async () => {
              const body = await runtimeJson<JsonObject>(
                `/admin/config/export?${query({ tenant_id: tenant, include_history: String(history) })}`,
              );
              const link = document.createElement('a');
              link.href = URL.createObjectURL(new Blob([jsonText(body)], { type: 'application/json' }));
              link.download = `config_export_${tenant.replace(/[^A-Za-z0-9_-]/g, '_')}.json`;
              link.click();
              URL.revokeObjectURL(link.href);
            })
          }
        >
          {exporting.pending ? 'Exporting…' : 'Export configs'}
        </button>
        {exporting.error && <Alert>{exporting.error}</Alert>}
      </div>
      <p className="muted">The export holds every secret: it is a backup the import restores whole.</p>
      <form
        className="inline-form"
        aria-label="Import configs"
        onSubmit={(e) => {
          e.preventDefault();
          importing.run(async () => {
            if (!file) throw new Error('Choose an export file first.');
            let configs: unknown;
            try {
              configs = JSON.parse(await file.text());
            } catch {
              throw new Error(`${file.name} is not valid JSON.`);
            }
            const result = await runtimeJson<{ imported: number }>('/admin/config/import', {
              method: 'POST',
              body: { tenant_id: tenant, configs },
            });
            onImported(`Imported ${result.imported} configs into ${tenant}.`);
          });
        }}
      >
        <label>
          Export file
          <input type="file" accept="application/json,.json" onChange={(e) => setFile(e.target.files?.[0])} />
        </label>
        <button type="submit" disabled={importing.pending}>
          {importing.pending ? 'Importing…' : 'Import configs'}
        </button>
        {importing.error && <Alert>{importing.error}</Alert>}
      </form>
    </Panel>
  );
}

function StoreStats() {
  const stats = useLoad(
    (signal) =>
      runtimeJson<{
        total_configs: number;
        total_versions: number;
        total_tenants: number;
        configs_per_scope: Record<string, number>;
      }>('/admin/config/stats', { signal }),
    [],
  );
  return (
    <Panel title="Config store" actions={<button onClick={stats.reload}>Refresh</button>}>
      {stats.error && <Alert>{stats.error}</Alert>}
      {stats.data && (
        <dl className="facts">
          <dt>Configs</dt>
          <dd>{stats.data.total_configs}</dd>
          <dt>Versions</dt>
          <dd>{stats.data.total_versions}</dd>
          <dt>Tenants</dt>
          <dd>{stats.data.total_tenants}</dd>
          <dt>By scope</dt>
          <dd>
            {Object.entries(stats.data.configs_per_scope)
              .sort()
              .map(([scope, count]) => `${scope}: ${count}`)
              .join(', ')}
          </dd>
        </dl>
      )}
    </Panel>
  );
}
