import { useState } from 'react';
import { Alert, ConfirmDelete, Panel, useAction, useLoad } from './common';
import { jsonText, parseJsonObject, sameJson, type JsonObject } from './forms';
import { runtimeJson, seg } from './http';

interface ProfileSummary {
  profile_name: string;
  type: string;
  description: string;
  schema_name: string;
  embedding_model: string;
  schema_deployed: boolean;
}

interface ProfileDetail {
  profile_name: string;
  tenant_id: string;
  type: string;
  description: string;
  schema_name: string;
  embedding_model: string;
  pipeline_config: JsonObject;
  strategies: JsonObject;
  embedding_type: string;
  schema_config: JsonObject;
  model_specific: JsonObject | null;
  schema_deployed: boolean;
  tenant_schema_name: string | null;
  version: number;
}

interface Deployment {
  schema_name: string;
  tenant_schema_name: string;
  deployment_status: 'success' | 'failed' | 'already_deployed';
  error_message?: string | null;
}

const EMBEDDING_TYPES = ['multi_vector', 'single_vector'];

async function knownTenants(signal: AbortSignal): Promise<string[]> {
  const { organizations } = await runtimeJson<{ organizations: { org_id: string }[] }>(
    '/admin/organizations',
    { signal },
  );
  const lists = await Promise.all(
    organizations.map((org) =>
      runtimeJson<{ tenants: { tenant_full_id: string }[] }>(
        `/admin/organizations/${seg(org.org_id)}/tenants`,
        { signal },
      ),
    ),
  );
  return lists.flatMap((list) => list.tenants.map((tenant) => tenant.tenant_full_id)).sort();
}

export function ProfilesView() {
  const known = useLoad(knownTenants, []);
  const [draft, setDraft] = useState('');
  const [tenant, setTenant] = useState('');
  const [selected, setSelected] = useState<string>();
  const [version, setVersion] = useState(0);
  const [notice, setNotice] = useState('');
  const changed = (message: string, select?: string | null) => {
    setNotice(message);
    if (select !== undefined) setSelected(select ?? undefined);
    setVersion((n) => n + 1);
  };

  return (
    <div className="ops-view">
      <Panel title="Tenant">
        <form
          className="inline-form"
          aria-label="Choose tenant"
          onSubmit={(e) => {
            e.preventDefault();
            setTenant(draft.trim());
            setSelected(undefined);
            setNotice('');
          }}
        >
          <label>
            Tenant ID
            <input
              required
              list="profile-tenants"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="acme:production"
            />
            <datalist id="profile-tenants">
              {(known.data ?? []).map((id) => (
                <option key={id} value={id} />
              ))}
            </datalist>
          </label>
          <button type="submit">Show profiles</button>
          {known.error && <Alert>Tenant suggestions are unavailable: {known.error}</Alert>}
        </form>
      </Panel>
      {notice && <Alert tone="ok">{notice}</Alert>}
      {tenant && (
        <>
          <ProfilesPanel
            key={`${tenant}-${version}`}
            tenant={tenant}
            selected={selected}
            onSelect={setSelected}
          />
          {selected && (
            <ProfilePanel key={`${tenant}-${selected}-${version}`} tenant={tenant} name={selected} onChanged={changed} />
          )}
          <CreateProfile key={tenant} tenant={tenant} onCreated={changed} />
        </>
      )}
    </div>
  );
}

function ProfilesPanel({
  tenant,
  selected,
  onSelect,
}: {
  tenant: string;
  selected?: string;
  onSelect: (name: string) => void;
}) {
  const profiles = useLoad(
    (signal) =>
      runtimeJson<{ profiles: ProfileSummary[] }>(`/admin/profiles?tenant_id=${seg(tenant)}`, {
        signal,
      }).then((body) => body.profiles),
    [tenant],
  );
  return (
    <Panel title={`Profiles of ${tenant}`} actions={<button onClick={profiles.reload}>Refresh</button>}>
      {profiles.error && <Alert>{profiles.error}</Alert>}
      {profiles.data && profiles.data.length === 0 && (
        <p className="muted">No profiles created for {tenant}.</p>
      )}
      {profiles.data && profiles.data.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Profile</th>
              <th>Type</th>
              <th>Schema</th>
              <th>Embedding model</th>
              <th>Schema deployed</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            {profiles.data.map((profile) => (
              <tr key={profile.profile_name} className={profile.profile_name === selected ? 'selected' : undefined}>
                <td>
                  <button className="link" onClick={() => onSelect(profile.profile_name)}>
                    {profile.profile_name}
                  </button>
                </td>
                <td>{profile.type}</td>
                <td>{profile.schema_name}</td>
                <td>{profile.embedding_model}</td>
                <td>{profile.schema_deployed ? 'yes' : 'no'}</td>
                <td>{profile.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="muted">Shipped profiles are not listed; these are the profiles created for this tenant.</p>
    </Panel>
  );
}

function ProfilePanel({
  tenant,
  name,
  onChanged,
}: {
  tenant: string;
  name: string;
  onChanged: (notice: string, select?: string | null) => void;
}) {
  const detail = useLoad(
    (signal) =>
      runtimeJson<ProfileDetail>(`/admin/profiles/${seg(name)}?tenant_id=${seg(tenant)}`, { signal }),
    [tenant, name],
  );
  const profile = detail.data;
  return (
    <Panel title={`Profile ${name}`}>
      {detail.error && <Alert>{detail.error}</Alert>}
      {profile && (
        <>
          <dl className="facts">
            <dt>Type</dt>
            <dd>{profile.type}</dd>
            <dt>Schema</dt>
            <dd>{profile.schema_name}</dd>
            <dt>Deployed as</dt>
            <dd>{profile.tenant_schema_name ?? 'not deployed'}</dd>
            <dt>Embedding model</dt>
            <dd>{profile.embedding_model}</dd>
            <dt>Embedding type</dt>
            <dd>{profile.embedding_type}</dd>
            <dt>Config version</dt>
            <dd>{profile.version}</dd>
            <dt>Schema config</dt>
            <dd>
              <pre>{jsonText(profile.schema_config)}</pre>
            </dd>
          </dl>
          <EditProfile tenant={tenant} profile={profile} onSaved={(notice) => onChanged(notice)} />
          <DeploySchema tenant={tenant} profile={profile} onDeployed={(notice) => onChanged(notice)} />
          <DeleteProfile tenant={tenant} profile={profile} onDeleted={(notice) => onChanged(notice, null)} />
        </>
      )}
    </Panel>
  );
}

function EditProfile({
  tenant,
  profile,
  onSaved,
}: {
  tenant: string;
  profile: ProfileDetail;
  onSaved: (notice: string) => void;
}) {
  const [description, setDescription] = useState(profile.description);
  const [pipeline, setPipeline] = useState(jsonText(profile.pipeline_config));
  const [strategies, setStrategies] = useState(jsonText(profile.strategies));
  const [modelSpecific, setModelSpecific] = useState(jsonText(profile.model_specific));
  const action = useAction();
  return (
    <form
      className="stacked-form"
      aria-label={`Edit profile ${profile.profile_name}`}
      onSubmit={(e) => {
        e.preventDefault();
        action.run(async () => {
          const changes: JsonObject = {};
          if (description !== profile.description) changes.description = description;
          const pipelineValue = parseJsonObject('Pipeline config', pipeline) ?? {};
          if (!sameJson(pipelineValue, profile.pipeline_config)) changes.pipeline_config = pipelineValue;
          const strategiesValue = parseJsonObject('Strategies', strategies) ?? {};
          if (!sameJson(strategiesValue, profile.strategies)) changes.strategies = strategiesValue;
          const modelValue = parseJsonObject('Model-specific parameters', modelSpecific);
          if (modelValue === undefined && profile.model_specific)
            throw new Error('Model-specific parameters cannot be removed; enter {} to clear them.');
          if (modelValue !== undefined && !sameJson(modelValue, profile.model_specific))
            changes.model_specific = modelValue;
          if (!Object.keys(changes).length) throw new Error('Nothing to save; no field changed.');
          const result = await runtimeJson<{ updated_fields: string[]; version: number }>(
            `/admin/profiles/${seg(profile.profile_name)}`,
            { method: 'PUT', body: { tenant_id: tenant, ...changes } },
          );
          onSaved(
            `Saved ${result.updated_fields.join(', ')} of ${profile.profile_name} (config version ${result.version}).`,
          );
        });
      }}
    >
      <h3>Edit</h3>
      <label>
        Description
        <input value={description} onChange={(e) => setDescription(e.target.value)} />
      </label>
      <JsonField label="Pipeline config" value={pipeline} onChange={setPipeline} />
      <JsonField label="Strategies" value={strategies} onChange={setStrategies} />
      <JsonField label="Model-specific parameters" value={modelSpecific} onChange={setModelSpecific} />
      <button type="submit" disabled={action.pending}>
        {action.pending ? 'Saving…' : 'Save changes'}
      </button>
      {action.error && <Alert>{action.error}</Alert>}
    </form>
  );
}

function DeploySchema({
  tenant,
  profile,
  onDeployed,
}: {
  tenant: string;
  profile: ProfileDetail;
  onDeployed: (notice: string) => void;
}) {
  const [force, setForce] = useState(false);
  const action = useAction();
  return (
    <div className="inline-form" role="group" aria-label="Deploy schema">
      <h3>Schema</h3>
      <label className="check">
        <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
        Redeploy even if already deployed
      </label>
      <button
        disabled={action.pending}
        onClick={() =>
          action.run(async () => {
            const result = await runtimeJson<Deployment>(`/admin/profiles/${seg(profile.profile_name)}/deploy`, {
              method: 'POST',
              body: { tenant_id: tenant, force },
            });
            if (result.deployment_status === 'failed')
              throw new Error(`Deploying schema ${result.schema_name} failed: ${result.error_message}`);
            onDeployed(
              result.deployment_status === 'already_deployed'
                ? `Schema ${result.schema_name} is already deployed as ${result.tenant_schema_name}.`
                : `Deployed schema ${result.schema_name} as ${result.tenant_schema_name}.`,
            );
          })
        }
      >
        {action.pending ? 'Deploying schema…' : 'Deploy schema'}
      </button>
      {action.error && <Alert>{action.error}</Alert>}
    </div>
  );
}

function DeleteProfile({
  tenant,
  profile,
  onDeleted,
}: {
  tenant: string;
  profile: ProfileDetail;
  onDeleted: (notice: string) => void;
}) {
  const [deleteSchema, setDeleteSchema] = useState(false);
  return (
    <div className="inline-form" role="group" aria-label="Delete profile">
      <h3>Delete</h3>
      <label className="check">
        <input type="checkbox" checked={deleteSchema} onChange={(e) => setDeleteSchema(e.target.checked)} />
        Also delete schema {profile.schema_name}
      </label>
      <ConfirmDelete
        name={profile.profile_name}
        what="profile"
        onDelete={async () => {
          const result = await runtimeJson<{ schema_deleted: boolean }>(
            `/admin/profiles/${seg(profile.profile_name)}?tenant_id=${seg(tenant)}&delete_schema=${deleteSchema}`,
            { method: 'DELETE' },
          );
          onDeleted(
            result.schema_deleted
              ? `Deleted profile ${profile.profile_name} and schema ${profile.schema_name}.`
              : deleteSchema
                ? `Deleted profile ${profile.profile_name}; schema ${profile.schema_name} was not deployed.`
                : `Deleted profile ${profile.profile_name}.`,
          );
        }}
      />
    </div>
  );
}

function CreateProfile({
  tenant,
  onCreated,
}: {
  tenant: string;
  onCreated: (notice: string, select: string) => void;
}) {
  const [name, setName] = useState('');
  const [type, setType] = useState('video');
  const [description, setDescription] = useState('');
  const [schemaName, setSchemaName] = useState('');
  const [embeddingModel, setEmbeddingModel] = useState('');
  const [embeddingType, setEmbeddingType] = useState(EMBEDDING_TYPES[0]);
  const [pipeline, setPipeline] = useState('');
  const [strategies, setStrategies] = useState('');
  const [schemaConfig, setSchemaConfig] = useState('');
  const [modelSpecific, setModelSpecific] = useState('');
  const [deploy, setDeploy] = useState(false);
  const action = useAction();
  return (
    <Panel title={`New profile for ${tenant}`}>
      <form
        className="stacked-form"
        aria-label="Create profile"
        onSubmit={(e) => {
          e.preventDefault();
          action.run(async () => {
            const body = {
              profile_name: name.trim(),
              tenant_id: tenant,
              type: type.trim(),
              description,
              schema_name: schemaName.trim(),
              embedding_model: embeddingModel.trim(),
              embedding_type: embeddingType,
              pipeline_config: parseJsonObject('Pipeline config', pipeline) ?? {},
              strategies: parseJsonObject('Strategies', strategies) ?? {},
              schema_config: parseJsonObject('Schema config', schemaConfig) ?? {},
              model_specific: parseJsonObject('Model-specific parameters', modelSpecific) ?? null,
              deploy_schema: deploy,
            };
            const created = await runtimeJson<{
              profile_name: string;
              version: number;
              schema_deployed: boolean;
              tenant_schema_name: string | null;
            }>('/admin/profiles', { method: 'POST', body });
            onCreated(
              `Created profile ${created.profile_name} (config version ${created.version})${
                created.schema_deployed ? ` and deployed schema ${created.tenant_schema_name}` : ''
              }.`,
              created.profile_name,
            );
            setName('');
          });
        }}
      >
        <div className="inline-form">
          <label>
            Profile name
            <input required value={name} onChange={(e) => setName(e.target.value)} placeholder="custom_colpali" />
          </label>
          <label>
            Type
            <input required value={type} onChange={(e) => setType(e.target.value)} />
          </label>
          <label>
            Schema name
            <input
              required
              value={schemaName}
              onChange={(e) => setSchemaName(e.target.value)}
              placeholder="video_colpali_smol500_mv_frame"
            />
          </label>
          <label>
            Embedding model
            <input
              required
              value={embeddingModel}
              onChange={(e) => setEmbeddingModel(e.target.value)}
              placeholder="TomoroAI/tomoro-colqwen3-embed-4b"
            />
          </label>
          <label>
            Embedding type
            <select value={embeddingType} onChange={(e) => setEmbeddingType(e.target.value)}>
              {EMBEDDING_TYPES.map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
        </div>
        <label>
          Description
          <input value={description} onChange={(e) => setDescription(e.target.value)} />
        </label>
        <JsonField label="Pipeline config" value={pipeline} onChange={setPipeline} />
        <JsonField label="Strategies" value={strategies} onChange={setStrategies} />
        <JsonField label="Schema config" value={schemaConfig} onChange={setSchemaConfig} />
        <JsonField label="Model-specific parameters" value={modelSpecific} onChange={setModelSpecific} />
        <label className="check">
          <input type="checkbox" checked={deploy} onChange={(e) => setDeploy(e.target.checked)} />
          Deploy the schema now
        </label>
        <button type="submit" disabled={action.pending}>
          {action.pending ? (deploy ? 'Creating and deploying…' : 'Creating…') : 'Create profile'}
        </button>
        {action.error && <Alert>{action.error}</Alert>}
      </form>
    </Panel>
  );
}

function JsonField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label>
      {label} (JSON)
      <textarea aria-label={label} rows={4} spellCheck={false} value={value} onChange={(e) => onChange(e.target.value)} placeholder="{}" />
    </label>
  );
}
