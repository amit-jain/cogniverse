import { useState } from 'react';
import { Alert, ConfirmDelete, Panel, formatMillis, useAction, useLoad } from './common';
import { runtimeJson, seg } from './http';

interface Organization {
  org_id: string;
  org_name: string;
  created_at: number;
  created_by: string;
  status: string;
  tenant_count: number;
}

interface Tenant {
  tenant_full_id: string;
  org_id: string;
  tenant_name: string;
  created_at: number;
  created_by: string;
  status: string;
  schemas_deployed: string[];
}

/** ``n`` with ``noun``, plural unless one. */
function counted(n: number, noun: string): string {
  return `${n} ${noun}${n === 1 ? '' : 's'}`;
}

interface RouterTiers {
  tiers: string[];
  default: string;
}

export function TenantsView() {
  const orgs = useLoad(
    (signal) =>
      runtimeJson<{ organizations: Organization[] }>('/admin/organizations', { signal }).then(
        (body) => body.organizations,
      ),
    [],
  );
  // Undefined until one is chosen (the first is shown); null once the chosen
  // one was deleted, so none is shown.
  const [selectedOrg, setSelectedOrg] = useState<string | null>();
  const [tenantsVersion, setTenantsVersion] = useState(0);
  const [notice, setNotice] = useState('');
  const orgIds = (orgs.data ?? []).map((org) => org.org_id);
  const activeOrg =
    selectedOrg === null ? undefined : selectedOrg && orgIds.includes(selectedOrg) ? selectedOrg : orgIds[0];
  const deleted = (orgId: string) => {
    if (orgId === activeOrg) setSelectedOrg(null);
  };
  const changed = (message = '') => {
    setNotice(message);
    orgs.reload();
    setTenantsVersion((n) => n + 1);
  };

  return (
    <div className="ops-view">
      {notice && <Alert tone="ok">{notice}</Alert>}
      <Panel title="Organizations" actions={<button onClick={orgs.reload}>Refresh</button>}>
        {orgs.error && <Alert>{orgs.error}</Alert>}
        {orgs.data && orgs.data.length === 0 && <p className="muted">No organizations yet.</p>}
        {orgs.data && orgs.data.length > 0 && (
          <p className="muted" aria-label="Organization count">
            {counted(orgs.data.length, 'organization')},{' '}
            {counted(
              orgs.data.reduce((sum, org) => sum + org.tenant_count, 0),
              'tenant',
            )}{' '}
            in all.
          </p>
        )}
        {orgs.data && orgs.data.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>Organization</th>
                <th>Name</th>
                <th>Status</th>
                <th>Tenants</th>
                <th>Created</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {orgs.data.map((org) => (
                <tr key={org.org_id} className={org.org_id === activeOrg ? 'selected' : undefined}>
                  <td>
                    <button className="link" onClick={() => setSelectedOrg(org.org_id)}>
                      {org.org_id}
                    </button>
                  </td>
                  <td>{org.org_name}</td>
                  <td>{org.status}</td>
                  <td>{org.tenant_count}</td>
                  <td>
                    {formatMillis(org.created_at)} by {org.created_by}
                  </td>
                  <td>
                    <ConfirmDelete
                      name={org.org_id}
                      what="organization"
                      onDelete={async () => {
                        const result = await runtimeJson<{ tenants_deleted: number }>(
                          `/admin/organizations/${seg(org.org_id)}`,
                          { method: 'DELETE' },
                        );
                        deleted(org.org_id);
                        changed(
                          `Deleted organization ${org.org_id} and its ${result.tenants_deleted} tenant(s).`,
                        );
                      }}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <CreateOrganization onCreated={() => changed()} />
      </Panel>
      {activeOrg && (
        <TenantsPanel
          key={`${activeOrg}-${tenantsVersion}`}
          orgId={activeOrg}
          onChanged={changed}
          onOrgDeleted={() => deleted(activeOrg)}
        />
      )}
      <CreateTenant orgIds={orgIds} defaultOrg={activeOrg} onCreated={() => changed()} />
    </div>
  );
}

function TenantsPanel({
  orgId,
  onChanged,
  onOrgDeleted,
}: {
  orgId: string;
  onChanged: (notice: string) => void;
  onOrgDeleted: () => void;
}) {
  const tenants = useLoad(
    (signal) =>
      runtimeJson<{ tenants: Tenant[] }>(`/admin/organizations/${seg(orgId)}/tenants`, {
        signal,
      }).then((body) => body.tenants),
    [orgId],
  );
  const tiers = useLoad((signal) => runtimeJson<RouterTiers>('/admin/router-tiers', { signal }), []);
  return (
    <Panel title={`Tenants of ${orgId}`} actions={<button onClick={tenants.reload}>Refresh</button>}>
      {tenants.error && <Alert>{tenants.error}</Alert>}
      {tiers.error && <Alert>{tiers.error}</Alert>}
      {tenants.data && tenants.data.length === 0 && <p className="muted">No tenants in {orgId}.</p>}
      {tenants.data && tenants.data.length > 0 && (
        <p className="muted" aria-label="Tenant count">
          {counted(tenants.data.length, 'tenant')} in {orgId}.
        </p>
      )}
      {tenants.data && tenants.data.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Tenant</th>
              <th>Status</th>
              <th>Schemas</th>
              <th>Router tier</th>
              <th>Created</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {tenants.data.map((tenant) => (
              <tr key={tenant.tenant_full_id}>
                <td>{tenant.tenant_full_id}</td>
                <td>{tenant.status}</td>
                <td>{tenant.schemas_deployed.join(', ') || '—'}</td>
                <td>
                  {tiers.data && <TierPicker tenantId={tenant.tenant_full_id} tiers={tiers.data.tiers} />}
                </td>
                <td>
                  {formatMillis(tenant.created_at)} by {tenant.created_by}
                </td>
                <td>
                  <ConfirmDelete
                    name={tenant.tenant_full_id}
                    what="tenant"
                    onDelete={async () => {
                      const result = await runtimeJson<{ organization_deleted?: boolean }>(
                        `/admin/tenants/${seg(tenant.tenant_full_id)}`,
                        { method: 'DELETE' },
                      );
                      if (result.organization_deleted) onOrgDeleted();
                      onChanged(
                        result.organization_deleted
                          ? `Deleted tenant ${tenant.tenant_full_id} and organization ${orgId}, which had no tenants left.`
                          : `Deleted tenant ${tenant.tenant_full_id}.`,
                      );
                    }}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

function TierPicker({ tenantId, tiers }: { tenantId: string; tiers: string[] }) {
  const current = useLoad(
    (signal) =>
      runtimeJson<{ tier: string }>(`/admin/tenants/${seg(tenantId)}/tier`, { signal }).then(
        (body) => body.tier,
      ),
    [tenantId],
  );
  const action = useAction();
  if (current.error) return <Alert>{current.error}</Alert>;
  if (current.data === undefined) return <span className="muted">…</span>;
  return (
    <>
      <select
        aria-label={`Router tier for ${tenantId}`}
        value={current.data}
        disabled={action.pending}
        onChange={(e) => {
          const tier = e.target.value;
          action.run(async () => {
            await runtimeJson(`/admin/tenants/${seg(tenantId)}/tier`, {
              method: 'PUT',
              body: { tier },
            });
            current.reload();
          });
        }}
      >
        {tiers.map((tier) => (
          <option key={tier}>{tier}</option>
        ))}
      </select>
      {action.error && <Alert>{action.error}</Alert>}
    </>
  );
}

function CreateOrganization({ onCreated }: { onCreated: () => void }) {
  const [orgId, setOrgId] = useState('');
  const [orgName, setOrgName] = useState('');
  const [createdBy, setCreatedBy] = useState('admin');
  const [done, setDone] = useState('');
  const action = useAction();
  return (
    <form
      className="inline-form"
      aria-label="Create organization"
      onSubmit={(e) => {
        e.preventDefault();
        setDone('');
        action.run(async () => {
          const org = await runtimeJson<Organization>('/admin/organizations', {
            method: 'POST',
            body: { org_id: orgId.trim(), org_name: orgName.trim(), created_by: createdBy.trim() },
          });
          setDone(`Created organization ${org.org_id}.`);
          setOrgId('');
          setOrgName('');
          onCreated();
        });
      }}
    >
      <h3>New organization</h3>
      <label>
        Organization ID
        <input required value={orgId} onChange={(e) => setOrgId(e.target.value)} placeholder="acme" />
      </label>
      <label>
        Name
        <input required value={orgName} onChange={(e) => setOrgName(e.target.value)} placeholder="Acme Corporation" />
      </label>
      <label>
        Created by
        <input required value={createdBy} onChange={(e) => setCreatedBy(e.target.value)} />
      </label>
      <button type="submit" disabled={action.pending}>
        {action.pending ? 'Creating…' : 'Create organization'}
      </button>
      {action.error && <Alert>{action.error}</Alert>}
      {done && <Alert tone="ok">{done}</Alert>}
    </form>
  );
}

function CreateTenant({
  orgIds,
  defaultOrg,
  onCreated,
}: {
  orgIds: string[];
  defaultOrg?: string;
  onCreated: () => void;
}) {
  const bases = useLoad(
    (signal) => runtimeJson<{ schemas: string[]; default: string[] }>('/admin/base-schemas', { signal }),
    [],
  );
  const [orgId, setOrgId] = useState('');
  const [tenantName, setTenantName] = useState('');
  const [createdBy, setCreatedBy] = useState('admin');
  const [chosen, setChosen] = useState<string[]>();
  const [done, setDone] = useState('');
  const action = useAction();
  const org = orgId || defaultOrg || '';
  const selected = chosen ?? bases.data?.default ?? [];
  const toggle = (schema: string, on: boolean) =>
    setChosen(on ? [...selected, schema] : selected.filter((name) => name !== schema));
  return (
    <Panel title="New tenant">
      <form
        className="stacked-form"
        aria-label="Create tenant"
        onSubmit={(e) => {
          e.preventDefault();
          setDone('');
          action.run(async () => {
            if (!selected.length) throw new Error('Choose at least one base schema.');
            const tenant = await runtimeJson<Tenant>('/admin/tenants', {
              method: 'POST',
              body: {
                tenant_id: `${org.trim()}:${tenantName.trim()}`,
                created_by: createdBy.trim(),
                base_schemas: (bases.data?.schemas ?? []).filter((name) => selected.includes(name)),
              },
            });
            setDone(
              `Created ${tenant.tenant_full_id} with schemas ${tenant.schemas_deployed.join(', ')}.`,
            );
            setTenantName('');
            onCreated();
          });
        }}
      >
        <div className="inline-form">
          <label>
            Organization
            <input
              required
              list="known-orgs"
              value={org}
              onChange={(e) => setOrgId(e.target.value)}
              placeholder="acme"
            />
            <datalist id="known-orgs">
              {orgIds.map((id) => (
                <option key={id} value={id} />
              ))}
            </datalist>
          </label>
          <label>
            Tenant name
            <input
              required
              value={tenantName}
              onChange={(e) => setTenantName(e.target.value)}
              placeholder="production"
            />
          </label>
          <label>
            Created by
            <input required value={createdBy} onChange={(e) => setCreatedBy(e.target.value)} />
          </label>
        </div>
        <fieldset>
          <legend>Base schemas</legend>
          {bases.error && <Alert>{bases.error}</Alert>}
          {(bases.data?.schemas ?? []).map((schema) => (
            <label key={schema} className="check">
              <input
                type="checkbox"
                checked={selected.includes(schema)}
                onChange={(e) => toggle(schema, e.target.checked)}
              />
              {schema}
            </label>
          ))}
        </fieldset>
        <button type="submit" disabled={action.pending || !bases.data}>
          {action.pending ? 'Creating and deploying schemas…' : 'Create tenant'}
        </button>
        {action.error && <Alert>{action.error}</Alert>}
        {done && <Alert tone="ok">{done}</Alert>}
      </form>
      <p className="muted">
        A new organization is created when the name is new. The tenant gets the checked base schemas; the
        runtime's defaults are checked to start with.
      </p>
    </Panel>
  );
}
