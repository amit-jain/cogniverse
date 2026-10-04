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
  const [selectedOrg, setSelectedOrg] = useState<string>();
  const [tenantsVersion, setTenantsVersion] = useState(0);
  const [notice, setNotice] = useState('');
  const orgIds = (orgs.data ?? []).map((org) => org.org_id);
  const activeOrg = selectedOrg && orgIds.includes(selectedOrg) ? selectedOrg : orgIds[0];
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
        <TenantsPanel key={`${activeOrg}-${tenantsVersion}`} orgId={activeOrg} onChanged={changed} />
      )}
      <CreateTenant orgIds={orgIds} defaultOrg={activeOrg} onCreated={() => changed()} />
    </div>
  );
}

function TenantsPanel({ orgId, onChanged }: { orgId: string; onChanged: (notice: string) => void }) {
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
  const [orgId, setOrgId] = useState('');
  const [tenantName, setTenantName] = useState('');
  const [createdBy, setCreatedBy] = useState('admin');
  const [done, setDone] = useState('');
  const action = useAction();
  const org = orgId || defaultOrg || '';
  return (
    <Panel title="New tenant">
      <form
        className="inline-form"
        aria-label="Create tenant"
        onSubmit={(e) => {
          e.preventDefault();
          setDone('');
          action.run(async () => {
            const tenant = await runtimeJson<Tenant>('/admin/tenants', {
              method: 'POST',
              body: { tenant_id: `${org.trim()}:${tenantName.trim()}`, created_by: createdBy.trim() },
            });
            setDone(
              `Created ${tenant.tenant_full_id} with schemas ${tenant.schemas_deployed.join(', ')}.`,
            );
            setTenantName('');
            onCreated();
          });
        }}
      >
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
          <input required value={tenantName} onChange={(e) => setTenantName(e.target.value)} placeholder="production" />
        </label>
        <label>
          Created by
          <input required value={createdBy} onChange={(e) => setCreatedBy(e.target.value)} />
        </label>
        <button type="submit" disabled={action.pending}>
          {action.pending ? 'Creating and deploying schemas…' : 'Create tenant'}
        </button>
        {action.error && <Alert>{action.error}</Alert>}
        {done && <Alert tone="ok">{done}</Alert>}
      </form>
      <p className="muted">
        A new organization is created when the name is new. The tenant gets the runtime's base
        schemas.
      </p>
    </Panel>
  );
}
