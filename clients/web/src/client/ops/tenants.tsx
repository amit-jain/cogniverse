import { useState } from 'react';
import { Alert, Panel, useLoad } from './common';
import { runtimeJson, seg } from './http';

/** Every registered tenant's full ID, sorted. */
export async function knownTenants(signal: AbortSignal): Promise<string[]> {
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

/** A "Tenant" panel whose form hands the typed tenant ID to ``onChoose``,
 * suggesting the registered tenants. */
export function TenantChooser({ action, onChoose }: { action: string; onChoose: (tenant: string) => void }) {
  const known = useLoad(knownTenants, []);
  const [draft, setDraft] = useState('');
  return (
    <Panel title="Tenant">
      <form
        className="inline-form"
        aria-label="Choose tenant"
        onSubmit={(e) => {
          e.preventDefault();
          onChoose(draft.trim());
        }}
      >
        <label>
          Tenant ID
          <input
            required
            list="known-tenants"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="acme:production"
          />
          <datalist id="known-tenants">
            {(known.data ?? []).map((id) => (
              <option key={id} value={id} />
            ))}
          </datalist>
        </label>
        <button type="submit">{action}</button>
        {known.error && <Alert>Tenant suggestions are unavailable: {known.error}</Alert>}
      </form>
    </Panel>
  );
}
