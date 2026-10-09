import { useEffect, useRef, useState } from 'react';
import { chooseTenant, useTenant, type TenantNotice } from '../tenant';
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

/** A "Tenant" panel whose form chooses the active tenant every view shares
 * (``chooseTenant``), suggesting the registered tenants. ``onChoose`` gets
 * the active tenant now and whenever it changes. */
export function TenantChooser({ action, onChoose }: { action: string; onChoose: (tenant: string) => void }) {
  const known = useLoad(knownTenants, []);
  const { tenant, notice, checking } = useTenant();
  const [draft, setDraft] = useState(tenant);
  const report = useRef(onChoose);
  report.current = onChoose;
  useEffect(() => {
    setDraft(tenant);
    if (tenant) report.current(tenant);
  }, [tenant]);
  return (
    <Panel title="Tenant">
      <form
        className="inline-form"
        aria-label="Choose tenant"
        onSubmit={(e) => {
          e.preventDefault();
          chooseTenant(draft).then((active) => {
            // Choosing the tenant already active reloads it in this view.
            if (active && active === tenant) report.current(active);
          });
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
        <button type="submit" disabled={checking}>
          {action}
        </button>
        {known.error && <Alert>Tenant suggestions are unavailable: {known.error}</Alert>}
      </form>
      {notice && <TenantNoticeLine notice={notice} />}
      {!tenant && (
        <p className="muted">
          Every view reads one tenant. Choose a registered tenant; register a new one in the Tenants view (POST
          /admin/tenants) first.
        </p>
      )}
    </Panel>
  );
}

export function TenantNoticeLine({ notice }: { notice: TenantNotice }) {
  return (
    <p className={`alert ${notice.tone}`} role={notice.tone === 'error' ? 'alert' : undefined}>
      {notice.text}
    </p>
  );
}
