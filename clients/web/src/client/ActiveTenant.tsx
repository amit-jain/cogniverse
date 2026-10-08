import { useEffect, useState } from 'react';
import { TenantNoticeLine } from './ops/tenants';
import { chooseTenant, useTenant } from './tenant';

/** The sidebar's choice of the tenant every view and agent acts for. */
export function ActiveTenant() {
  const { tenant, notice, checking } = useTenant();
  const [draft, setDraft] = useState(tenant);
  useEffect(() => setDraft(tenant), [tenant]);
  return (
    <section className="active-tenant" aria-label="Active tenant">
      <form
        aria-label="Active tenant"
        onSubmit={(e) => {
          e.preventDefault();
          chooseTenant(draft);
        }}
      >
        <label>
          Active tenant
          <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="org:tenant" required />
        </label>
        <button type="submit" disabled={checking}>
          {checking ? 'Checking…' : 'Use'}
        </button>
      </form>
      <p className="current-tenant">
        {tenant ? `Current tenant: ${tenant}` : 'No tenant chosen.'}
      </p>
      {notice && <TenantNoticeLine notice={notice} />}
    </section>
  );
}
