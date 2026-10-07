import { useMemo, useState } from 'react';
import { Alert, Panel, useAction, useLoad } from './common';
import { runtimeJson, seg } from './http';
import { Plot } from './Plot';
import { TenantChooser } from './tenants';

export interface AtlasPoint {
  id: string;
  x: number;
  y: number;
  title: string | null;
  text: string | null;
}

export interface Atlas {
  tenant_id: string;
  profile: string;
  schema_name: string;
  embedding_field: string;
  dimensions: number;
  explained_variance: number[];
  without_embedding: number;
  points: AtlasPoint[];
}

/** The points whose title or text holds ``query`` (any case), by title. */
export function matchingPoints(points: AtlasPoint[], query: string): AtlasPoint[] {
  const needle = query.trim().toLowerCase();
  return points
    .filter((p) => !needle || `${p.title ?? ''} ${p.text ?? ''}`.toLowerCase().includes(needle))
    .sort((a, b) => (a.title ?? a.id).localeCompare(b.title ?? b.id) || a.id.localeCompare(b.id));
}

const percent = (share: number) => `${(share * 100).toFixed(1)}%`;

export function EmbeddingAtlasView() {
  const [tenant, setTenant] = useState('');
  return (
    <div className="ops-view">
      <TenantChooser action="Use tenant" onChoose={setTenant} />
      {tenant && <MapPanel key={tenant} tenant={tenant} />}
    </div>
  );
}

function MapPanel({ tenant }: { tenant: string }) {
  const profiles = useLoad(
    async (signal) => {
      const [created, shipped] = await Promise.all([
        runtimeJson<{ profiles: { profile_name: string }[] }>(`/admin/profiles?tenant_id=${seg(tenant)}`, { signal }),
        runtimeJson<{ templates: { profile_name: string }[] }>(`/admin/profile-templates?tenant_id=${seg(tenant)}`, {
          signal,
        }),
      ]);
      return [...new Set([...created.profiles, ...shipped.templates].map((p) => p.profile_name))].sort();
    },
    [tenant],
  );
  const [profile, setProfile] = useState('');
  const [limit, setLimit] = useState('500');
  const [atlas, setAtlas] = useState<Atlas>();
  const action = useAction();
  const chosen = profile || profiles.data?.[0] || '';
  return (
    <>
      <Panel title={`Map documents of ${tenant}`}>
        {profiles.error && <Alert>{profiles.error}</Alert>}
        {profiles.data && (
          <form
            className="inline-form"
            aria-label="Map documents"
            onSubmit={(e) => {
              e.preventDefault();
              action.run(async () => {
                const count = Number(limit);
                if (!Number.isInteger(count) || count < 1) throw new Error('Documents must be a whole number above 0.');
                setAtlas(undefined);
                setAtlas(
                  await runtimeJson<Atlas>(
                    `/admin/tenant/${seg(tenant)}/embeddings/atlas?profile=${seg(chosen)}&limit=${count}`,
                  ),
                );
              });
            }}
          >
            <label>
              Profile
              <select aria-label="Profile" value={chosen} onChange={(e) => setProfile(e.target.value)}>
                {profiles.data.map((name) => (
                  <option key={name}>{name}</option>
                ))}
              </select>
            </label>
            <label>
              Documents
              <input inputMode="numeric" value={limit} onChange={(e) => setLimit(e.target.value)} />
            </label>
            <button type="submit" disabled={action.pending}>
              {action.pending ? 'Mapping…' : 'Show map'}
            </button>
            {action.error && <Alert>{action.error}</Alert>}
          </form>
        )}
      </Panel>
      {atlas && <AtlasPanel atlas={atlas} />}
    </>
  );
}

function AtlasPanel({ atlas }: { atlas: Atlas }) {
  const [query, setQuery] = useState('');
  const shown = useMemo(() => matchingPoints(atlas.points, query), [atlas, query]);
  const data = useMemo(
    () => [
      {
        type: 'scatter',
        mode: 'markers',
        x: atlas.points.map((p) => p.x),
        y: atlas.points.map((p) => p.y),
        text: atlas.points.map((p) => p.title ?? p.id),
        hovertemplate: '%{text}<extra></extra>',
        marker: { size: 8, opacity: 0.8 },
      },
    ],
    [atlas],
  );
  const title = `Profile ${atlas.profile} of ${atlas.tenant_id}`;
  return (
    <Panel title={title}>
      <dl className="facts">
        <dt>Schema</dt>
        <dd>{atlas.schema_name}</dd>
        <dt>Embedding</dt>
        <dd>
          {atlas.embedding_field}, {atlas.dimensions} dimensions
        </dd>
        <dt>Documents mapped</dt>
        <dd>{atlas.points.length}</dd>
        <dt>Without an embedding</dt>
        <dd>{atlas.without_embedding}</dd>
        <dt>Variance shown</dt>
        <dd>
          {percent(atlas.explained_variance[0] ?? 0)} across, {percent(atlas.explained_variance[1] ?? 0)} up
        </dd>
      </dl>
      {atlas.points.length === 0 ? (
        <p className="muted">No documents with an embedding under {atlas.profile}.</p>
      ) : (
        <>
          <Plot
            title={`Documents of ${atlas.profile} by embedding`}
            data={data}
            layout={{ xaxis: { title: { text: 'First principal axis' } }, yaxis: { title: { text: 'Second principal axis' } } }}
          />
          <label>
            Find documents
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="title or text" />
          </label>
          <table aria-label="Mapped documents">
            <thead>
              <tr>
                <th>Title</th>
                <th>Across</th>
                <th>Up</th>
                <th>Text</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((p) => (
                <tr key={p.id}>
                  <td>{p.title ?? p.id}</td>
                  <td>{p.x.toFixed(3)}</td>
                  <td>{p.y.toFixed(3)}</td>
                  <td>{p.text ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Panel>
  );
}
