import { useMemo, useState } from 'react';
import {
  clusterCounts,
  clusterName,
  isExportAtlas,
  kindCounts,
  parseQueries,
  selectedIds,
  selectedPoints,
  umapTraces,
  type ExportAtlas,
  type UmapAtlas,
} from './atlas';
import { Alert, Panel, useAction, useLoad } from './common';
import { runtimeJson, seg } from './http';
import { Bars } from './metrics';
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
export function matchingPoints<P extends AtlasPoint>(points: P[], query: string): P[] {
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
  const [projection, setProjection] = useState<'pca' | 'umap' | 'export'>('pca');
  const [exportFile, setExportFile] = useState<File | null>(null);
  const [queries, setQueries] = useState('');
  const [atlas, setAtlas] = useState<Atlas>();
  const [umap, setUmap] = useState<UmapAtlas>();
  const action = useAction();
  const chosen = profile || profiles.data?.[0] || '';
  const umapPath = `/admin/tenant/${seg(tenant)}/embeddings/atlas/umap`;
  const loadUmap = async (count: number, placed: string[], recompute: boolean) => {
    if (recompute) await runtimeJson(`${umapPath}?profile=${seg(chosen)}`, { method: 'DELETE' });
    setUmap(
      await runtimeJson<UmapAtlas>(umapPath, {
        method: 'POST',
        body: { profile: chosen, limit: count, queries: placed },
      }),
    );
  };
  const show = (recompute: boolean) =>
    action.run(async () => {
      setAtlas(undefined);
      setUmap(undefined);
      if (projection === 'export') {
        if (!exportFile) throw new Error('Choose an embedding export file (.parquet) first.');
        const body = new FormData();
        body.append('file', exportFile);
        setUmap(
          await runtimeJson<ExportAtlas>(`/admin/tenant/${seg(tenant)}/embeddings/atlas/export`, {
            method: 'POST',
            body,
          }),
        );
        return;
      }
      const count = Number(limit);
      if (!Number.isInteger(count) || count < 1) throw new Error('Documents must be a whole number above 0.');
      if (projection === 'pca') {
        setAtlas(
          await runtimeJson<Atlas>(`/admin/tenant/${seg(tenant)}/embeddings/atlas?profile=${seg(chosen)}&limit=${count}`),
        );
        return;
      }
      const parsed = parseQueries(queries);
      if ('error' in parsed) throw new Error(parsed.error);
      await loadUmap(count, parsed.queries, recompute);
    });
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
              show(false);
            }}
          >
            {projection !== 'export' && (
              <>
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
              </>
            )}
            <label>
              Projection
              <select
                value={projection}
                onChange={(e) => setProjection(e.target.value as 'pca' | 'umap' | 'export')}
              >
                <option value="pca">PCA, read live</option>
                <option value="umap">UMAP with clusters</option>
                <option value="export">Exported file</option>
              </select>
            </label>
            {projection === 'export' && (
              <label>
                Embedding export file
                <input
                  type="file"
                  accept=".parquet"
                  onChange={(e) => setExportFile(e.target.files?.[0] ?? null)}
                />
              </label>
            )}
            {projection === 'umap' && (
              <label>
                Queries (one per line)
                <textarea rows={3} value={queries} onChange={(e) => setQueries(e.target.value)} />
              </label>
            )}
            <button type="submit" disabled={action.pending}>
              {action.pending ? 'Mapping…' : 'Show map'}
            </button>
            {projection === 'umap' && (
              <button type="button" disabled={action.pending} onClick={() => show(true)}>
                Recompute
              </button>
            )}
            {action.error && <Alert>{action.error}</Alert>}
          </form>
        )}
      </Panel>
      {atlas && <AtlasPanel atlas={atlas} />}
      {umap && <UmapPanel key={`${umap.generation}-${umap.computed_at}`} atlas={umap} />}
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

const when = (iso: string) => new Date(iso).toLocaleString();

function UmapPanel({ atlas }: { atlas: UmapAtlas }) {
  const [density, setDensity] = useState(false);
  const [selection, setSelection] = useState<Set<string> | null>(null);
  const [query, setQuery] = useState('');
  const data = useMemo(() => umapTraces(atlas, density), [atlas, density]);
  const chosen = useMemo(() => selectedPoints(atlas, selection), [atlas, selection]);
  const shown = useMemo(() => {
    const ids = new Set(chosen.map((p) => p.id));
    return matchingPoints(atlas.points, query).filter((p) => ids.has(p.id));
  }, [atlas, chosen, query]);
  const exported = isExportAtlas(atlas) ? atlas : null;
  const fromFile = exported?.layout === 'file';
  const layout = useMemo(
    () => ({
      dragmode: 'lasso',
      xaxis: { title: { text: fromFile ? 'x' : 'UMAP 1' } },
      yaxis: { title: { text: fromFile ? 'y' : 'UMAP 2' } },
    }),
    [fromFile],
  );
  const cluster = (id: number) => clusterName(atlas, id);
  const plotTitle = exported ? `Documents of ${exported.file_name}` : `Documents of ${atlas.profile} by UMAP`;
  return (
    <>
      <Panel
        title={
          exported
            ? `Map of ${exported.file_name} for ${atlas.tenant_id}`
            : `UMAP map of ${atlas.profile} for ${atlas.tenant_id}`
        }
      >
        <dl className="facts" aria-label="Map facts">
          {exported && (
            <>
              <dt>File</dt>
              <dd>
                {exported.file_name}, {exported.rows} rows
              </dd>
              <dt>Places</dt>
              <dd>{fromFile ? "The file's x/y columns" : 'UMAP over the embeddings'}</dd>
              <dt>Encoder profile</dt>
              <dd>{atlas.profile || 'not recorded'}</dd>
            </>
          )}
          <dt>Schema</dt>
          <dd>{atlas.schema_name || 'not recorded'}</dd>
          <dt>Embedding</dt>
          <dd>{atlas.embedding_field ? `${atlas.embedding_field}, ${atlas.dimensions} dimensions` : 'none'}</dd>
          <dt>Documents mapped</dt>
          <dd>{atlas.points.length}</dd>
          <dt>Without an embedding</dt>
          <dd>{atlas.without_embedding}</dd>
          <dt>Clusters</dt>
          <dd>{atlas.clusters.length}</dd>
          <dt>Queries placed</dt>
          <dd>{atlas.queries.length}</dd>
          <dt>Laid out</dt>
          <dd>{when(atlas.computed_at)}</dd>
        </dl>
        <label>
          <input type="checkbox" checked={density} onChange={(e) => setDensity(e.target.checked)} /> Density
        </label>
        <Plot
          title={plotTitle}
          data={data}
          layout={layout}
          onSelect={(points) => setSelection(points ? selectedIds(points) : null)}
        />
        <p className="muted" aria-label="Selection">
          {selection
            ? `Selection: ${chosen.length} documents of ${atlas.points.length}.`
            : 'Draw a lasso or box on the map to select documents.'}
          {selection && (
            <button type="button" onClick={() => setSelection(null)}>
              Clear selection
            </button>
          )}
        </p>
        <div className="chart-grid">
          <Bars title="Documents per cluster" entries={clusterCounts(atlas, chosen)} format={String} />
          <Bars title="Points by kind" entries={kindCounts(atlas, selection)} format={String} />
        </div>
        <Plot
          title="Text length"
          data={[{ type: 'histogram', x: chosen.map((p) => (p.text ?? '').length), name: 'Documents' }]}
          layout={{ xaxis: { title: { text: 'Characters' } }, yaxis: { title: { text: 'Documents' } } }}
        />
        <label>
          Find documents
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="title or text" />
        </label>
        <table aria-label="Mapped documents">
          <thead>
            <tr>
              <th>Title</th>
              <th>Cluster</th>
              <th>Across</th>
              <th>Up</th>
              <th>Text</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((p) => (
              <tr key={p.id}>
                <td>{p.title ?? p.id}</td>
                <td>{cluster(p.cluster)}</td>
                <td>{p.x.toFixed(3)}</td>
                <td>{p.y.toFixed(3)}</td>
                <td>{p.text ?? ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
      {atlas.queries.length > 0 && (
        <Panel title="Query analysis">
          {atlas.queries.map((q) => (
            <section key={q.label} aria-label={q.label}>
              <h3>
                {q.label}: {q.text}
              </h3>
              <ol aria-label={`Documents most similar to ${q.label}`}>
                {q.similar.map((d) => (
                  <li key={d.id}>
                    {d.title ?? d.id} (similarity {d.similarity.toFixed(3)})
                  </li>
                ))}
              </ol>
            </section>
          ))}
        </Panel>
      )}
    </>
  );
}
