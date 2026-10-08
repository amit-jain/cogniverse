/** The UMAP map the runtime lays a profile's documents out on, and the
 * figures the Embedding atlas view draws from it. */

export interface UmapPoint {
  id: string;
  x: number;
  y: number;
  title: string | null;
  text: string | null;
  /** Cluster id, ``UNCLUSTERED`` for a document in none. */
  cluster: number;
}

export interface UmapCluster {
  id: number;
  label: string;
  size: number;
}

export interface SimilarDocument {
  id: string;
  title: string | null;
  similarity: number;
}

export interface QueryPoint {
  label: string;
  text: string;
  x: number;
  y: number;
  similar: SimilarDocument[];
}

export interface UmapAtlas {
  tenant_id: string;
  profile: string;
  schema_name: string;
  embedding_field: string;
  dimensions: number;
  without_embedding: number;
  computed_at: string;
  generation: number;
  points: UmapPoint[];
  clusters: UmapCluster[];
  queries: QueryPoint[];
}

export const UNCLUSTERED = -1;
/** The most queries one map places. */
export const MAX_QUERIES = 10;
const HOVER_TEXT_CHARS = 120;

/** The queries typed one per line, or why they cannot be placed. */
export function parseQueries(text: string): { queries: string[] } | { error: string } {
  const queries = text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  if (queries.length > MAX_QUERIES) return { error: `Place at most ${MAX_QUERIES} queries, one per line.` };
  return { queries };
}

/** A cluster's name as the map shows it. */
export function clusterName(atlas: UmapAtlas, cluster: number): string {
  return atlas.clusters.find((c) => c.id === cluster)?.label ?? 'Unclustered';
}

function preview(text: string | null): string {
  if (!text) return '';
  return text.length > HOVER_TEXT_CHARS ? `${text.slice(0, HOVER_TEXT_CHARS)}…` : text;
}

/** Each cluster's points (clusters by id, the unclustered last), then the
 * queries, as Plotly traces whose hover names the point, its kind and
 * cluster and a preview of its text. ``density`` adds a density contour of
 * the documents beneath them. */
export function umapTraces(atlas: UmapAtlas, density: boolean): Record<string, unknown>[] {
  const order = [...atlas.clusters.map((c) => c.id), UNCLUSTERED];
  const traces: Record<string, unknown>[] = [];
  if (density)
    traces.push({
      type: 'histogram2dcontour',
      name: 'Density',
      x: atlas.points.map((p) => p.x),
      y: atlas.points.map((p) => p.y),
      showscale: false,
      ncontours: 12,
      colorscale: 'Blues',
      hoverinfo: 'skip',
    });
  for (const cluster of order) {
    const points = atlas.points.filter((p) => p.cluster === cluster);
    if (points.length === 0) continue;
    const name = clusterName(atlas, cluster);
    traces.push({
      type: 'scatter',
      mode: 'markers',
      name,
      x: points.map((p) => p.x),
      y: points.map((p) => p.y),
      customdata: points.map((p) => [p.title ?? p.id, name, preview(p.text), p.id, 'Document']),
      hovertemplate: '<b>%{customdata[0]}</b><br>%{customdata[4]} · %{customdata[1]}<br>%{customdata[2]}<extra></extra>',
      marker: { size: 8, opacity: 0.8 },
    });
  }
  if (atlas.queries.length)
    traces.push({
      type: 'scatter',
      mode: 'markers+text',
      name: 'Queries',
      x: atlas.queries.map((q) => q.x),
      y: atlas.queries.map((q) => q.y),
      text: atlas.queries.map((q) => q.label),
      textposition: 'top center',
      customdata: atlas.queries.map((q) => [q.label, 'Query', preview(q.text), q.label, 'Query']),
      hovertemplate: '<b>%{customdata[0]}</b><br>%{customdata[2]}<extra></extra>',
      marker: { size: 14, symbol: 'star' },
    });
  return traces;
}

/** The ids a selection holds: the documents and queries named by each
 * selected point's ``customdata``. */
export function selectedIds(points: { customdata?: unknown }[]): Set<string> {
  return new Set(
    points.flatMap((point) => {
      const data = point.customdata;
      return Array.isArray(data) && typeof data[3] === 'string' ? [data[3]] : [];
    }),
  );
}

/** The documents in ``selection`` (every document when there is none). */
export function selectedPoints(atlas: UmapAtlas, selection: Set<string> | null): UmapPoint[] {
  return selection ? atlas.points.filter((p) => selection.has(p.id)) : atlas.points;
}

/** How many of ``points`` each cluster holds, clusters by id, the
 * unclustered last; clusters without a point are left out. */
export function clusterCounts(atlas: UmapAtlas, points: UmapPoint[]): { label: string; value: number }[] {
  return [...atlas.clusters.map((c) => c.id), UNCLUSTERED]
    .map((cluster) => ({
      label: clusterName(atlas, cluster),
      value: points.filter((p) => p.cluster === cluster).length,
    }))
    .filter((entry) => entry.value > 0);
}

/** How many documents and queries ``selection`` holds. */
export function kindCounts(atlas: UmapAtlas, selection: Set<string> | null): { label: string; value: number }[] {
  const queries = selection ? atlas.queries.filter((q) => selection.has(q.label)) : atlas.queries;
  return [
    { label: 'Documents', value: selectedPoints(atlas, selection).length },
    { label: 'Queries', value: queries.length },
  ];
}
