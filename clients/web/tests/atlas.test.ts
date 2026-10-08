import { describe, expect, it } from 'vitest';
import {
  clusterCounts,
  clusterName,
  kindCounts,
  parseQueries,
  selectedIds,
  selectedPoints,
  umapTraces,
  type UmapAtlas,
  type UmapPoint,
} from '../src/client/ops/atlas';

function point(id: string, cluster: number, text: string | null = null): UmapPoint {
  return { id, x: Number(id.length), y: cluster, title: `${id}.txt`, text, cluster };
}

const atlas: UmapAtlas = {
  tenant_id: 'acme:prod',
  profile: 'notes',
  schema_name: 'document_text_acme_prod',
  embedding_field: 'embedding',
  dimensions: 128,
  without_embedding: 0,
  computed_at: '2026-10-08T10:00:00+00:00',
  generation: 2,
  points: [point('a', 1, 'x'.repeat(130)), point('b', 0), point('c', -1), point('d', 1)],
  clusters: [
    { id: 0, label: 'rivers, canyons', size: 1 },
    { id: 1, label: 'lava, islands', size: 2 },
  ],
  queries: [{ label: 'Query 1', text: 'rivers', x: 0.5, y: 0.25, similar: [] }],
};

describe('parseQueries', () => {
  it('takes one query per non-empty line', () => {
    expect(parseQueries(' rivers \n\n lava flows\n')).toEqual({ queries: ['rivers', 'lava flows'] });
  });

  it('refuses more than ten queries', () => {
    expect(parseQueries(Array(11).fill('q').join('\n'))).toEqual({
      error: 'Place at most 10 queries, one per line.',
    });
  });
});

describe('umapTraces', () => {
  it('draws each cluster by id, the unclustered last, then the queries', () => {
    const traces = umapTraces(atlas, false);
    expect(traces.map((t) => [t.name, t.x])).toEqual([
      ['rivers, canyons', [1]],
      ['lava, islands', [1, 1]],
      ['Unclustered', [1]],
      ['Queries', [0.5]],
    ]);
    expect((traces[1].customdata as unknown[][])[0]).toEqual(['a.txt', 'lava, islands', `${'x'.repeat(120)}…`, 'a', 'Document']);
    expect(traces[3].text).toEqual(['Query 1']);
  });

  it('lays the density of every document beneath the points', () => {
    const [density, ...rest] = umapTraces(atlas, true);
    expect([density.type, density.x, density.y]).toEqual(['histogram2dcontour', [1, 1, 1, 1], [1, 0, -1, 1]]);
    expect(rest).toHaveLength(4);
  });
});

describe('selection', () => {
  it('names the documents and queries a lasso holds', () => {
    const ids = selectedIds([
      { customdata: ['a.txt', 'lava, islands', '', 'a', 'Document'] },
      { customdata: ['Query 1', 'Query', '', 'Query 1', 'Query'] },
      { customdata: undefined },
    ]);
    expect([...ids]).toEqual(['a', 'Query 1']);
    expect(selectedPoints(atlas, ids).map((p) => p.id)).toEqual(['a']);
    expect(selectedPoints(atlas, null).map((p) => p.id)).toEqual(['a', 'b', 'c', 'd']);
    expect(kindCounts(atlas, ids)).toEqual([
      { label: 'Documents', value: 1 },
      { label: 'Queries', value: 1 },
    ]);
  });

  it('counts the selected documents per cluster, leaving out empty clusters', () => {
    expect(clusterCounts(atlas, selectedPoints(atlas, new Set(['a', 'c', 'd'])))).toEqual([
      { label: 'lava, islands', value: 2 },
      { label: 'Unclustered', value: 1 },
    ]);
  });

  it('names a cluster the map does not list as unclustered', () => {
    expect([clusterName(atlas, 0), clusterName(atlas, -1), clusterName(atlas, 7)]).toEqual([
      'rivers, canyons',
      'Unclustered',
      'Unclustered',
    ]);
  });
});
