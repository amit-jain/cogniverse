import { useState } from 'react';
import { Alert, Panel, useLoad } from './common';
import { runtimeJson, seg } from './http';
import { Bars, LookbackSelect, delta, percent } from './metrics';
import { TenantChooser } from './tenants';

interface DatasetComparison {
  queries_dataset: string | null;
  rows: number;
  avg_latency_delta_ms: number | null;
  avg_tokens_delta: number | null;
  avg_judge_delta: number | null;
}

interface Comparison {
  ab_id: string | null;
  query: string | null;
  queries_dataset: string | null;
  latency_delta_ms: number | null;
  tokens_delta: number | null;
  judge_delta: number | null;
  with_rlm_was_fallback: boolean;
  start_time: string | null;
}

interface RlmAb {
  rows: number;
  avg_latency_delta_ms: number | null;
  avg_tokens_delta: number | null;
  avg_judge_delta: number | null;
  fallback_rate: number | null;
  per_dataset: DatasetComparison[];
  comparisons: Comparison[];
}

export function RlmAbView() {
  const [tenant, setTenant] = useState('');
  const [lookback, setLookback] = useState(24);
  return (
    <div className="ops-view">
      <TenantChooser action="Show comparisons" onChoose={setTenant} />
      {tenant && <Comparisons key={`${tenant}-${lookback}`} tenant={tenant} lookback={lookback} onLookback={setLookback} />}
    </div>
  );
}

function Comparisons({
  tenant,
  lookback,
  onLookback,
}: {
  tenant: string;
  lookback: number;
  onLookback: (hours: number) => void;
}) {
  const ab = useLoad(
    (signal) =>
      runtimeJson<RlmAb>(`/admin/tenant/${seg(tenant)}/telemetry/rlm-ab?lookback_hours=${lookback}`, { signal }),
    [tenant, lookback],
  );
  const data = ab.data;
  return (
    <>
      <Panel title={`RLM A/B comparisons of ${tenant}`} actions={<button onClick={ab.reload}>Refresh</button>}>
        <div className="inline-form">
          <LookbackSelect value={lookback} onChange={onLookback} />
        </div>
        {ab.error && <Alert>{ab.error}</Alert>}
        {data && data.rows === 0 && (
          <p className="muted">
            No comparisons in this window. Run <code>cogniverse-optim --mode ab-compare</code> for this tenant to
            record some.
          </p>
        )}
        {data && data.rows > 0 && (
          <dl className="facts" aria-label="Comparison averages">
            <dt>Comparisons</dt>
            <dd>{data.rows}</dd>
            <dt>Latency change with RLM</dt>
            <dd>{delta(data.avg_latency_delta_ms)} ms</dd>
            <dt>Token change with RLM</dt>
            <dd>{delta(data.avg_tokens_delta)}</dd>
            <dt>Judge score change with RLM</dt>
            <dd>{delta(data.avg_judge_delta, 3)}</dd>
            <dt>RLM fell back</dt>
            <dd>{data.fallback_rate === null ? '—' : percent(data.fallback_rate)}</dd>
          </dl>
        )}
      </Panel>
      {data && data.per_dataset.length > 0 && (
        <Panel title="Per dataset">
          <table aria-label="Comparisons per dataset">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Comparisons</th>
                <th>Latency change (ms)</th>
                <th>Token change</th>
                <th>Judge change</th>
              </tr>
            </thead>
            <tbody>
              {data.per_dataset.map((row) => (
                <tr key={row.queries_dataset ?? ''}>
                  <td>{row.queries_dataset ?? '—'}</td>
                  <td>{row.rows}</td>
                  <td>{delta(row.avg_latency_delta_ms)}</td>
                  <td>{delta(row.avg_tokens_delta)}</td>
                  <td>{delta(row.avg_judge_delta, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Bars
            title="Average latency change per dataset"
            entries={data.per_dataset.map((row) => ({
              label: row.queries_dataset ?? '—',
              value: row.avg_latency_delta_ms ?? 0,
            }))}
            format={(value) => `${delta(value)} ms`}
          />
        </Panel>
      )}
      {data && data.comparisons.length > 0 && (
        <Panel title="Comparisons">
          <table aria-label="Comparisons">
            <thead>
              <tr>
                <th>When</th>
                <th>Query</th>
                <th>Dataset</th>
                <th>Latency change (ms)</th>
                <th>Token change</th>
                <th>Judge change</th>
                <th>RLM fell back</th>
              </tr>
            </thead>
            <tbody>
              {data.comparisons.map((row, index) => (
                <tr key={row.ab_id ?? index}>
                  <td>{row.start_time ? new Date(row.start_time).toLocaleString() : '—'}</td>
                  <td>{row.query ?? '—'}</td>
                  <td>{row.queries_dataset ?? '—'}</td>
                  <td>{delta(row.latency_delta_ms)}</td>
                  <td>{delta(row.tokens_delta)}</td>
                  <td>{delta(row.judge_delta, 3)}</td>
                  <td>{row.with_rlm_was_fallback ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}
    </>
  );
}
