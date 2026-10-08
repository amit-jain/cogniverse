import { useState } from 'react';
import { Alert, Panel, useLoad } from './common';
import { runtimeJson, seg } from './http';
import { Bars, LookbackSelect, percent } from './metrics';
import { TenantChooser } from './tenants';

interface ModalityMetrics {
  modality: string;
  count: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  success_rate: number;
}

const ms = (value: number) => `${value.toFixed(1)} ms`;

export function ProfileMetricsView() {
  const [tenant, setTenant] = useState('');
  const [lookback, setLookback] = useState(24);
  return (
    <div className="ops-view">
      <TenantChooser action="Show metrics" onChoose={setTenant} />
      {tenant && <Metrics key={`${tenant}-${lookback}`} tenant={tenant} lookback={lookback} onLookback={setLookback} />}
    </div>
  );
}

function Metrics({
  tenant,
  lookback,
  onLookback,
}: {
  tenant: string;
  lookback: number;
  onLookback: (hours: number) => void;
}) {
  const metrics = useLoad(
    (signal) =>
      runtimeJson<{ modalities: ModalityMetrics[] }>(
        `/admin/tenant/${seg(tenant)}/telemetry/profile-selection?lookback_hours=${lookback}`,
        { signal },
      ).then((body) => body.modalities),
    [tenant, lookback],
  );
  const modalities = metrics.data ?? [];
  return (
    <Panel title={`Profile selections of ${tenant}`} actions={<button onClick={metrics.reload}>Refresh</button>}>
      <div className="inline-form">
        <LookbackSelect value={lookback} onChange={onLookback} />
      </div>
      {metrics.error && <Alert>{metrics.error}</Alert>}
      {metrics.data && modalities.length === 0 && (
        <p className="muted">No profile selections in this window.</p>
      )}
      {modalities.length > 0 && (
        <>
          <table aria-label="Selections by modality">
            <thead>
              <tr>
                <th>Modality</th>
                <th>Selections</th>
                <th>P50</th>
                <th>P95</th>
                <th>P99</th>
                <th>Succeeded</th>
              </tr>
            </thead>
            <tbody>
              {modalities.map((row) => (
                <tr key={row.modality}>
                  <td>{row.modality}</td>
                  <td>{row.count}</td>
                  <td>{ms(row.p50_ms)}</td>
                  <td>{ms(row.p95_ms)}</td>
                  <td>{ms(row.p99_ms)}</td>
                  <td>{percent(row.success_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="chart-grid">
            <Bars
              title="Selections per modality"
              entries={modalities.map((row) => ({ label: row.modality, value: row.count }))}
              format={String}
            />
            <Bars
              title="P95 latency per modality"
              entries={modalities.map((row) => ({ label: row.modality, value: row.p95_ms }))}
              format={ms}
            />
          </div>
        </>
      )}
    </Panel>
  );
}
