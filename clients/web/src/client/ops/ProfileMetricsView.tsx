import { useState } from 'react';
import { Alert, Panel } from './common';
import { seg } from './http';
import { HoursInput } from './lookback';
import { Bars, percent } from './metrics';
import { Plot } from './Plot';
import { TenantChooser } from './tenants';
import { TtlCache, useCachedJson } from './ttlCache';

interface ModalityMetrics {
  modality: string;
  count: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  success_rate: number;
}

export interface ProfileSelections {
  project: string;
  spans: number;
  modalities: ModalityMetrics[];
}

/** The span every profile selection is recorded as. */
export const PROFILE_SELECTION_SPAN = 'cogniverse.profile_selection';
/** The longest window the profile-selection route reads, in hours. */
export const PROFILE_METRICS_MAX_HOURS = 720;
// The most modalities shown as cards; the table lists every one.
const CARDS = 5;

const ms = (value: number) => `${value.toFixed(1)} ms`;
// Selections are kept for half a minute; Refresh reads again.
const cache = new TtlCache<unknown>(30_000);

/** What the view says when the window holds no modality to chart. */
export function emptyWindowMessage(body: ProfileSelections, hours: number): string | null {
  if (body.spans === 0)
    return (
      `No ${PROFILE_SELECTION_SPAN} spans in ${body.project} for the last ${hours} hours. ` +
      'Drive traffic through profile_selection_agent first.'
    );
  if (body.modalities.length === 0)
    return (
      `${body.spans} ${PROFILE_SELECTION_SPAN} spans in this window, but none names a modality. ` +
      'Verify ProfileSelectionAgent is recording it.'
    );
  return null;
}

export function ProfileMetricsView() {
  const [tenant, setTenant] = useState('');
  return (
    <div className="ops-view">
      <TenantChooser action="Show metrics" onChoose={setTenant} />
      {tenant && <Metrics key={tenant} tenant={tenant} />}
    </div>
  );
}

function Metrics({ tenant }: { tenant: string }) {
  const [lookback, setLookback] = useState(24);
  const metrics = useCachedJson<ProfileSelections>(
    cache,
    `/admin/tenant/${seg(tenant)}/telemetry/profile-selection?lookback_hours=${lookback}`,
  );
  const body = metrics.data;
  const modalities = body?.modalities ?? [];
  const empty = body ? emptyWindowMessage(body, lookback) : null;
  return (
    <Panel title={`Profile selections of ${tenant}`} actions={<button onClick={metrics.refresh}>Refresh</button>}>
      <HoursInput value={lookback} onChange={setLookback} min={1} max={PROFILE_METRICS_MAX_HOURS} />
      {metrics.error &&
        (metrics.errorStatus === 504 ? (
          <p className="alert warning" role="status">
            {metrics.error}
          </p>
        ) : (
          <Alert>{metrics.error}</Alert>
        ))}
      {empty && <p className="muted">{empty}</p>}
      {modalities.length > 0 && (
        <>
          <ul className="metric-cards" aria-label="Per-modality metrics">
            {modalities.slice(0, CARDS).map((row) => (
              <li key={row.modality}>
                <span className="card-label">{row.modality.toUpperCase()}</span>
                <span className="card-value">{row.count}</span>
                <span className="card-delta">P95 {row.p95_ms.toFixed(0)} ms</span>
              </li>
            ))}
          </ul>
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
          <Plot
            title="Queries per modality"
            data={[
              {
                type: 'pie',
                labels: modalities.map((row) => row.modality),
                values: modalities.map((row) => row.count),
              },
            ]}
          />
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
