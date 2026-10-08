import { useMemo, useState } from 'react';
import { Alert, Panel, useLoad } from '../common';
import { percent } from '../metrics';
import { Plot } from '../Plot';
import { optimizationMetrics } from './api';

const TIMEFRAMES = [7, 30, 90];
const TRAINING_LAYOUT = { xaxis: { title: { text: 'Date' } }, yaxis: { title: { text: 'Training runs' } } };

export function MetricsTab({ tenant }: { tenant: string }) {
  const [days, setDays] = useState(7);
  const metrics = useLoad((signal) => optimizationMetrics(tenant, days, signal), [tenant, days]);
  const data = metrics.data;
  const chart = useMemo(
    () =>
      data && data.training.length
        ? [
            {
              type: 'bar',
              name: 'Training runs',
              x: data.training.map((day) => day.date),
              y: data.training.map((day) => day.runs),
            },
          ]
        : [],
    [data],
  );
  return (
    <>
      <Panel
        title="Optimization metrics"
        actions={
          <button onClick={metrics.reload} disabled={metrics.loading}>
            Refresh metrics
          </button>
        }
      >
        <div className="inline-form">
          <label>
            Timeframe
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              {TIMEFRAMES.map((value) => (
                <option key={value} value={value}>
                  Last {value} days
                </option>
              ))}
            </select>
          </label>
        </div>
        {metrics.error && <Alert>Optimization metrics are unavailable: {metrics.error}</Alert>}
        {data && data.spans === 0 && <p role="status">No spans in the last {days} days. Run some queries first.</p>}
      </Panel>
      {data && data.spans > 0 && (
        <>
          <Panel title="Routing optimization metrics">
            {data.routing ? (
              <>
                <dl className="facts" aria-label="Routing metrics">
                  <dt>Routing accuracy</dt>
                  <dd>{percent(data.routing.accuracy)}</dd>
                  <dt>Total decisions</dt>
                  <dd>{data.routing.total_decisions}</dd>
                  <dt>Average routing latency</dt>
                  <dd>{Math.round(data.routing.avg_latency_ms)}ms</dd>
                  <dt>Confidence calibration</dt>
                  <dd>{data.routing.confidence_calibration.toFixed(3)}</dd>
                </dl>
                <table aria-label="Per-agent performance">
                  <thead>
                    <tr>
                      <th>Agent</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1 score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.routing.per_agent.map((agent) => (
                      <tr key={agent.agent}>
                        <td>{agent.agent}</td>
                        <td>{agent.precision.toFixed(3)}</td>
                        <td>{agent.recall.toFixed(3)}</td>
                        <td>{agent.f1.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="muted">
                  Recall has no ground truth of the agent a query should have reached: it is 1 for an agent with any
                  successful decision.
                </p>
              </>
            ) : (
              <p className="muted">No routing spans: routing metrics need cogniverse.routing spans.</p>
            )}
          </Panel>
          <Panel title="Search quality metrics">
            {data.evaluation.spans ? (
              <dl className="facts" aria-label="Evaluation activity">
                <dt>Evaluation runs</dt>
                <dd>{data.evaluation.spans}</dd>
                <dt>Search queries evaluated</dt>
                <dd>{data.evaluation.queries}</dd>
              </dl>
            ) : (
              <p className="muted">No search evaluation spans. Run search evaluations to see NDCG metrics.</p>
            )}
          </Panel>
          <Panel title="Optimization training runs">
            {chart.length ? (
              <Plot
                title="Training activity over time"
                data={chart}
                layout={TRAINING_LAYOUT}
              />
            ) : (
              <p className="muted">No training runs. Run module optimizations to see training history.</p>
            )}
          </Panel>
        </>
      )}
    </>
  );
}
