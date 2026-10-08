import { useMemo, useState } from 'react';
import { Alert, Panel, useLoad } from './common';
import {
  EVALUATION_LOOKBACKS,
  markRetrieved,
  pairName,
  successMatrix,
  type GoldenEvaluation,
  type StrategyScores,
} from './evaluation';
import { runtimeJson, seg } from './http';
import { LookbackSelect, percent } from './metrics';
import { Plot } from './Plot';
import { TenantChooser } from './tenants';

const score = (value: number) => value.toFixed(3);
const when = (iso: string) => new Date(iso).toLocaleString();

export function EvaluationView() {
  const [tenant, setTenant] = useState('');
  const [lookback, setLookback] = useState(168);
  return (
    <div className="ops-view">
      <TenantChooser action="Evaluate" onChoose={setTenant} />
      {tenant && <Evaluation key={`${tenant}-${lookback}`} tenant={tenant} lookback={lookback} onLookback={setLookback} />}
    </div>
  );
}

function Evaluation({ tenant, lookback, onLookback }: { tenant: string; lookback: number; onLookback: (hours: number) => void }) {
  const evaluation = useLoad(
    (signal) =>
      runtimeJson<GoldenEvaluation>(`/admin/tenant/${seg(tenant)}/evaluation/golden?lookback_hours=${lookback}`, {
        signal,
      }),
    [tenant, lookback],
  );
  const data = evaluation.data;
  const scoredQueries = new Set(data?.queries.map((query) => query.query)).size;
  return (
    <>
      <Panel title={`Golden set evaluation of ${tenant}`} actions={<button onClick={evaluation.reload}>Refresh</button>}>
        <div className="inline-form">
          <LookbackSelect value={lookback} onChange={onLookback} options={EVALUATION_LOOKBACKS} />
        </div>
        {evaluation.error && <Alert>{evaluation.error}</Alert>}
        {data && (
          <dl className="facts" aria-label="Evaluation summary">
            <dt>Golden queries</dt>
            <dd>{data.golden_queries}</dd>
            <dt>Searched</dt>
            <dd>{scoredQueries}</dd>
            <dt>Not searched</dt>
            <dd>{data.unsearched_queries.length}</dd>
            <dt>Failed searches</dt>
            <dd>{data.failed_searches}</dd>
            <dt>Unscored searches</dt>
            <dd>{data.unscored_searches}</dd>
          </dl>
        )}
        {data && data.unscored_searches > 0 && (
          <p className="muted">
            Unscored searches returned a result with no source title, so they cannot be matched to the golden set.
          </p>
        )}
        {data && data.strategies.length === 0 && (
          <p className="muted">No searches of the golden queries were recorded in this window.</p>
        )}
      </Panel>
      {data && data.strategies.length > 0 && (
        <>
          <Scores strategies={data.strategies} />
          <Queries data={data} />
        </>
      )}
      {data && data.unsearched_queries.length > 0 && (
        <Panel title="Golden queries not searched">
          <ul aria-label="Golden queries not searched">
            {data.unsearched_queries.map((query) => (
              <li key={query}>{query}</li>
            ))}
          </ul>
        </Panel>
      )}
    </>
  );
}

function Scores({ strategies }: { strategies: StrategyScores[] }) {
  const matrix = useMemo(() => successMatrix(strategies), [strategies]);
  const data = useMemo(
    () => [
      {
        type: 'heatmap',
        x: matrix.strategies,
        y: matrix.profiles,
        z: matrix.cells,
        zmin: 0,
        zmax: 1,
        colorscale: [
          [0, '#e74c3c'],
          [1, '#27ae60'],
        ],
        showscale: false,
        text: matrix.cells.map((row) => row.map((cell) => (cell === null ? '' : percent(cell)))),
        texttemplate: '%{text}',
        hovertemplate: 'Profile: %{y}<br>Strategy: %{x}<br>Success: %{text}<extra></extra>',
      },
    ],
    [matrix],
  );
  return (
    <Panel title="Scores by profile and strategy">
      <table aria-label="Scores by profile and strategy">
        <thead>
          <tr>
            <th>Profile</th>
            <th>Strategy</th>
            <th>Queries</th>
            <th>MRR</th>
            <th>nDCG@10</th>
            <th>Recall@1</th>
            <th>Recall@5</th>
            <th>Precision@5</th>
            <th>Success</th>
          </tr>
        </thead>
        <tbody>
          {strategies.map((row) => (
            <tr key={pairName(row)}>
              <td>{row.profile}</td>
              <td>{row.strategy}</td>
              <td>{row.queries}</td>
              <td>{score(row.mrr)}</td>
              <td>{score(row.ndcg)}</td>
              <td>{score(row.recall_at_1)}</td>
              <td>{score(row.recall_at_5)}</td>
              <td>{score(row.precision_at_5)}</td>
              <td>{percent(row.success_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted">Success is the share of queries whose first result is an expected source.</p>
      <Plot
        title="Success by profile and strategy"
        data={data}
        layout={{ xaxis: { title: { text: 'Strategy' } }, yaxis: { title: { text: 'Profile' } } }}
      />
    </Panel>
  );
}

function Queries({ data }: { data: GoldenEvaluation }) {
  const [pair, setPair] = useState(pairName(data.strategies[0]));
  const queries = data.queries.filter((query) => pairName(query) === pair);
  return (
    <Panel title="Query results">
      <div className="inline-form">
        <label>
          Profile and strategy
          <select value={pair} onChange={(e) => setPair(e.target.value)}>
            {data.strategies.map((row) => (
              <option key={pairName(row)}>{pairName(row)}</option>
            ))}
          </select>
        </label>
      </div>
      <table aria-label="Query results">
        <thead>
          <tr>
            <th>Query</th>
            <th>Expected</th>
            <th>Retrieved (top 5)</th>
            <th>MRR</th>
            <th>Recall@1</th>
            <th>Recall@5</th>
            <th>Searched</th>
          </tr>
        </thead>
        <tbody>
          {queries.map((query) => (
            <tr key={query.query}>
              <td>{query.query}</td>
              <td>{query.expected.join(', ')}</td>
              <td>
                {query.retrieved.length === 0 ? (
                  'No results'
                ) : (
                  <ol className="retrieved">
                    {markRetrieved(query).map(({ source, expected }) => (
                      <li key={source} className={expected ? 'hit' : 'miss'}>
                        {expected ? '✓' : '✗'} {source}
                      </li>
                    ))}
                  </ol>
                )}
              </td>
              <td>{score(query.mrr)}</td>
              <td>{score(query.recall_at_1)}</td>
              <td>{score(query.recall_at_5)}</td>
              <td title={query.trace_id ? `Trace ${query.trace_id}` : undefined}>{when(query.searched_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
