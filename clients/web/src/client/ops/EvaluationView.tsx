import { useMemo, useState } from 'react';
import { Alert, Panel } from './common';
import {
  EVALUATION_MAX_HOURS,
  createdDate,
  datasetOption,
  markRetrieved,
  pairName,
  phoenixDatasetUrl,
  scoreTone,
  strategiesByProfile,
  successMatrix,
  type DatasetEvaluation,
  type EvaluationDataset,
  type EvaluationDatasets,
  type GoldenEvaluation,
  type StrategyScores,
} from './evaluation';
import { seg } from './http';
import { HoursInput } from './lookback';
import { percent } from './metrics';
import { Plot } from './Plot';
import { Tabs } from './Tabs';
import { TenantChooser } from './tenants';
import { TtlCache, useCachedJson } from './ttlCache';

const score = (value: number) => value.toFixed(3);
const when = (iso: string) => new Date(iso).toLocaleString();
// Evaluations are kept for a minute; Refresh reads again.
const cache = new TtlCache<unknown>(60_000);

export function EvaluationView() {
  const [tenant, setTenant] = useState('');
  return (
    <div className="ops-view">
      <TenantChooser action="Evaluate" onChoose={setTenant} />
      {tenant && (
        <Tabs
          key={tenant}
          label="Evaluate against"
          tabs={[
            { name: 'Golden set', panel: () => <GoldenSet tenant={tenant} /> },
            { name: 'Phoenix datasets', panel: () => <Datasets tenant={tenant} /> },
          ]}
        />
      )}
    </div>
  );
}

function GoldenSet({ tenant }: { tenant: string }) {
  const [lookback, setLookback] = useState(168);
  const evaluation = useCachedJson<GoldenEvaluation>(
    cache,
    `/admin/tenant/${seg(tenant)}/evaluation/golden?lookback_hours=${lookback}`,
  );
  return (
    <Evaluation
      title={`Golden set evaluation of ${tenant}`}
      loaded={evaluation}
      lookback={lookback}
      onLookback={setLookback}
      none={`No searches of the golden queries were recorded for tenant ${tenant} in the last ${lookback} hours.`}
    />
  );
}

function Datasets({ tenant }: { tenant: string }) {
  const listing = useCachedJson<EvaluationDatasets>(cache, `/admin/tenant/${seg(tenant)}/evaluation/datasets`);
  const [chosen, setChosen] = useState('');
  const datasets = listing.data?.datasets ?? [];
  const dataset = datasets.find((d) => d.id === chosen) ?? datasets[0];
  const phoenixUrl = listing.data?.phoenix_url ?? null;
  return (
    <>
      <Panel title={`Phoenix datasets of ${tenant}`} actions={<button onClick={listing.refresh}>Refresh</button>}>
        {listing.error && <Alert>{listing.error}</Alert>}
        {listing.data && datasets.length === 0 && (
          <p className="muted">No datasets of tenant {tenant} were found in Phoenix.</p>
        )}
        {dataset && (
          <>
            <div className="inline-form">
              <label>
                Dataset
                <select value={dataset.id} onChange={(e) => setChosen(e.target.value)}>
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {datasetOption(d)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <dl className="facts" aria-label="Dataset details">
              <dt>Dataset examples</dt>
              <dd>{dataset.example_count}</dd>
              <dt>Created</dt>
              <dd>{createdDate(dataset)}</dd>
            </dl>
            {phoenixUrl ? (
              <a href={phoenixDatasetUrl(phoenixUrl, dataset)} target="_blank" rel="noreferrer">
                View in Phoenix
              </a>
            ) : (
              <p className="muted">No Phoenix address is configured, so the dataset is not linked.</p>
            )}
          </>
        )}
      </Panel>
      {dataset && <DatasetScores key={dataset.id} tenant={tenant} dataset={dataset} />}
    </>
  );
}

function DatasetScores({ tenant, dataset }: { tenant: string; dataset: EvaluationDataset }) {
  const [lookback, setLookback] = useState(168);
  const evaluation = useCachedJson<DatasetEvaluation>(
    cache,
    `/admin/tenant/${seg(tenant)}/evaluation/dataset?dataset_id=${seg(dataset.id)}&lookback_hours=${lookback}`,
  );
  return (
    <Evaluation
      title={`Evaluation of ${dataset.name}`}
      loaded={evaluation}
      lookback={lookback}
      onLookback={setLookback}
      none={`No searches of this dataset's queries were recorded for tenant ${tenant} in the last ${lookback} hours.`}
    />
  );
}

function Evaluation({
  title,
  loaded,
  lookback,
  onLookback,
  none,
}: {
  title: string;
  loaded: { data?: GoldenEvaluation; error?: string; refresh: () => void };
  lookback: number;
  onLookback: (hours: number) => void;
  none: string;
}) {
  const data = loaded.data;
  const scoredQueries = new Set(data?.queries.map((query) => query.query)).size;
  return (
    <>
      <Panel title={title} actions={<button onClick={loaded.refresh}>Refresh</button>}>
        <HoursInput value={lookback} onChange={onLookback} min={1} max={EVALUATION_MAX_HOURS} />
        {loaded.error && <Alert>{loaded.error}</Alert>}
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
        {data && data.strategies.length === 0 && <p className="muted">{none}</p>}
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
  const groups = strategiesByProfile(data.strategies);
  return (
    <Panel title="Query results">
      <Tabs
        label="Profiles"
        tabs={groups.map((group) => ({
          name: group.profile,
          panel: () => (
            <Tabs
              key={group.profile}
              label={`Strategies of ${group.profile}`}
              tabs={group.strategies.map((scores) => ({
                name: scores.strategy,
                panel: () => <StrategyResults key={pairName(scores)} data={data} scores={scores} />,
              }))}
            />
          ),
        }))}
      />
    </Panel>
  );
}

function Badge({ value }: { value: number }) {
  return <td className={`score ${scoreTone(value)}`}>{score(value)}</td>;
}

function StrategyResults({ data, scores }: { data: GoldenEvaluation; scores: StrategyScores }) {
  const queries = data.queries.filter((query) => pairName(query) === pairName(scores));
  return (
    <>
      <dl className="facts" aria-label={`Summary of ${pairName(scores)}`}>
        <dt>MRR</dt>
        <dd>{percent(scores.mrr)}</dd>
        <dt>Recall@1</dt>
        <dd>{percent(scores.recall_at_1)}</dd>
        <dt>Recall@5</dt>
        <dd>{percent(scores.recall_at_5)}</dd>
        <dt>Queries</dt>
        <dd>{scores.queries}</dd>
      </dl>
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
              <Badge value={query.mrr} />
              <Badge value={query.recall_at_1} />
              <Badge value={query.recall_at_5} />
              <td title={query.trace_id ? `Trace ${query.trace_id}` : undefined}>{when(query.searched_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
