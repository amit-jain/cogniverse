import { useState } from 'react';
import { Alert, Panel, useLoad } from './common';
import { formatRunAge } from './framework';
import { GoldenDatasetTab } from './framework/GoldenDatasetTab';
import { MetricsTab } from './framework/MetricsTab';
import { ModuleOptimizationTab } from './framework/ModuleOptimizationTab';
import { ProfileSelectionTab } from './framework/ProfileSelectionTab';
import { SearchAnnotationsTab } from './framework/SearchAnnotationsTab';
import { SyntheticDataTab } from './framework/SyntheticDataTab';
import { annotationCount, recentRuns, type RunSummary } from './framework/api';
import { formatArgoTime } from './OptimizationView';
import { TenantChooser } from './tenants';

const TABS = [
  'Overview',
  'Search annotations',
  'Golden dataset',
  'Synthetic data',
  'Module optimization',
  'Reranking',
  'Profile selection',
  'Metrics',
] as const;
type Tab = (typeof TABS)[number];

export function OptimizationFrameworkView() {
  const [tenant, setTenant] = useState('');
  const [tab, setTab] = useState<Tab>('Overview');
  const [goldenSize, setGoldenSize] = useState(0);
  return (
    <div className="ops-view">
      <TenantChooser
        action="Open"
        onChoose={(chosen) => {
          setTenant(chosen);
          setGoldenSize(0);
        }}
      />
      {tenant && (
        <>
          <nav className="section-tabs" aria-label="Optimization sections">
            {TABS.map((name) => (
              <button key={name} aria-pressed={tab === name} onClick={() => setTab(name)}>
                {name}
              </button>
            ))}
          </nav>
          {tab === 'Overview' && <Overview key={tenant} tenant={tenant} goldenSize={goldenSize} />}
          {tab === 'Search annotations' && <SearchAnnotationsTab key={tenant} tenant={tenant} />}
          {tab === 'Golden dataset' && <GoldenDatasetTab key={tenant} tenant={tenant} onBuilt={setGoldenSize} />}
          {tab === 'Synthetic data' && <SyntheticDataTab key={tenant} tenant={tenant} />}
          {tab === 'Module optimization' && <ModuleOptimizationTab key={tenant} tenant={tenant} />}
          {tab === 'Reranking' && <Reranking key={tenant} tenant={tenant} />}
          {tab === 'Profile selection' && <ProfileSelectionTab key={tenant} tenant={tenant} />}
          {tab === 'Metrics' && <MetricsTab key={tenant} tenant={tenant} />}
        </>
      )}
    </div>
  );
}

/** Days of annotated searches the overview and reranking counts cover. */
const ANNOTATION_WINDOW_DAYS = 90;
/** Runs the overview lists. */
const OVERVIEW_RUNS = 10;

function Overview({ tenant, goldenSize }: { tenant: string; goldenSize: number }) {
  const annotations = useLoad((signal) => annotationCount(tenant, ANNOTATION_WINDOW_DAYS, signal), [tenant]);
  const runs = useLoad((signal) => recentRuns(tenant, OVERVIEW_RUNS, signal), [tenant]);
  const last: RunSummary | undefined = runs.data?.[0];
  return (
    <>
      <Panel title="Optimization overview">
        <dl className="facts" aria-label="Optimization totals">
          <dt>Total annotations</dt>
          <dd>{annotations.error ? '—' : (annotations.data?.annotated_searches ?? '…')}</dd>
          <dt>Golden dataset size</dt>
          <dd>{goldenSize}</dd>
          <dt>Optimization runs</dt>
          <dd>{runs.error ? '—' : (runs.data?.length ?? '…')}</dd>
          <dt>Last optimization</dt>
          <dd>
            {runs.error
              ? '—'
              : !runs.data
                ? '…'
                : last
                  ? `${formatRunAge(last.started_at, new Date())} (${last.phase ?? 'Pending'})`
                  : 'Never'}
          </dd>
        </dl>
        <p className="muted">
          Annotations are the tenant's rated searches of the last {ANNOTATION_WINDOW_DAYS} days; the golden dataset
          size is the one built in this session.
        </p>
        {annotations.error && <Alert>Annotations unavailable: {annotations.error}</Alert>}
        {runs.error && <Alert>Optimization runs unavailable: {runs.error}</Alert>}
      </Panel>
      <Panel title="Optimization workflow">
        <ol>
          <li>
            <strong>Collect annotations</strong> — rate search results with thumbs, stars or a relevance score.
          </li>
          <li>
            <strong>Build a golden dataset</strong> — ground truth from the well-rated searches.
          </li>
          <li>
            <strong>Train optimizers</strong> — routing, workflow, synthetic training data and profile selection.
          </li>
          <li>
            <strong>Monitor metrics</strong> — routing accuracy, evaluation and training activity.
          </li>
          <li>
            <strong>Iterate</strong> — new annotations and feedback feed the next run.
          </li>
        </ol>
      </Panel>
      <Panel title="Recent optimization history">
        {runs.error && <p className="muted">History unavailable while the runtime cannot list runs.</p>}
        {runs.data && runs.data.length === 0 && (
          <p className="muted">No optimization runs yet. Start one under Module optimization or Synthetic data.</p>
        )}
        {runs.data && runs.data.length > 0 && (
          <table aria-label="Recent optimization runs">
            <thead>
              <tr>
                <th>Workflow</th>
                <th>Mode</th>
                <th>Trigger</th>
                <th>Phase</th>
                <th>Started</th>
                <th>Finished</th>
              </tr>
            </thead>
            <tbody>
              {runs.data.map((run) => (
                <tr key={run.workflow_name}>
                  <td>{run.workflow_name}</td>
                  <td>{run.mode ?? '—'}</td>
                  <td>{run.trigger}</td>
                  <td>{run.phase ?? 'Pending'}</td>
                  <td>{formatArgoTime(run.started_at)}</td>
                  <td>{formatArgoTime(run.finished_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </>
  );
}

function Reranking({ tenant }: { tenant: string }) {
  const annotations = useLoad((signal) => annotationCount(tenant, ANNOTATION_WINDOW_DAYS, signal), [tenant]);
  const [minimum, setMinimum] = useState('50');
  const needed = Number(minimum);
  const have = annotations.data?.annotated_searches;
  return (
    <Panel title="Reranking optimization">
      <p className="muted">Ranking learned from annotation feedback:</p>
      <ol>
        <li>Collect feedback — reviewers rate search results.</li>
        <li>Learn preferences — prioritize positively rated results.</li>
        <li>Optimize weights — balance lexical and semantic scores by feedback.</li>
        <li>A/B test — compare against the current ranking.</li>
      </ol>
      <div className="inline-form">
        <label>
          Minimum annotations
          <input
            type="number"
            min={10}
            max={1000}
            value={minimum}
            onChange={(e) => setMinimum(e.target.value)}
          />
        </label>
        <dl className="facts">
          <dt>Current annotations</dt>
          <dd>{annotations.error ? '—' : (have ?? '…')}</dd>
        </dl>
      </div>
      {annotations.error && <Alert>Annotations unavailable: {annotations.error}</Alert>}
      {have !== undefined && (
        <p className="muted" role="status">
          {have >= needed
            ? `${have} annotations meet the minimum of ${needed}. No reranker trainer exists; the annotations feed the golden dataset and the profile recommender.`
            : `${needed - have} more annotations are needed to reach ${needed}.`}
        </p>
      )}
    </Panel>
  );
}
