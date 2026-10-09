import { Fragment, useEffect, useState } from 'react';
import { Alert, Panel, useAction, useLoad } from '../common';
import {
  cellText,
  compactTimestamp,
  confidenceBand,
  downloadJson,
  entityText,
  exampleColumns,
  expectedReviewRate,
  millis,
} from '../framework';
import {
  POLL_MS,
  recentRuns,
  startRun,
  syntheticResults,
  syntheticSettings,
  type GeneratedItem,
  type OptimizerOutcome,
  type SyntheticSettings,
} from './api';

const DEFAULT_OPTIMIZER = 'profile';
/** Pending items reviewed inline; the Approvals view holds the rest. */
const INLINE_REVIEW = 5;
const SAMPLE_ROWS = 10;

export function SyntheticDataTab({ tenant }: { tenant: string }) {
  const settings = useLoad((signal) => syntheticSettings(tenant, signal), [tenant]);
  const [run, setRun] = useState('');
  return (
    <>
      {settings.error && <Alert>Synthetic generation settings are unavailable: {settings.error}</Alert>}
      {settings.data && <Generate tenant={tenant} settings={settings.data} onStarted={setRun} />}
      <EarlierRuns tenant={tenant} current={run} onChoose={setRun} />
      {run && <RunResults key={run} tenant={tenant} name={run} />}
      {settings.data && <OptimizerInfo settings={settings.data} />}
    </>
  );
}

function Generate({
  tenant,
  settings,
  onStarted,
}: {
  tenant: string;
  settings: SyntheticSettings;
  onStarted: (name: string) => void;
}) {
  const names = settings.optimizers.map((optimizer) => optimizer.name);
  const [optimizer, setOptimizer] = useState(names.includes(DEFAULT_OPTIMIZER) ? DEFAULT_OPTIMIZER : names[0]);
  const [count, setCount] = useState('100');
  const [sample, setSample] = useState('200');
  const [review, setReview] = useState(true);
  const [strategy, setStrategy] = useState('');
  const [maxProfiles, setMaxProfiles] = useState(3);
  const [notice, setNotice] = useState('');
  const action = useAction();
  const threshold = settings.confidence_threshold;
  return (
    <Panel title="Synthetic data generation">
      <p className="muted">Generate training data for an optimizer by sampling the tenant's backend content.</p>
      <form
        className="stacked-form"
        aria-label="Generate synthetic data"
        onSubmit={(e) => {
          e.preventDefault();
          action.run(async () => {
            const examples = Number(count);
            const documents = Number(sample);
            if (!Number.isInteger(examples) || examples < 1 || examples > 10000)
              throw new Error('Examples to generate must be a whole number from 1 to 10000.');
            if (!Number.isInteger(documents) || documents < 1 || documents > 10000)
              throw new Error('Backend sample size must be a whole number from 1 to 10000.');
            const started = await startRun(tenant, {
              mode: 'synthetic',
              optimizers: [optimizer],
              options: {
                count: examples,
                vespa_sample_size: documents,
                max_profiles: maxProfiles,
                human_review: review,
                ...(strategy ? { strategy } : {}),
              },
            });
            setNotice(`Generating ${examples} examples for ${optimizer}: ${started.workflow_name}.`);
            onStarted(started.workflow_name);
          });
        }}
      >
        <div className="inline-form">
          <label>
            Optimizer
            <select aria-label="Optimizer" value={optimizer} onChange={(e) => setOptimizer(e.target.value)}>
              {names.map((name) => (
                <option key={name}>{name}</option>
              ))}
            </select>
          </label>
          <label>
            Examples to generate
            <input type="number" min={1} max={10000} value={count} onChange={(e) => setCount(e.target.value)} />
          </label>
          <label>
            Backend sample size
            <input
              type="number"
              min={1}
              max={10000}
             
              value={sample}
              onChange={(e) => setSample(e.target.value)}
            />
          </label>
        </div>
        <fieldset>
          <legend>Quality control</legend>
          <label className="check">
            <input type="checkbox" checked={review} onChange={(e) => setReview(e.target.checked)} />
            Human-in-the-loop review
          </label>
          {review ? (
            <>
              <p className="muted">
                Examples at or above the threshold are approved for training at once; the rest wait for review.
              </p>
              <dl className="facts" aria-label="Confidence settings">
                <dt>Auto-approval threshold</dt>
                <dd>{threshold.toFixed(2)}</dd>
                <dt>Expected review rate</dt>
                <dd>{expectedReviewRate(threshold)}</dd>
              </dl>
              <p className="muted">
                With threshold {threshold.toFixed(2)}, examples scoring {threshold.toFixed(2)}+ are approved
                automatically; lower-confidence examples need your review.
              </p>
            </>
          ) : (
            <p className="muted">Without review every generated example is approved for training.</p>
          )}
        </fieldset>
        <details>
          <summary>Advanced options</summary>
          <div className="inline-form">
            <label>
              Sampling strategy
              <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                <option value="">The optimizer's own</option>
                {settings.sampling_strategies.map((name) => (
                  <option key={name}>{name}</option>
                ))}
              </select>
            </label>
            <label>
              Max profiles
              <input
                type="range"
                min={1}
                max={10}
                value={maxProfiles}
                onChange={(e) => setMaxProfiles(Number(e.target.value))}
              />
              <span className="muted">{maxProfiles}</span>
            </label>
            <p className="muted">Tenant: {tenant}</p>
          </div>
        </details>
        <div className="inline-form">
          <button type="submit" disabled={action.pending}>
            {action.pending ? 'Starting…' : 'Generate synthetic data'}
          </button>
        </div>
        {action.error && <Alert>Generation failed to start: {action.error}</Alert>}
        {notice && <Alert tone="ok">{notice}</Alert>}
      </form>
    </Panel>
  );
}

function EarlierRuns({
  tenant,
  current,
  onChoose,
}: {
  tenant: string;
  current: string;
  onChoose: (name: string) => void;
}) {
  const runs = useLoad((signal) => recentRuns(tenant, 50, signal), [tenant, current]);
  const synthetic = (runs.data ?? []).filter((run) => run.mode === 'synthetic');
  return (
    <Panel title="Synthetic runs" actions={<button onClick={runs.reload}>Refresh</button>}>
      {runs.error && <Alert>Synthetic runs are unavailable: {runs.error}</Alert>}
      {runs.data && synthetic.length === 0 && <p className="muted">No synthetic runs yet.</p>}
      {synthetic.length > 0 && (
        <label>
          Show the results of
          <select aria-label="Synthetic run" value={current} onChange={(e) => onChoose(e.target.value)}>
            <option value="">Choose a run</option>
            {synthetic.map((run) => (
              <option key={run.workflow_name} value={run.workflow_name}>
                {run.workflow_name} ({run.phase ?? 'Pending'})
              </option>
            ))}
          </select>
        </label>
      )}
    </Panel>
  );
}

function RunResults({ tenant, name }: { tenant: string; name: string }) {
  const results = useLoad((signal) => syntheticResults(tenant, name, signal), [tenant, name]);
  const settled = results.data?.settled;
  useEffect(() => {
    if (settled !== false) return;
    const timer = setInterval(results.reload, POLL_MS);
    return () => clearInterval(timer);
  }, [settled, results.reload]);
  return (
    <Panel title={`Results of ${name}`} actions={<button onClick={results.reload}>Refresh</button>}>
      {results.error && <Alert>The run's results are unavailable: {results.error}</Alert>}
      {results.data && !results.data.settled && (
        <p role="status">The run is {results.data.phase ?? 'Pending'}; results appear when it finishes.</p>
      )}
      {results.data?.settled &&
        results.data.outcomes.map((outcome) => (
          <Outcome key={outcome.optimizer} tenant={tenant} outcome={outcome} />
        ))}
      {results.data?.settled && results.data.error && <Alert>The run failed: {results.data.error}</Alert>}
      {results.data?.settled && results.data.outcomes.length === 0 && !results.data.error && (
        <p className="muted">
          {results.data.phase === 'Cancelled'
            ? 'The run was cancelled before it reported an outcome.'
            : `The run ended ${results.data.phase} without generating for any optimizer.`}
        </p>
      )}
    </Panel>
  );
}

function Outcome({ tenant, outcome }: { tenant: string; outcome: OptimizerOutcome }) {
  const [filename, setFilename] = useState(`synthetic_${outcome.optimizer}_${compactTimestamp(new Date())}.json`);
  const pending = outcome.items.filter((item) => item.status === 'pending_review');
  const rows = outcome.items.slice(0, SAMPLE_ROWS).map((item) => item.data);
  return (
    <section aria-label={`Outcome for ${outcome.optimizer}`}>
      <h3>{outcome.optimizer}</h3>
      {(outcome.status === 'failed' || outcome.status === 'error') && (
        <Alert>
          Generation for {outcome.optimizer} failed: {outcome.error ?? 'no error recorded'}
        </Alert>
      )}
      {outcome.status === 'no_data' && (
        <p className="muted">The backend sample held nothing to generate {outcome.optimizer} examples from.</p>
      )}
      {outcome.status === 'success' && (
        <>
          <p role="status">
            Generated {outcome.examples_generated} examples: {outcome.auto_approved} auto-approved,{' '}
            {outcome.pending_review} awaiting review.
          </p>
          <dl className="facts" aria-label={`Approval statistics of ${outcome.optimizer}`}>
            <dt>Auto-approved</dt>
            <dd>{outcome.auto_approved}</dd>
            <dt>Pending review</dt>
            <dd>{outcome.pending_review}</dd>
            <dt>Average confidence</dt>
            <dd>{outcome.avg_confidence === null ? '—' : outcome.avg_confidence.toFixed(2)}</dd>
          </dl>
        </>
      )}
      {outcome.profile_selection_reasoning && (
        <p>
          <strong>Profile selection:</strong> {outcome.profile_selection_reasoning}
        </p>
      )}
      {outcome.status !== 'failed' && outcome.status !== 'error' && (
        <dl className="facts" aria-label={`Generation of ${outcome.optimizer}`}>
          <dt>Schema</dt>
          <dd>{outcome.schema_name ?? '—'}</dd>
          <dt>Generation time</dt>
          <dd>{millis(outcome.generation_time_ms)}</dd>
          <dt>Profiles used</dt>
          <dd>{outcome.selected_profiles.length}</dd>
        </dl>
      )}
      {outcome.selected_profiles.length > 0 && (
        <>
          <h3>Selected profiles</h3>
          <ul aria-label="Selected profiles">
            {outcome.selected_profiles.map((profile) => (
              <li key={profile}>
                <code>{profile}</code>
              </li>
            ))}
          </ul>
        </>
      )}
      {rows.length > 0 && (
        <>
          <h3>Sample generated examples</h3>
          <table aria-label="Sample generated examples">
            <thead>
              <tr>
                {exampleColumns(rows).map((column) => (
                  <th key={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={index}>
                  {exampleColumns(rows).map((column) => (
                    <td key={column}>{cellText(row[column])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {outcome.status === 'success' && <InlineReview tenant={tenant} pending={pending} />}
      {outcome.status === 'success' && (
        <div className="inline-form" role="group" aria-label={`Export ${outcome.optimizer}`}>
          <label>
            Filename
            <input value={filename} onChange={(e) => setFilename(e.target.value)} />
          </label>
          <button
            onClick={() =>
              downloadJson(filename, {
                optimizer: outcome.optimizer,
                schema_name: outcome.schema_name,
                count: outcome.items.length,
                selected_profiles: outcome.selected_profiles,
                profile_selection_reasoning: outcome.profile_selection_reasoning,
                data: outcome.items.map((item) => item.data),
                metadata: {
                  batch_id: outcome.batch_id,
                  generation_time_ms: outcome.generation_time_ms,
                  auto_approved: outcome.auto_approved,
                  pending_review: outcome.pending_review,
                },
              })
            }
          >
            Download JSON
          </button>
        </div>
      )}
      {outcome.status === 'success' && (
        <p className="muted">
          Approved examples train the {outcome.optimizer} optimizer on its next run: start it under Module
          optimization or Optimization runs.
        </p>
      )}
    </section>
  );
}

export function InlineReview({ tenant, pending }: { tenant: string; pending: GeneratedItem[] }) {
  if (!pending.length) return <p role="status">All items were approved automatically; no review is needed.</p>;
  return (
    <>
      <h3>Review low-confidence items</h3>
      <p>
        <strong>{pending.length} items</strong> need your review: they scored below the confidence threshold.
      </p>
      {pending.slice(0, INLINE_REVIEW).map((item, index) => (
        <ReviewItem key={item.item_id} item={item} position={index + 1} total={pending.length} />
      ))}
      <p className="muted">
        {pending.length > INLINE_REVIEW ? `Showing ${INLINE_REVIEW} of ${pending.length} items. ` : ''}
        Approve or reject them in the <a href="#/ops/approvals">Approvals</a> view under tenant {tenant}.
      </p>
    </>
  );
}

export function ReviewItem({ item, position, total }: { item: GeneratedItem; position: number; total: number }) {
  const band = confidenceBand(item.confidence);
  const title = `Item ${position}/${total} - Confidence: ${item.confidence.toFixed(2)} - ${(item.query ?? 'N/A').slice(0, 60)}`;
  return (
    <details className="result-card" aria-label={title} open={position === 1}>
      <summary>{title}</summary>
      <dl className="facts">
        <dt>Generated query</dt>
        <dd>{item.query ?? 'N/A'}</dd>
        {item.reasoning && (
          <>
            <dt>Reasoning</dt>
            <dd>{item.reasoning}</dd>
          </>
        )}
        <dt>Entities</dt>
        <dd>{item.entities.length ? item.entities.map(entityText).join(', ') : 'No entities'}</dd>
        <dt>Confidence</dt>
        <dd>{item.confidence.toFixed(2)}</dd>
        <dt>Retries</dt>
        <dd>{item.retry_count ?? 0}</dd>
        <dt>Band</dt>
        <dd className={`band ${band.tone}`}>{band.label}</dd>
      </dl>
      {Object.keys(item.generation_metadata).length > 0 && (
        <details>
          <summary>Generation details</summary>
          <dl className="facts">
            {Object.entries(item.generation_metadata).map(([key, value]) => (
              <Fragment key={key}>
                <dt>{key}</dt>
                <dd>{cellText(value)}</dd>
              </Fragment>
            ))}
          </dl>
        </details>
      )}
    </details>
  );
}

function OptimizerInfo({ settings }: { settings: SyntheticSettings }) {
  return (
    <Panel title="Optimizer information">
      <table aria-label="Optimizers">
        <thead>
          <tr>
            <th>Optimizer</th>
            <th>Description</th>
            <th>Schema</th>
            <th>Trains</th>
            <th>Default sampling</th>
          </tr>
        </thead>
        <tbody>
          {settings.optimizers.map((optimizer) => (
            <tr key={optimizer.name}>
              <td>{optimizer.name}</td>
              <td>{optimizer.description}</td>
              <td>
                <code>{optimizer.schema_name}</code>
              </td>
              <td>{optimizer.agent_type}</td>
              <td>{optimizer.backend_query_strategy}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
