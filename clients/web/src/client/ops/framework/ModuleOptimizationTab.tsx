import { useEffect, useState } from 'react';
import { Alert, Panel, useAction, useLoad } from '../common';
import { MODULES } from '../framework';
import { POLL_MS, runStatus, SETTLED, startRun, trainingDatasets, uploadDataset } from './api';

export function ModuleOptimizationTab({ tenant }: { tenant: string }) {
  const [mode, setMode] = useState('routing');
  const [iterations, setIterations] = useState('100');
  const [lookback, setLookback] = useState('48');
  const [useSynthetic, setUseSynthetic] = useState(true);
  const [dataset, setDataset] = useState('');
  const [run, setRun] = useState('');
  const action = useAction();
  const module = MODULES.find((entry) => entry.mode === mode) ?? MODULES[0];
  return (
    <>
      <Panel title="Module optimization">
        <p className="muted">
          Optimize the routing and workflow modules; each DSPy step picks its optimizer from its training data.
        </p>
        <form
          className="stacked-form"
          aria-label="Optimize a module"
          onSubmit={(e) => {
            e.preventDefault();
            action.run(async () => {
              const maxIterations = Number(iterations);
              const hours = Number(lookback);
              if (!Number.isInteger(maxIterations) || maxIterations < 1 || maxIterations > 500)
                throw new Error('Max iterations must be a whole number from 1 to 500.');
              if (!(hours > 0)) throw new Error('Lookback hours must be a number above 0.');
              const started = await startRun(tenant, {
                mode,
                lookback_hours: hours,
                options: {
                  max_iterations: maxIterations,
                  use_synthetic_data: useSynthetic,
                  ...(!useSynthetic && dataset.trim() ? { dataset_name: dataset.trim() } : {}),
                },
              });
              setRun(started.workflow_name);
            });
          }}
        >
          <div className="inline-form">
            <label>
              Module to optimize
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                {MODULES.map((entry) => (
                  <option key={entry.mode} value={entry.mode}>
                    {entry.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Max iterations
              <input
                type="number"
                min={1}
                max={500}
                value={iterations}
                onChange={(e) => setIterations(e.target.value)}
              />
            </label>
            <label>
              Lookback hours
              <input inputMode="decimal" value={lookback} onChange={(e) => setLookback(e.target.value)} />
            </label>
          </div>
          <p className="muted">{module.description}</p>
          <p className="muted">
            An iteration is one bootstrap round of a DSPy compile, or one 50-span evaluation batch of the workflow
            optimizer.
          </p>
          <label className="check">
            <input type="checkbox" checked={useSynthetic} onChange={(e) => setUseSynthetic(e.target.checked)} />
            Use synthetic data
          </label>
          <p className="muted">
            {useSynthetic
              ? "The tenant's approved synthetic examples train the DSPy steps too."
              : 'Training uses production data only; a golden dataset can stand in for the profile ground truth.'}
          </p>
          {!useSynthetic && <DatasetPicker tenant={tenant} value={dataset} onChange={setDataset} />}
          <div className="inline-form">
            <button type="submit" disabled={action.pending}>
              {action.pending ? 'Submitting…' : 'Submit module optimization'}
            </button>
          </div>
          {action.error && <Alert>Submitting the optimization failed: {action.error}</Alert>}
        </form>
      </Panel>
      {run && <SubmittedRun key={run} tenant={tenant} name={run} />}
    </>
  );
}

function DatasetPicker({
  tenant,
  value,
  onChange,
}: {
  tenant: string;
  value: string;
  onChange: (name: string) => void;
}) {
  const datasets = useLoad((signal) => trainingDatasets(tenant, signal), [tenant]);
  const [file, setFile] = useState<File>();
  const [name, setName] = useState('');
  const [created, setCreated] = useState('');
  const upload = useAction();
  const chosen = datasets.data?.datasets.find((entry) => entry.name === value);
  useEffect(() => {
    const first = datasets.data?.datasets[0];
    if (first && !value) onChange(first.name);
  }, [datasets.data, value, onChange]);
  return (
    <fieldset>
      <legend>Golden dataset</legend>
      <p className="muted">
        A telemetry dataset of query and expected_videos rows; the profile step learns from it in place of the
        uploaded ground truth.
      </p>
      {datasets.error && (
        <>
          <Alert>The tenant's datasets are unavailable: {datasets.error}</Alert>
          <label>
            Dataset name (manual)
            <input
              value={value}
              placeholder={`golden_eval-${tenant}`}
              onChange={(e) => onChange(e.target.value)}
            />
          </label>
        </>
      )}
      {datasets.data && datasets.data.datasets.length > 0 && (
        <div className="inline-form">
          <label>
            Telemetry dataset
            <select value={value} onChange={(e) => onChange(e.target.value)}>
              {datasets.data.datasets.map((entry) => (
                <option key={entry.name}>{entry.name}</option>
              ))}
            </select>
          </label>
          {chosen && (
            <dl className="facts" aria-label="Chosen dataset">
              <dt>Dataset size</dt>
              <dd>{chosen.examples ?? '—'}</dd>
              <dt>Created</dt>
              <dd>{chosen.created_at ? chosen.created_at.slice(0, 10) : 'N/A'}</dd>
              {chosen.description && (
                <>
                  <dt>Description</dt>
                  <dd>{chosen.description}</dd>
                </>
              )}
            </dl>
          )}
        </div>
      )}
      {datasets.data && datasets.data.datasets.length === 0 && (
        <p className="muted">The tenant has no datasets yet. Upload a CSV to create one.</p>
      )}
      {datasets.data && (
        <div className="inline-form" role="group" aria-label="Upload CSV dataset">
          <label>
            CSV dataset
            <input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files?.[0])} />
          </label>
          <label>
            Dataset name
            <input value={name} placeholder="my_eval_dataset" onChange={(e) => setName(e.target.value)} />
          </label>
          <button
            type="button"
            disabled={!file || !name.trim() || upload.pending}
            onClick={() =>
              upload.run(async () => {
                if (!file) return;
                const result = await uploadDataset(tenant, name.trim(), file);
                setCreated(`Created dataset ${result.name} with ${result.examples} queries.`);
                onChange(result.name);
                datasets.reload();
              })
            }
          >
            {upload.pending ? 'Uploading…' : 'Upload dataset'}
          </button>
          {created && <Alert tone="ok">{created}</Alert>}
          {upload.error && <Alert>Creating the dataset failed: {upload.error}</Alert>}
        </div>
      )}
      <p className="muted">CSV columns: query, expected_videos (comma-separated) and an optional category.</p>
    </fieldset>
  );
}

function SubmittedRun({ tenant, name }: { tenant: string; name: string }) {
  const status = useLoad((signal) => runStatus(tenant, name, signal), [tenant, name]);
  const settled = status.data ? SETTLED.has(status.data.phase ?? '') : true;
  useEffect(() => {
    if (settled) return;
    const timer = setInterval(status.reload, POLL_MS);
    return () => clearInterval(timer);
  }, [settled, status.reload]);
  return (
    <Panel title={`Run ${name}`}>
      <p role="status">Submitted {name}.</p>
      {status.error && <Alert>The run's status is unavailable: {status.error}</Alert>}
      {status.data && (
        <dl className="facts">
          <dt>Phase</dt>
          <dd>{status.data.phase ?? 'Pending'}</dd>
          {status.data.blocked_reason && (
            <>
              <dt>Waiting</dt>
              <dd>{status.data.blocked_reason}</dd>
            </>
          )}
          {status.data.message && (
            <>
              <dt>Message</dt>
              <dd>{status.data.message}</dd>
            </>
          )}
        </dl>
      )}
      <p className="muted">
        Follow, cancel or retry it in <a href="#/ops/optimization">Optimization runs</a>.
      </p>
    </Panel>
  );
}
