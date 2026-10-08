import { useEffect, useState } from 'react';
import { Alert, messageOf, Panel, useAction, useLoad } from './common';
import { errorMessage, runtimeJson, seg } from './http';
import {
  checkTrainingFile,
  REPORT_START,
  reportFilename,
  reportProgress,
  reportSummary,
  runActionError,
  saveJson,
  templateFile,
  type ExampleTemplates,
  type FileCheck,
  type ReportEvent,
  type ReportProgress,
} from './optimization';
import { parseSse } from './sse';
import { TenantChooser } from './tenants';

interface RunSummary {
  workflow_name: string;
  mode: string | null;
  trigger: string;
  phase: string | null;
  started_at: string | null;
  finished_at: string | null;
}

interface RunStatus {
  workflow_name: string;
  phase: string | null;
  started_at: string | null;
  finished_at: string | null;
  message: string | null;
  steps: Record<string, string>;
  blocked_reason: string | null;
}

/** Phases after which Argo changes nothing about a run. */
const SETTLED = new Set(['Succeeded', 'Failed', 'Error']);
const POLL_MS = 5000;

/** An Argo RFC-3339 UTC time as ``YYYY-MM-DD HH:MM UTC``. */
export function formatArgoTime(value: string | null): string {
  if (!value) return '—';
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/.exec(value);
  return match ? `${match[1]} ${match[2]} UTC` : value;
}

function runsPath(tenant: string): string {
  return `/admin/tenant/${seg(tenant)}/optimize`;
}

/** Re-runs ``reload`` every few seconds while ``active``. */
function usePoll(active: boolean, reload: () => void) {
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(reload, POLL_MS);
    return () => clearInterval(timer);
  }, [active, reload]);
}

interface StartedRun {
  workflow_name: string;
  mode: string;
}

export function OptimizationView() {
  const [tenant, setTenant] = useState('');
  const [selected, setSelected] = useState<string>();
  const [version, setVersion] = useState(0);
  const [notice, setNotice] = useState('');
  const [lastRun, setLastRun] = useState<StartedRun>();
  const changed = (message: string, select?: string) => {
    setNotice(message);
    if (select) setSelected(select);
    setVersion((n) => n + 1);
  };
  return (
    <div className="ops-view">
      <TenantChooser
        action="Show runs"
        onChoose={(chosen) => {
          setTenant(chosen);
          setSelected(undefined);
          setNotice('');
          setLastRun(undefined);
        }}
      />
      {notice && <Alert tone="ok">{notice}</Alert>}
      {tenant && (
        <>
          <TrainingExamples key={`examples-${tenant}`} tenant={tenant} onUploaded={setNotice} />
          <StartRun
            key={tenant}
            tenant={tenant}
            lastRun={lastRun}
            onSelect={setSelected}
            onStarted={(run) => {
              setLastRun(run);
              changed(`Started a ${run.mode} run: ${run.workflow_name}.`, run.workflow_name);
            }}
          />
          <RunsPanel key={`${tenant}-${version}`} tenant={tenant} selected={selected} onSelect={setSelected} />
          {selected && (
            <RunPanel key={`${tenant}-${selected}-${version}`} tenant={tenant} name={selected} onChanged={changed} />
          )}
          <ReportPanel key={`report-${tenant}`} tenant={tenant} />
        </>
      )}
    </div>
  );
}

interface UploadedFile {
  name: string;
  check: FileCheck;
}

interface UploadResult {
  batch_id: string;
  optimizer: string;
  dataset: string;
  item_ids: string[];
}

function TrainingExamples({ tenant, onUploaded }: { tenant: string; onUploaded: (notice: string) => void }) {
  const catalog = useLoad(
    (signal) => runtimeJson<ExampleTemplates>('/admin/tenant/training-example-templates', { signal }),
    [],
  );
  const [templateFor, setTemplateFor] = useState('');
  const [reviewer, setReviewer] = useState('');
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const optimizers = Object.keys(catalog.data?.templates ?? {}).sort();
  const chosen = templateFor || optimizers[0] || '';
  return (
    <Panel title="Upload training examples">
      {catalog.error && <Alert>{catalog.error}</Alert>}
      {catalog.data && (
        <>
          <div className="inline-form" role="group" aria-label="Example template">
            <label>
              Optimizer
              <select aria-label="Template optimizer" value={chosen} onChange={(e) => setTemplateFor(e.target.value)}>
                {optimizers.map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
            <button
              type="button"
              onClick={() => {
                const file = templateFile(chosen, catalog.data!.templates[chosen]);
                saveJson(file.name, file.text);
              }}
            >
              Download template
            </button>
          </div>
          <div className="inline-form">
            <label>
              Examples files (JSON)
              <input
                type="file"
                accept=".json,application/json"
                multiple
                onChange={(e) => {
                  const chosenFiles = Array.from(e.target.files ?? []);
                  void Promise.all(
                    chosenFiles.map(async (file) => ({
                      name: file.name,
                      check: checkTrainingFile(await file.text(), catalog.data!),
                    })),
                  ).then(setFiles);
                }}
              />
            </label>
            <label>
              Uploaded by
              <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="you@example.com" />
            </label>
          </div>
          <p className="muted">
            Uploaded examples are approved into {tenant}'s training dataset; runs of their optimizer train on them.
          </p>
          {files.map((file) => (
            <ExamplesFile key={file.name} tenant={tenant} file={file} reviewer={reviewer} onUploaded={onUploaded} />
          ))}
        </>
      )}
    </Panel>
  );
}

function ExamplesFile({
  tenant,
  file,
  reviewer,
  onUploaded,
}: {
  tenant: string;
  file: UploadedFile;
  reviewer: string;
  onUploaded: (notice: string) => void;
}) {
  const action = useAction();
  const [uploaded, setUploaded] = useState<UploadResult>();
  const { check } = file;
  return (
    <section className="examples-file" aria-label={`File ${file.name}`}>
      <h3>{file.name}</h3>
      {check.ok ? <p className="muted">{check.summary}</p> : <Alert>{check.errors.join(' ')}</Alert>}
      {check.preview !== undefined && (
        <details>
          <summary>Preview: {file.name}</summary>
          <pre aria-label={`Preview of ${file.name}`}>{check.preview}</pre>
        </details>
      )}
      {check.ok && (
        <div className="inline-form">
          <button
            disabled={action.pending || Boolean(uploaded)}
            onClick={() =>
              action.run(async () => {
                if (!reviewer.trim()) throw new Error('Enter your name under Uploaded by first.');
                const result = await runtimeJson<UploadResult>(`/admin/tenant/${seg(tenant)}/training-examples`, {
                  method: 'POST',
                  body: { optimizer: check.optimizer, reviewer: reviewer.trim(), source: file.name, examples: check.examples },
                });
                setUploaded(result);
                onUploaded(
                  `Approved ${result.item_ids.length} ${result.optimizer} examples from ${file.name} into ` +
                    `${result.dataset} as batch ${result.batch_id}.`,
                );
              })
            }
          >
            {action.pending ? 'Uploading…' : uploaded ? 'Uploaded' : `Upload ${file.name}`}
          </button>
          {action.error && <Alert>{action.error}</Alert>}
        </div>
      )}
    </section>
  );
}

interface OptimizeModes {
  modes: string[];
  synthetic_optimizers: string[];
}

const SYNTHETIC_MODE = 'synthetic';

function StartRun({
  tenant,
  lastRun,
  onSelect,
  onStarted,
}: {
  tenant: string;
  lastRun?: StartedRun;
  onSelect: (name: string) => void;
  onStarted: (run: StartedRun) => void;
}) {
  const modes = useLoad(
    (signal) => runtimeJson<OptimizeModes>('/admin/tenant/optimize-modes', { signal }),
    [],
  );
  const [mode, setMode] = useState('');
  const [lookback, setLookback] = useState('48');
  const [optimizers, setOptimizers] = useState<string[]>([]);
  const action = useAction();
  const chosen = mode || modes.data?.modes[0] || '';
  const synthetic = chosen === SYNTHETIC_MODE;
  return (
    <Panel title="Run an optimization">
      {modes.error && <Alert>{modes.error}</Alert>}
      {modes.data && (
        <form
          className="inline-form"
          aria-label="Start optimization"
          onSubmit={(e) => {
            e.preventDefault();
            action.run(async () => {
              const hours = Number(lookback);
              if (!lookback.trim() || !(hours > 0)) throw new Error('Lookback hours must be a number above 0.');
              if (synthetic && !optimizers.length) throw new Error('Choose the optimizers to generate data for.');
              const run = await runtimeJson<StartedRun>(runsPath(tenant), {
                method: 'POST',
                body: { mode: chosen, lookback_hours: hours, ...(synthetic ? { optimizers } : {}) },
              });
              onStarted(run);
            });
          }}
        >
          <label>
            Mode
            <select aria-label="Mode" value={chosen} onChange={(e) => setMode(e.target.value)}>
              {modes.data.modes.map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <label>
            Lookback hours
            <input inputMode="decimal" value={lookback} onChange={(e) => setLookback(e.target.value)} />
          </label>
          {synthetic && (
            <fieldset>
              <legend>Generate training data for</legend>
              {modes.data.synthetic_optimizers.map((value) => (
                <label key={value} className="check">
                  <input
                    type="checkbox"
                    checked={optimizers.includes(value)}
                    onChange={(e) =>
                      setOptimizers((current) =>
                        e.target.checked ? [...current, value] : current.filter((item) => item !== value),
                      )
                    }
                  />
                  {value}
                </label>
              ))}
            </fieldset>
          )}
          <button type="submit" disabled={action.pending}>
            {action.pending ? 'Starting…' : 'Start run'}
          </button>
          {action.error && <Alert>{action.error}</Alert>}
        </form>
      )}
      {synthetic && (
        <p className="muted">Generated examples wait in the Approvals view for review before training uses them.</p>
      )}
      {lastRun && (
        <p className="last-run">
          Last run:{' '}
          <button className="link" onClick={() => onSelect(lastRun.workflow_name)}>
            {lastRun.workflow_name}
          </button>{' '}
          (mode: {lastRun.mode})
        </p>
      )}
    </Panel>
  );
}

function RunsPanel({
  tenant,
  selected,
  onSelect,
}: {
  tenant: string;
  selected?: string;
  onSelect: (name: string) => void;
}) {
  const runs = useLoad(
    (signal) => runtimeJson<{ runs: RunSummary[] }>(`${runsPath(tenant)}/runs`, { signal }).then((body) => body.runs),
    [tenant],
  );
  usePoll(Boolean(runs.data?.some((run) => !SETTLED.has(run.phase ?? ''))), runs.reload);
  return (
    <Panel title={`Optimization runs of ${tenant}`} actions={<button onClick={runs.reload}>Refresh</button>}>
      {runs.error && <Alert>{runs.error}</Alert>}
      {runs.data && runs.data.length === 0 && <p className="muted">No optimization runs for {tenant}.</p>}
      {runs.data && runs.data.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Mode</th>
              <th>Trigger</th>
              <th>Phase</th>
              <th>Started</th>
              <th>Finished</th>
            </tr>
          </thead>
          <tbody>
            {runs.data.map((run) => (
              <tr key={run.workflow_name} className={run.workflow_name === selected ? 'selected' : undefined}>
                <td>
                  <button className="link" onClick={() => onSelect(run.workflow_name)}>
                    {run.workflow_name}
                  </button>
                </td>
                <td>{run.mode ?? 'pipeline'}</td>
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
  );
}

function RunPanel({
  tenant,
  name,
  onChanged,
}: {
  tenant: string;
  name: string;
  onChanged: (notice: string) => void;
}) {
  const status = useLoad(
    (signal) =>
      runtimeJson<RunStatus>(`${runsPath(tenant)}/runs/${seg(name)}`, { signal }).catch((error: unknown) => {
        throw new Error(runActionError(name, error));
      }),
    [tenant, name],
  );
  usePoll(Boolean(status.data && !SETTLED.has(status.data.phase ?? '')), status.reload);
  const action = useAction();
  const run = status.data;
  const act = (verb: 'cancel' | 'retry', done: string) =>
    action.run(async () => {
      let result: RunStatus;
      try {
        result = await runtimeJson<RunStatus>(`${runsPath(tenant)}/runs/${seg(name)}/${verb}`, { method: 'POST' });
      } catch (error) {
        throw new Error(runActionError(name, error));
      }
      onChanged(`${done} ${name}; Argo reports ${result.phase ?? 'no phase'}.`);
    });
  return (
    <Panel title={`Run ${name}`} actions={<button onClick={status.reload}>Refresh</button>}>
      {status.error && <Alert>{status.error}</Alert>}
      {run && (
        <>
          <dl className="facts">
            <dt>Phase</dt>
            <dd>{run.phase ?? 'Pending'}</dd>
            <dt>Started</dt>
            <dd>{formatArgoTime(run.started_at)}</dd>
            <dt>Finished</dt>
            <dd>{formatArgoTime(run.finished_at)}</dd>
            {run.blocked_reason && (
              <>
                <dt>Waiting</dt>
                <dd>{run.blocked_reason}</dd>
              </>
            )}
            {run.message && (
              <>
                <dt>Message</dt>
                <dd>{run.message}</dd>
              </>
            )}
          </dl>
          {Object.keys(run.steps).length > 0 && (
            <table aria-label="Steps">
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Phase</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(run.steps).map(([step, phase]) => (
                  <tr key={step}>
                    <td>{step}</td>
                    <td>{phase}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="inline-form">
            {!SETTLED.has(run.phase ?? '') && (
              <button className="danger" disabled={action.pending} onClick={() => act('cancel', 'Cancelled')}>
                Cancel run
              </button>
            )}
            {(run.phase === 'Failed' || run.phase === 'Error') && (
              <button disabled={action.pending} onClick={() => act('retry', 'Retried')}>
                Retry failed steps
              </button>
            )}
            {action.error && <Alert>{action.error}</Alert>}
          </div>
        </>
      )}
    </Panel>
  );
}

const REPORT_AGENT = 'detailed_report_agent';

function ReportPanel({ tenant }: { tenant: string }) {
  const agents = useLoad(
    (signal) => runtimeJson<{ agents: string[] }>('/agents/', { signal }).then((body) => body.agents),
    [],
  );
  const [progress, setProgress] = useState<ReportProgress>();
  const [running, setRunning] = useState(false);
  const registered = agents.data?.includes(REPORT_AGENT);
  const generate = async () => {
    setRunning(true);
    setProgress(REPORT_START);
    try {
      const response = await fetch(`/api/runtime${runsPath(tenant)}/report`, {
        method: 'POST',
        headers: { accept: 'text/event-stream' },
      });
      if (!response.ok || !response.body) {
        const body = await response.json().catch(() => null);
        throw new Error(errorMessage(body, response.status));
      }
      const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = '';
      let ended = false;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        const parsed = parseSse(buffer + value);
        buffer = parsed.rest;
        for (const frame of parsed.frames) {
          const event = JSON.parse(frame.data) as ReportEvent;
          ended ||= event.type === 'final' || event.type === 'error';
          setProgress((current) => reportProgress(current ?? REPORT_START, event));
        }
      }
      if (!ended)
        setProgress((current) => ({
          ...(current ?? REPORT_START),
          status: '',
          error: 'The report stream ended before the report was finished.',
        }));
    } catch (error) {
      setProgress((current) => ({ ...(current ?? REPORT_START), status: '', error: messageOf(error) }));
    } finally {
      setRunning(false);
    }
  };
  const report = progress?.report;
  const summary = report ? reportSummary(report) : undefined;
  return (
    <Panel title="Optimization report">
      {agents.error && <Alert>{agents.error}</Alert>}
      {agents.data && (
        <p className="muted">
          {registered
            ? `${REPORT_AGENT} is registered with the runtime and writes the report.`
            : `${REPORT_AGENT} is not registered with the runtime, so no report can be generated.`}
        </p>
      )}
      <div className="inline-form">
        <button disabled={running || !registered} onClick={() => void generate()}>
          {running ? 'Generating…' : 'Generate report'}
        </button>
        {report && (
          <button onClick={() => saveJson(reportFilename(new Date()), JSON.stringify(report, null, 2))}>
            Download report
          </button>
        )}
      </div>
      {progress?.status && <p className="muted" aria-live="polite">{progress.status}</p>}
      {progress?.text && !report && <p className="report-text">{progress.text}</p>}
      {summary && (
        <div aria-label="Report">
          <p className="report-text">{summary.summary}</p>
          {summary.recommendations.length > 0 && (
            <ul aria-label="Recommendations">
              {summary.recommendations.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          )}
        </div>
      )}
      {progress?.error && <Alert>{progress.error}</Alert>}
    </Panel>
  );
}
