import { useEffect, useState } from 'react';
import { Alert, Panel, useAction, useLoad } from './common';
import { runtimeJson, seg } from './http';
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

export function OptimizationView() {
  const [tenant, setTenant] = useState('');
  const [selected, setSelected] = useState<string>();
  const [version, setVersion] = useState(0);
  const [notice, setNotice] = useState('');
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
        }}
      />
      {notice && <Alert tone="ok">{notice}</Alert>}
      {tenant && (
        <>
          <StartRun key={tenant} tenant={tenant} onStarted={changed} />
          <RunsPanel key={`${tenant}-${version}`} tenant={tenant} selected={selected} onSelect={setSelected} />
          {selected && (
            <RunPanel key={`${tenant}-${selected}-${version}`} tenant={tenant} name={selected} onChanged={changed} />
          )}
        </>
      )}
    </div>
  );
}

interface OptimizeModes {
  modes: string[];
  synthetic_optimizers: string[];
}

const SYNTHETIC_MODE = 'synthetic';

function StartRun({ tenant, onStarted }: { tenant: string; onStarted: (notice: string, select: string) => void }) {
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
              const run = await runtimeJson<{ workflow_name: string; mode: string }>(runsPath(tenant), {
                method: 'POST',
                body: { mode: chosen, lookback_hours: hours, ...(synthetic ? { optimizers } : {}) },
              });
              onStarted(`Started a ${run.mode} run: ${run.workflow_name}.`, run.workflow_name);
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
    (signal) => runtimeJson<RunStatus>(`${runsPath(tenant)}/runs/${seg(name)}`, { signal }),
    [tenant, name],
  );
  usePoll(Boolean(status.data && !SETTLED.has(status.data.phase ?? '')), status.reload);
  const action = useAction();
  const run = status.data;
  const act = (verb: 'cancel' | 'retry', done: string) =>
    action.run(async () => {
      const result = await runtimeJson<RunStatus>(`${runsPath(tenant)}/runs/${seg(name)}/${verb}`, {
        method: 'POST',
      });
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
