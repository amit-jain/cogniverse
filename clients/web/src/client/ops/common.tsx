import { useCallback, useEffect, useState, type ReactNode } from 'react';

export interface Loaded<T> {
  data?: T;
  error?: string;
  loading: boolean;
  reload: () => void;
}

/** Runs ``load`` on mount and whenever ``deps`` change or ``reload`` is called. */
export function useLoad<T>(load: (signal: AbortSignal) => Promise<T>, deps: unknown[]): Loaded<T> {
  const [state, setState] = useState<{ data?: T; error?: string; loading: boolean }>({
    loading: true,
  });
  const [attempt, setAttempt] = useState(0);
  const run = useCallback(load, deps);
  useEffect(() => {
    const controller = new AbortController();
    setState((previous) => ({ data: previous.data, loading: true }));
    run(controller.signal)
      .then((data) => setState({ data, loading: false }))
      .catch((error: unknown) => {
        if (!controller.signal.aborted)
          setState({ error: messageOf(error), loading: false });
      });
    return () => controller.abort();
  }, [run, attempt]);
  return { ...state, reload: () => setAttempt((n) => n + 1) };
}

export function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Wraps an async action with its own pending flag and error message. */
export function useAction() {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const run = async (action: () => Promise<void>) => {
    setPending(true);
    setError('');
    try {
      await action();
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setPending(false);
    }
  };
  return { pending, error, run, clear: () => setError('') };
}

export function Alert({ children, tone = 'error' }: { children: ReactNode; tone?: 'error' | 'ok' }) {
  return (
    <p className={`alert ${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      {children}
    </p>
  );
}

export function Panel({ title, actions, children }: { title: string; actions?: ReactNode; children: ReactNode }) {
  return (
    <section className="panel" aria-label={title}>
      <header className="panel-head">
        <h2>{title}</h2>
        {actions}
      </header>
      {children}
    </section>
  );
}

/** A delete button that asks the operator to type ``name`` before it fires. */
export function ConfirmDelete({
  name,
  what,
  onDelete,
}: {
  name: string;
  what: string;
  onDelete: () => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [typed, setTyped] = useState('');
  const action = useAction();
  if (!open)
    return (
      <button className="danger" onClick={() => setOpen(true)}>
        Delete
      </button>
    );
  return (
    <span className="confirm">
      <input
        aria-label={`Type ${name} to delete this ${what}`}
        placeholder={name}
        value={typed}
        onChange={(e) => setTyped(e.target.value)}
      />
      <button
        className="danger"
        disabled={typed !== name || action.pending}
        onClick={() => action.run(onDelete)}
      >
        {action.pending ? 'Deleting…' : `Delete ${what}`}
      </button>
      <button onClick={() => (setOpen(false), setTyped(''), action.clear())}>Cancel</button>
      {action.error && <Alert>{action.error}</Alert>}
    </span>
  );
}

/** Milliseconds since the epoch as a local date and time. */
export function formatMillis(millis: number): string {
  return new Date(millis).toLocaleString();
}
