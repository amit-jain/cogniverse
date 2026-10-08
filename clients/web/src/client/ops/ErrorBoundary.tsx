import { Component, type ErrorInfo, type ReactNode } from 'react';

/** What a view that failed to render shows: the error and, folded, its stack
 * trace and the component stack it was thrown in. */
export function ErrorReport({ view, error, componentStack }: { view: string; error: Error; componentStack?: string }) {
  return (
    <section className="panel" aria-label={`${view} failed`}>
      <p className="alert error" role="alert">
        The {view} view failed: {error.message}
      </p>
      <details>
        <summary>Traceback</summary>
        <pre className="traceback">{[error.stack ?? String(error), componentStack?.trim()].filter(Boolean).join('\n\n')}</pre>
      </details>
    </section>
  );
}

/** Renders ``children``, or the ``ErrorReport`` of the error they threw while
 * rendering. */
export class ViewErrorBoundary extends Component<
  { view: string; children: ReactNode },
  { error: Error | null; componentStack?: string }
> {
  state: { error: Error | null; componentStack?: string } = { error: null };

  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(_error: unknown, info: ErrorInfo) {
    this.setState({ componentStack: info.componentStack ?? undefined });
  }

  render() {
    if (this.state.error)
      return <ErrorReport view={this.props.view} error={this.state.error} componentStack={this.state.componentStack} />;
    return this.props.children;
  }
}
