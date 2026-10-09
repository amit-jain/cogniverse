import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { ErrorReport, ViewErrorBoundary } from '../src/client/ops/ErrorBoundary';
import { RuntimeRequestError, failureDetail } from '../src/client/ops/http';

describe('failureDetail', () => {
  it("lines up a runtime failure's code, failure type and status", () => {
    const stored = new RuntimeRequestError('The review of workflow span s1 was not stored.', 502, {
      detail: {
        error: 'annotation_not_stored',
        message: 'The review of workflow span s1 was not stored.',
        failure: 'RuntimeError',
        tenant_id: 'acme:prod',
      },
    });
    const plain = new RuntimeRequestError('start_time must include a timezone.', 422, {
      detail: 'start_time must include a timezone.',
    });
    expect([stored, plain, new Error('offline'), 'offline'].map(failureDetail)).toEqual([
      'error annotation_not_stored, failure RuntimeError, HTTP 502',
      'HTTP 422',
      null,
      null,
    ]);
  });
});

describe('ViewErrorBoundary', () => {
  const error = new Error('decisions is undefined');
  error.stack = 'TypeError: decisions is undefined\n    at Charts (RoutingView.tsx:301:7)';

  it('shows the error and folds its stack and component stack under Traceback', () => {
    expect(
      renderToStaticMarkup(
        <ErrorReport view="Routing evaluation" error={error} componentStack={'\n    at Charts\n    at Routing'} />,
      ),
    ).toBe(
      '<section class="panel" aria-label="Routing evaluation failed">' +
        '<p class="alert error" role="alert">The Routing evaluation view failed: decisions is undefined</p>' +
        '<details><summary>Traceback</summary><pre class="traceback">' +
        'TypeError: decisions is undefined\n    at Charts (RoutingView.tsx:301:7)\n\nat Charts\n    at Routing' +
        '</pre></details></section>',
    );
  });

  it('renders its children until one throws, then the report of what was thrown', () => {
    const boundary = new ViewErrorBoundary({ view: 'Workflow reviews', children: <p>workflows</p> });
    expect(renderToStaticMarkup(<>{boundary.render()}</>)).toBe('<p>workflows</p>');
    boundary.state = { ...ViewErrorBoundary.getDerivedStateFromError('a string was thrown') };
    expect(renderToStaticMarkup(<>{boundary.render()}</>).split('\n')[0]).toBe(
      '<section class="panel" aria-label="Workflow reviews failed">' +
        '<p class="alert error" role="alert">The Workflow reviews view failed: a string was thrown</p>' +
        '<details><summary>Traceback</summary><pre class="traceback">Error: a string was thrown',
    );
  });
});
