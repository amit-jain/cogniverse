import { useEffect, useRef, useState } from 'react';

const NO_LAYOUT = {};

/** A point a box or lasso selection holds. */
export interface SelectedPoint {
  customdata?: unknown;
}

interface Selectable {
  on(event: 'plotly_selected', handler: (event?: { points: SelectedPoint[] }) => void): void;
  on(event: 'plotly_deselect', handler: () => void): void;
}

/** A Plotly chart. Plotly loads with the first chart shown, so pages
 * without charts never download it. ``onSelect`` receives the points a box
 * or lasso selection holds, or null when the selection is cleared. */
export function Plot({
  title,
  data,
  layout = NO_LAYOUT,
  onSelect,
}: {
  title: string;
  data: Record<string, unknown>[];
  layout?: Record<string, unknown>;
  onSelect?: (points: SelectedPoint[] | null) => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  const [error, setError] = useState('');
  const select = useRef(onSelect);
  select.current = onSelect;
  const bound = useRef(false);
  useEffect(() => {
    const element = root.current;
    if (!element) return;
    let cancelled = false;
    import('plotly.js-dist-min')
      .then((plotly) => {
        if (cancelled) return;
        return plotly.react(
          element,
          data,
          { title: { text: title }, autosize: true, margin: { t: 48, r: 16, b: 48, l: 56 }, ...layout },
          { responsive: true, displaylogo: false },
        );
      })
      .then((drawn) => {
        if (!drawn || bound.current || !select.current) return;
        bound.current = true;
        const selectable = drawn as unknown as Selectable;
        selectable.on('plotly_selected', (event) => select.current?.(event ? event.points : null));
        selectable.on('plotly_deselect', () => select.current?.(null));
      })
      .catch((e: unknown) => setError(`The chart could not be drawn: ${e instanceof Error ? e.message : String(e)}`));
    return () => {
      cancelled = true;
    };
  }, [title, data, layout]);
  useEffect(() => {
    const element = root.current;
    return () => {
      if (element) import('plotly.js-dist-min').then((plotly) => plotly.purge(element));
    };
  }, []);
  return (
    <figure className="plot" aria-label={title}>
      {error && (
        <p className="alert error" role="alert">
          {error}
        </p>
      )}
      <div ref={root} className="plot-area" />
    </figure>
  );
}
