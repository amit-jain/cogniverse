import { useEffect, useRef, useState } from 'react';

const NO_LAYOUT = {};

/** A Plotly chart. Plotly loads with the first chart shown, so pages
 * without charts never download it. */
export function Plot({
  title,
  data,
  layout = NO_LAYOUT,
}: {
  title: string;
  data: Record<string, unknown>[];
  layout?: Record<string, unknown>;
}) {
  const root = useRef<HTMLDivElement>(null);
  const [error, setError] = useState('');
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
