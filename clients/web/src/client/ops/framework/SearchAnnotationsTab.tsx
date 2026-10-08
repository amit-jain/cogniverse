import { useState } from 'react';
import { Alert, Panel, useAction } from '../common';
import { pageNumbers, pageOf, RATING_KINDS, ratingText, type RatingKind } from '../framework';
import { annotatableSearches, annotateSearch, type AnnotatableSearch } from './api';

export function SearchAnnotationsTab({ tenant }: { tenant: string }) {
  const [lookback, setLookback] = useState('24');
  const [kind, setKind] = useState<RatingKind>('thumbs');
  const [searches, setSearches] = useState<AnnotatableSearch[]>();
  const [page, setPage] = useState(1);
  const fetching = useAction();
  return (
    <>
      <Panel title="Search annotations">
        <p className="muted">Rate the tenant's recorded searches; the ratings feed the golden dataset.</p>
        <form
          className="inline-form"
          aria-label="Fetch searches"
          onSubmit={(e) => {
            e.preventDefault();
            fetching.run(async () => {
              const hours = Number(lookback);
              if (!Number.isInteger(hours) || hours < 1 || hours > 168)
                throw new Error('Lookback hours must be a whole number from 1 to 168.');
              const body = await annotatableSearches(tenant, hours);
              setSearches(body.searches);
              setPage(1);
            });
          }}
        >
          <label>
            Lookback hours
            <input type="number" min={1} max={168} value={lookback} onChange={(e) => setLookback(e.target.value)} />
          </label>
          <label>
            Annotation type
            <select value={kind} onChange={(e) => setKind(e.target.value as RatingKind)}>
              {RATING_KINDS.map((option) => (
                <option key={option.kind} value={option.kind}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <button type="submit" disabled={fetching.pending}>
            {fetching.pending ? 'Fetching…' : 'Fetch search results'}
          </button>
          {fetching.error && <Alert>{fetching.error}</Alert>}
        </form>
        {searches && <p role="status">Fetched {searches.length} search results.</p>}
        {!searches && <p className="muted">Fetch the search results to start annotating.</p>}
      </Panel>
      {searches && searches.length > 0 && (
        <Panel title={`Annotation queue (${searches.length} results)`}>
          <label className="pager">
            Page
            <select aria-label="Page" value={page} onChange={(e) => setPage(Number(e.target.value))}>
              {pageNumbers(searches.length).map((number) => (
                <option key={number}>{number}</option>
              ))}
            </select>
          </label>
          {pageOf(searches, page).map((search, index) => (
            <SearchCard
              key={search.span_id}
              tenant={tenant}
              kind={kind}
              position={(page - 1) * 10 + index + 1}
              search={search}
              onSaved={(annotation) =>
                setSearches((current) =>
                  current?.map((item) => (item.span_id === search.span_id ? { ...item, annotation } : item)),
                )
              }
            />
          ))}
        </Panel>
      )}
    </>
  );
}

function SearchCard({
  tenant,
  kind,
  position,
  search,
  onSaved,
}: {
  tenant: string;
  kind: RatingKind;
  position: number;
  search: AnnotatableSearch;
  onSaved: (annotation: NonNullable<AnnotatableSearch['annotation']>) => void;
}) {
  const [notes, setNotes] = useState('');
  const [stars, setStars] = useState(3);
  const [relevance, setRelevance] = useState(0.5);
  const [saved, setSaved] = useState('');
  const action = useAction();
  const save = (value: number) =>
    action.run(async () => {
      const result = await annotateSearch(tenant, search.span_id, { kind, value, notes });
      onSaved({ label: result.label, score: result.score, annotation_type: kind, notes: notes.trim() || null });
      setSaved(`Saved: ${ratingText(kind, value)}`);
    });
  const title = `Result ${position}: ${search.query.slice(0, 80) || 'Unknown query'}`;
  return (
    <details className="result-card" aria-label={title}>
      <summary>{title}</summary>
      <dl className="facts">
        <dt>Query</dt>
        <dd>{search.query || 'N/A'}</dd>
        <dt>Profile</dt>
        <dd>{search.profile ?? 'N/A'}</dd>
        <dt>Strategy</dt>
        <dd>{search.strategy ?? 'N/A'}</dd>
        <dt>Latency</dt>
        <dd>{search.latency_ms === null ? 'N/A' : `${Math.round(search.latency_ms)}ms`}</dd>
        {search.annotation && (
          <>
            <dt>Current rating</dt>
            <dd>
              {search.annotation.label} ({search.annotation.score.toFixed(2)})
              {search.annotation.notes ? ` — ${search.annotation.notes}` : ''}
            </dd>
          </>
        )}
      </dl>
      <h3>Results</h3>
      {search.results.length ? (
        <ol aria-label="Top results">
          {search.results.map((result, index) => (
            <li key={`${result}-${index}`}>{result}</li>
          ))}
        </ol>
      ) : (
        <p className="muted">No results returned.</p>
      )}
      <form
        className="inline-form"
        aria-label="Your annotation"
        onSubmit={(e) => {
          e.preventDefault();
          save(kind === 'stars' ? stars : relevance);
        }}
      >
        {kind === 'thumbs' && (
          <>
            <button type="button" disabled={action.pending} onClick={() => save(1)}>
              Good
            </button>
            <button type="button" disabled={action.pending} onClick={() => save(0)}>
              Bad
            </button>
          </>
        )}
        {kind === 'stars' && (
          <label>
            Rating
            <input
              type="range"
              min={1}
              max={5}
              step={1}
              value={stars}
              onChange={(e) => setStars(Number(e.target.value))}
            />
            <span className="muted">{stars}</span>
          </label>
        )}
        {kind === 'relevance' && (
          <label>
            Relevance
            <input
              type="range"
              min={0}
              max={1}
              step={0.1}
              value={relevance}
              onChange={(e) => setRelevance(Number(e.target.value))}
            />
            <span className="muted">{relevance.toFixed(1)}</span>
          </label>
        )}
        <label>
          Notes (optional)
          <input value={notes} onChange={(e) => setNotes(e.target.value)} />
        </label>
        {kind !== 'thumbs' && (
          <button type="submit" disabled={action.pending}>
            {kind === 'stars' ? 'Save rating' : 'Save score'}
          </button>
        )}
        {saved && <Alert tone="ok">{saved}</Alert>}
        {action.error && <Alert>{action.error}</Alert>}
      </form>
    </details>
  );
}
