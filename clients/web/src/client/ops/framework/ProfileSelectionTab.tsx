import { useState } from 'react';
import { Alert, Panel, useAction, useLoad } from '../common';
import { Bars, percent } from '../metrics';
import {
  predictProfile,
  profileAnalysis,
  recommenderState,
  trainRecommender,
  type ProfileSpanAnalysis,
  type TrainedRecommender,
} from './api';

export function ProfileSelectionTab({ tenant }: { tenant: string }) {
  const [lookback, setLookback] = useState(30);
  const [analysis, setAnalysis] = useState<ProfileSpanAnalysis>();
  const analyzing = useAction();
  return (
    <>
      <Panel title="Profile selection optimization">
        <p className="muted">Learn which processing profile serves which kind of query from the tenant's telemetry.</p>
        <form
          className="inline-form"
          aria-label="Analyze search spans"
          onSubmit={(e) => {
            e.preventDefault();
            analyzing.run(async () => setAnalysis(await profileAnalysis(tenant, lookback)));
          }}
        >
          <label>
            Lookback days
            <input
              type="range"
              min={1}
              max={90}
              value={lookback}
              onChange={(e) => setLookback(Number(e.target.value))}
            />
            <span className="muted">{lookback}</span>
          </label>
          <button type="submit" disabled={analyzing.pending}>
            {analyzing.pending ? 'Analyzing…' : 'Analyze search spans'}
          </button>
          {analyzing.error && <Alert>The span analysis did not complete: {analyzing.error}</Alert>}
        </form>
        {analysis && <Analysis analysis={analysis} />}
      </Panel>
      <Training tenant={tenant} lookback={lookback} />
      <Prediction tenant={tenant} />
    </>
  );
}

function Analysis({ analysis }: { analysis: ProfileSpanAnalysis }) {
  if (!analysis.search_spans)
    return (
      <p role="status">
        No search spans in the last {analysis.lookback_days} days. Run searches with different profiles first.
      </p>
    );
  const profileColumns = Object.keys(analysis.profile_usage);
  return (
    <>
      <p role="status">Found {analysis.search_spans} search spans.</p>
      <details>
        <summary>Available span data</summary>
        <p className="muted">{analysis.columns.join(', ')}</p>
      </details>
      <h3>Profile usage</h3>
      {profileColumns.length ? (
        <div className="chart-grid">
          {profileColumns.map((column) => (
            <Bars
              key={column}
              title={column}
              entries={Object.entries(analysis.profile_usage[column]).map(([label, value]) => ({ label, value }))}
              format={String}
            />
          ))}
        </div>
      ) : (
        <p className="muted">No profile information in the span attributes; searches must record their profile.</p>
      )}
      <h3>Quality metrics</h3>
      {analysis.quality.length ? (
        <table aria-label="Quality metrics">
          <thead>
            <tr>
              <th>Metric</th>
              {['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'].map((key) => (
                <th key={key}>{key}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {analysis.quality.map((metric) => (
              <tr key={metric.column}>
                <td>{metric.column}</td>
                {['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'].map((key) => (
                  <td key={key}>{formatStatistic(metric.statistics[key])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="muted">No quality metrics (NDCG, accuracy) in the spans. Run evaluations to collect them.</p>
      )}
      {analysis.profile_quality.map((table) => (
        <table
          key={`${table.profile_column}-${table.quality_column}`}
          aria-label={`${table.profile_column} by ${table.quality_column}`}
        >
          <caption>
            {table.profile_column} vs {table.quality_column}
          </caption>
          <thead>
            <tr>
              <th>Profile</th>
              <th>Mean</th>
              <th>Count</th>
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row) => (
              <tr key={row.profile}>
                <td>{row.profile}</td>
                <td>{formatStatistic(row.mean)}</td>
                <td>{row.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ))}
    </>
  );
}

function formatStatistic(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return Number.isInteger(value) ? String(value) : value.toFixed(3);
}

function Training({ tenant, lookback }: { tenant: string; lookback: number }) {
  const state = useLoad((signal) => recommenderState(tenant, signal), [tenant]);
  const [trained, setTrained] = useState<TrainedRecommender>();
  const training = useAction();
  return (
    <Panel title="Train profile selector">
      <p className="muted">Training needs search spans that record their profile, a quality score and the query.</p>
      <ul className="muted">
        <li>Each query becomes six features: length, word count, temporal, spatial and object keywords, word length.</li>
        <li>An XGBoost classifier learns the profile of each search from them.</li>
        <li>The model is stored for the tenant and serves the predictions below.</li>
      </ul>
      {state.data?.trained && (
        <p role="status">A trained model is stored for {tenant}: {state.data.profiles.join(', ')}.</p>
      )}
      {state.error && <Alert>The stored model is unavailable: {state.error}</Alert>}
      <div className="inline-form">
        <button
          disabled={training.pending}
          onClick={() =>
            training.run(async () => {
              setTrained(await trainRecommender(tenant, lookback));
              state.reload();
            })
          }
        >
          {training.pending ? 'Training…' : 'Train profile selector model'}
        </button>
      </div>
      {training.error && <Alert>Training failed: {training.error}</Alert>}
      {trained && (
        <>
          <p role="status">
            Model trained and stored: {trained.samples} samples, {trained.features} features, {trained.profiles.length}{' '}
            profiles ({trained.profiles.join(', ')}).
          </p>
          <dl className="facts" aria-label="Training results">
            <dt>Training accuracy</dt>
            <dd>{percent(trained.train_accuracy)}</dd>
            <dt>Test accuracy</dt>
            <dd>{percent(trained.test_accuracy)}</dd>
            <dt>Samples</dt>
            <dd>{trained.samples}</dd>
          </dl>
          <table aria-label="Feature importance">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Importance</th>
              </tr>
            </thead>
            <tbody>
              {trained.feature_importance.map((entry) => (
                <tr key={entry.feature}>
                  <td>{entry.feature}</td>
                  <td>{entry.importance.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Panel>
  );
}

function Prediction({ tenant }: { tenant: string }) {
  const [query, setQuery] = useState('Show me videos from last week');
  const [result, setResult] = useState<{ profile: string; confidence: number; features: Record<string, number> }>();
  const predicting = useAction();
  return (
    <Panel title="Test prediction">
      <form
        className="inline-form"
        aria-label="Predict a profile"
        onSubmit={(e) => {
          e.preventDefault();
          predicting.run(async () => setResult(await predictProfile(tenant, query)));
        }}
      >
        <label>
          Test query
          <input required value={query} onChange={(e) => setQuery(e.target.value)} />
        </label>
        <button type="submit" disabled={predicting.pending}>
          {predicting.pending ? 'Predicting…' : 'Load model and predict'}
        </button>
        {predicting.error && <Alert>Prediction failed: {predicting.error}</Alert>}
      </form>
      {result && (
        <>
          <p role="status">
            Recommended profile: <strong>{result.profile}</strong> (confidence: {percent(result.confidence)})
          </p>
          <table aria-label="Extracted features">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.features).map(([feature, value]) => (
                <tr key={feature}>
                  <td>{feature}</td>
                  <td>{Number.isInteger(value) ? value : value.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Panel>
  );
}
