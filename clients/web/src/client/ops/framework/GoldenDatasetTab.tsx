import { useState } from 'react';
import { Alert, Panel, useAction } from '../common';
import { compactDate, downloadJson, fileSafe, goldenSample, type GoldenEntry } from '../framework';
import { buildGoldenDataset } from './api';

export function GoldenDatasetTab({ tenant, onBuilt }: { tenant: string; onBuilt: (size: number) => void }) {
  const [minRating, setMinRating] = useState(0.8);
  const [lookback, setLookback] = useState('30');
  const [dataset, setDataset] = useState<Record<string, GoldenEntry>>();
  const [untitled, setUntitled] = useState(0);
  const [filename, setFilename] = useState(`golden_dataset_${fileSafe(tenant)}_${compactDate(new Date())}.json`);
  const building = useAction();
  const size = dataset ? Object.keys(dataset).length : 0;
  return (
    <>
      <Panel title="Golden dataset builder">
        <p className="muted">Ground truth from the tenant's annotated searches.</p>
        <form
          className="inline-form"
          aria-label="Build golden dataset"
          onSubmit={(e) => {
            e.preventDefault();
            building.run(async () => {
              const days = Number(lookback);
              if (!Number.isInteger(days) || days < 1 || days > 90)
                throw new Error('Lookback days must be a whole number from 1 to 90.');
              const body = await buildGoldenDataset(tenant, minRating, days);
              setDataset(body.dataset);
              setUntitled(body.untitled_results);
              onBuilt(Object.keys(body.dataset).length);
            });
          }}
        >
          <label>
            Minimum rating
            <input
              type="range"
              min={0}
              max={1}
              step={0.1}
              value={minRating}
              onChange={(e) => setMinRating(Number(e.target.value))}
            />
            <span className="muted">{minRating.toFixed(1)}</span>
          </label>
          <label>
            Lookback days
            <input type="number" min={1} max={90} value={lookback} onChange={(e) => setLookback(e.target.value)} />
          </label>
          <button type="submit" disabled={building.pending}>
            {building.pending ? 'Building…' : 'Build golden dataset'}
          </button>
          {building.error && <Alert>Building the golden dataset failed: {building.error}</Alert>}
        </form>
        {dataset && <p role="status">Built a golden dataset of {size} queries.</p>}
        {dataset && untitled > 0 && (
          <p className="muted">{untitled} annotated results carry no source title and were left out.</p>
        )}
      </Panel>
      {dataset && size > 0 && (
        <>
          <Panel title="Dataset sample">
            <table aria-label="Golden dataset sample">
              <thead>
                <tr>
                  <th>Query</th>
                  <th>Expected videos</th>
                  <th>Average relevance</th>
                </tr>
              </thead>
              <tbody>
                {goldenSample(dataset).map((row) => (
                  <tr key={row.query}>
                    <td>{row.query}</td>
                    <td>{row.expectedVideos}</td>
                    <td>{row.avgRelevance.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
          <Panel title="Export golden dataset">
            <div className="inline-form">
              <label>
                Filename
                <input value={filename} onChange={(e) => setFilename(e.target.value)} />
              </label>
              <button onClick={() => downloadJson(filename, dataset)}>Download JSON</button>
            </div>
          </Panel>
        </>
      )}
    </>
  );
}
