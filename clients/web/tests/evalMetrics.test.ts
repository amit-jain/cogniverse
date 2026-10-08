import { describe, expect, it } from 'vitest';
import {
  createdDate,
  datasetOption,
  phoenixDatasetUrl,
  scoreTone,
  strategiesByProfile,
  type EvaluationDataset,
  type StrategyScores,
} from '../src/client/ops/evaluation';
import { parseHours } from '../src/client/ops/lookback';
import { emptyWindowMessage } from '../src/client/ops/ProfileMetricsView';
import { abCompareCommand } from '../src/client/ops/RlmAbView';
import { TtlCache } from '../src/client/ops/ttlCache';

const dataset: EvaluationDataset = {
  id: 'RGF0YXNldDo3',
  name: 'golden v2',
  example_count: 12,
  created_at: '2026-10-07T23:59:58.123456+00:00',
  description: '',
};

function scores(profile: string, strategy: string): StrategyScores {
  return { profile, strategy, queries: 1, mrr: 0, ndcg: 0, recall_at_1: 0, recall_at_5: 0, precision_at_5: 0, success_rate: 0 };
}

describe('datasets', () => {
  it('offers a dataset by name and example count', () => {
    expect(datasetOption(dataset)).toBe('golden v2 (12 examples)');
  });

  it('dates a dataset by the day its timestamp names', () => {
    expect(createdDate(dataset)).toBe('2026-10-07');
  });

  it('links a dataset to its page in the Phoenix UI', () => {
    expect(phoenixDatasetUrl('http://localhost:33006', dataset)).toBe('http://localhost:33006/datasets/RGF0YXNldDo3');
    expect(phoenixDatasetUrl('http://p', { ...dataset, id: 'a/b=' })).toBe('http://p/datasets/a%2Fb%3D');
  });
});

describe('scoreTone', () => {
  it('bands scores at 0.7 and 0.3', () => {
    expect([1, 0.7, 0.6999, 0.3, 0.2999, 0].map(scoreTone)).toEqual(['good', 'good', 'fair', 'fair', 'poor', 'poor']);
  });
});

describe('strategiesByProfile', () => {
  it('groups strategies under their profile in listed order', () => {
    expect(
      strategiesByProfile([scores('video', 'bm25'), scores('audio', 'hybrid'), scores('video', 'hybrid')]).map(
        (group) => [group.profile, group.strategies.map((s) => s.strategy)],
      ),
    ).toEqual([
      ['video', ['bm25', 'hybrid']],
      ['audio', ['hybrid']],
    ]);
  });
});

describe('parseHours', () => {
  it('accepts hours within the bounds, fractions included', () => {
    expect(parseHours('720', 1, 720)).toEqual({ hours: 720 });
    expect(parseHours(' 0.1 ', 0.1, 720)).toEqual({ hours: 0.1 });
  });

  it('refuses hours outside the bounds or not a number', () => {
    const refused = { error: 'Lookback must be a number of hours from 1 to 720.' };
    expect(['0', '721', 'abc', '', '  '].map((text) => parseHours(text, 1, 720))).toEqual(Array(5).fill(refused));
  });
});

describe('emptyWindowMessage', () => {
  it('names the project and the agent to drive when no selection was recorded', () => {
    expect(emptyWindowMessage({ project: 'cogniverse-acme-acme', spans: 0, modalities: [] }, 6)).toBe(
      'No cogniverse.profile_selection spans in cogniverse-acme-acme for the last 6 hours. ' +
        'Drive traffic through profile_selection_agent first.',
    );
  });

  it('tells selections without a modality apart from none', () => {
    expect(emptyWindowMessage({ project: 'p', spans: 3, modalities: [] }, 24)).toBe(
      '3 cogniverse.profile_selection spans in this window, but none names a modality. ' +
        'Verify ProfileSelectionAgent is recording it.',
    );
  });

  it('says nothing when there are modalities to show', () => {
    const video = { modality: 'video', count: 1, p50_ms: 1, p95_ms: 1, p99_ms: 1, success_rate: 1 };
    expect(emptyWindowMessage({ project: 'p', spans: 2, modalities: [video] }, 24)).toBeNull();
  });
});

describe('abCompareCommand', () => {
  it('names the tenant and the dataset flag', () => {
    expect(abCompareCommand('acme:prod')).toBe(
      'cogniverse-optim --mode ab-compare --tenant-id acme:prod --queries-dataset <name>',
    );
  });
});

describe('TtlCache', () => {
  it('reuses an answer within its lifetime and reads again after it or when fresh', async () => {
    let now = 0;
    let reads = 0;
    const cache = new TtlCache<number>(1000, () => now);
    const load = () => Promise.resolve(++reads);
    expect(await cache.get('a', load)).toBe(1);
    now = 999;
    expect(await cache.get('a', load)).toBe(1);
    expect(await cache.get('b', load)).toBe(2);
    expect(await cache.get('a', load, true)).toBe(3);
    now = 1999;
    expect(await cache.get('a', load)).toBe(4);
  });

  it('shares one read among concurrent callers of a key', async () => {
    let reads = 0;
    let finish: (value: number) => void = () => {};
    const cache = new TtlCache<number>(1000, () => 0);
    const load = () => {
      reads += 1;
      return new Promise<number>((resolve) => (finish = resolve));
    };
    const answers = Promise.all([cache.get('a', load), cache.get('a', load), cache.get('a', load)]);
    finish(7);
    expect(await answers).toEqual([7, 7, 7]);
    expect(reads).toBe(1);
  });

  it('keeps no failed read, so the next call reads again', async () => {
    let reads = 0;
    const cache = new TtlCache<number>(1000, () => 0);
    await expect(cache.get('a', () => (reads++, Promise.reject(new Error('down'))))).rejects.toThrow('down');
    expect(await cache.get('a', () => (reads++, Promise.resolve(5)))).toBe(5);
    expect(reads).toBe(2);
  });
});
