/** A tenant's searches of its golden queries, scored as the runtime reports
 * them, and the figures the Evaluation view draws from them. */

export interface StrategyScores {
  profile: string;
  strategy: string;
  queries: number;
  mrr: number;
  ndcg: number;
  recall_at_1: number;
  recall_at_5: number;
  precision_at_5: number;
  success_rate: number;
}

export interface QueryScores {
  profile: string;
  strategy: string;
  query: string;
  expected: string[];
  retrieved: string[];
  searched_at: string;
  trace_id: string | null;
  mrr: number;
  ndcg: number;
  recall_at_1: number;
  recall_at_5: number;
  precision_at_5: number;
}

export interface GoldenEvaluation {
  golden_queries: number;
  strategies: StrategyScores[];
  queries: QueryScores[];
  unsearched_queries: string[];
  failed_searches: number;
  unscored_searches: number;
}

/** The longest window an evaluation reads, in hours. */
export const EVALUATION_MAX_HOURS = 24 * 90;

export interface EvaluationDataset {
  id: string;
  name: string;
  example_count: number;
  created_at: string;
  description: string;
}

export interface EvaluationDatasets {
  /** Where the browser opens the telemetry store's UI; null when unknown. */
  phoenix_url: string | null;
  datasets: EvaluationDataset[];
}

export interface DatasetEvaluation extends GoldenEvaluation {
  dataset: EvaluationDataset;
}

/** How a dataset is offered: its name and example count. */
export function datasetOption(dataset: EvaluationDataset): string {
  return `${dataset.name} (${dataset.example_count} examples)`;
}

/** The day a dataset was created, as its timestamp states it. */
export function createdDate(dataset: EvaluationDataset): string {
  return dataset.created_at.split('T')[0];
}

/** The dataset's page in the telemetry store's UI. */
export function phoenixDatasetUrl(phoenixUrl: string, dataset: EvaluationDataset): string {
  return `${phoenixUrl}/datasets/${encodeURIComponent(dataset.id)}`;
}

/** A score's band: good from 0.7, fair from 0.3, poor below. */
export function scoreTone(value: number): 'good' | 'fair' | 'poor' {
  return value >= 0.7 ? 'good' : value >= 0.3 ? 'fair' : 'poor';
}

/** Each profile with its strategies, in the order the scores list them. */
export function strategiesByProfile(scores: StrategyScores[]): { profile: string; strategies: StrategyScores[] }[] {
  const grouped = new Map<string, StrategyScores[]>();
  for (const row of scores) grouped.set(row.profile, [...(grouped.get(row.profile) ?? []), row]);
  return [...grouped].map(([profile, strategies]) => ({ profile, strategies }));
}

/** How a profile and strategy are named in the view. */
export function pairName({ profile, strategy }: { profile: string; strategy: string }): string {
  return `${profile} / ${strategy}`;
}

export interface SuccessMatrix {
  profiles: string[];
  strategies: string[];
  /** Success rate per profile (row) and strategy (column); ``null`` where
   * the profile was not searched with the strategy. */
  cells: (number | null)[][];
}

/** The success rate of every profile with every strategy, both sorted. */
export function successMatrix(scores: StrategyScores[]): SuccessMatrix {
  const profiles = [...new Set(scores.map((row) => row.profile))].sort();
  const strategies = [...new Set(scores.map((row) => row.strategy))].sort();
  const cells = profiles.map((profile) =>
    strategies.map(
      (strategy) => scores.find((row) => row.profile === profile && row.strategy === strategy)?.success_rate ?? null,
    ),
  );
  return { profiles, strategies, cells };
}

/** The first ``limit`` retrieved sources, each marked expected or not. */
export function markRetrieved(query: QueryScores, limit = 5): { source: string; expected: boolean }[] {
  return query.retrieved.slice(0, limit).map((source) => ({ source, expected: query.expected.includes(source) }));
}
