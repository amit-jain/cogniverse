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

export const EVALUATION_LOOKBACKS = [
  { hours: 24, label: 'Last day' },
  { hours: 168, label: 'Last week' },
  { hours: 720, label: 'Last 30 days' },
  { hours: 2160, label: 'Last 90 days' },
];

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
