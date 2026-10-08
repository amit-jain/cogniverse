import type { GoldenEntry, RatingKind } from '../framework';
import { runtimeJson, seg } from '../http';

export function tenantPath(tenant: string): string {
  return `/admin/tenant/${seg(tenant)}`;
}

export interface RunSummary {
  workflow_name: string;
  mode: string | null;
  trigger: string;
  phase: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface RunStatus {
  workflow_name: string;
  phase: string | null;
  message: string | null;
  blocked_reason: string | null;
}

/** Phases after which nothing changes about a run; Cancelled is a run that was shut down. */
export const SETTLED = new Set(['Succeeded', 'Failed', 'Error', 'Cancelled']);
export const POLL_MS = 5000;

export async function recentRuns(tenant: string, limit: number, signal?: AbortSignal): Promise<RunSummary[]> {
  const body = await runtimeJson<{ runs: RunSummary[] }>(`${tenantPath(tenant)}/optimize/runs?limit=${limit}`, {
    signal,
  });
  return body.runs;
}

export function runStatus(tenant: string, name: string, signal?: AbortSignal): Promise<RunStatus> {
  return runtimeJson<RunStatus>(`${tenantPath(tenant)}/optimize/runs/${seg(name)}`, { signal });
}

export function startRun(
  tenant: string,
  body: { mode: string; lookback_hours?: number; optimizers?: string[]; options: Record<string, unknown> },
): Promise<{ workflow_name: string; mode: string }> {
  return runtimeJson(`${tenantPath(tenant)}/optimize`, { method: 'POST', body });
}

export function annotationCount(
  tenant: string,
  lookbackDays: number,
  signal?: AbortSignal,
): Promise<{ lookback_days: number; annotated_searches: number }> {
  return runtimeJson(`${tenantPath(tenant)}/search-annotations/count?lookback_days=${lookbackDays}`, { signal });
}

export interface SearchAnnotation {
  label: string;
  score: number;
  annotation_type: RatingKind | null;
  notes: string | null;
}

export interface AnnotatableSearch {
  span_id: string;
  trace_id: string | null;
  start_time: string | null;
  query: string;
  results: string[];
  profile: string | null;
  strategy: string | null;
  latency_ms: number | null;
  annotation: SearchAnnotation | null;
}

export function annotatableSearches(tenant: string, lookbackHours: number): Promise<{ searches: AnnotatableSearch[] }> {
  return runtimeJson(`${tenantPath(tenant)}/search-annotations?lookback_hours=${lookbackHours}`);
}

export function annotateSearch(
  tenant: string,
  spanId: string,
  body: { kind: RatingKind; value: number; notes: string },
): Promise<{ span_id: string; label: string; score: number; annotation_type: RatingKind }> {
  return runtimeJson(`${tenantPath(tenant)}/search-annotations/${seg(spanId)}`, { method: 'POST', body });
}

export function buildGoldenDataset(
  tenant: string,
  minRating: number,
  lookbackDays: number,
): Promise<{ dataset: Record<string, GoldenEntry>; untitled_results: number }> {
  return runtimeJson(`${tenantPath(tenant)}/golden-dataset`, {
    method: 'POST',
    body: { min_rating: minRating, lookback_days: lookbackDays },
  });
}

export interface SyntheticOptimizer {
  name: string;
  description: string;
  schema_name: string;
  agent_type: string;
  backend_query_strategy: string;
}

export interface SyntheticSettings {
  confidence_threshold: number;
  sampling_strategies: string[];
  optimizers: SyntheticOptimizer[];
}

export function syntheticSettings(tenant: string, signal?: AbortSignal): Promise<SyntheticSettings> {
  return runtimeJson(`${tenantPath(tenant)}/synthetic/settings`, { signal });
}

export interface GeneratedItem {
  item_id: string;
  status: string;
  confidence: number;
  query: string | null;
  reasoning: string;
  entities: unknown[];
  schema_name: string | null;
  retry_count: number | null;
  generation_metadata: Record<string, unknown>;
  data: Record<string, unknown>;
}

export interface OptimizerOutcome {
  optimizer: string;
  status: string;
  error: string | null;
  batch_id: string | null;
  schema_name: string | null;
  selected_profiles: string[];
  profile_selection_reasoning: string | null;
  generation_time_ms: number | null;
  examples_generated: number;
  auto_approved: number;
  pending_review: number;
  avg_confidence: number | null;
  items: GeneratedItem[];
}

export interface SyntheticRunResults {
  workflow_name: string;
  phase: string | null;
  settled: boolean;
  status: string | null;
  /** Why the run failed before generating for any optimizer. */
  error: string | null;
  parameters: Record<string, unknown>;
  outcomes: OptimizerOutcome[];
}

export function syntheticResults(tenant: string, name: string, signal?: AbortSignal): Promise<SyntheticRunResults> {
  return runtimeJson(`${tenantPath(tenant)}/optimize/runs/${seg(name)}/synthetic`, { signal });
}

export interface TrainingDataset {
  name: string;
  examples: number | null;
  created_at: string | null;
  description: string | null;
}

export function trainingDatasets(tenant: string, signal?: AbortSignal): Promise<{ datasets: TrainingDataset[] }> {
  return runtimeJson(`${tenantPath(tenant)}/datasets`, { signal });
}

export function uploadDataset(
  tenant: string,
  name: string,
  file: File,
): Promise<{ name: string; dataset_id: string; examples: number }> {
  const form = new FormData();
  form.append('name', name);
  form.append('file', file);
  return runtimeJson(`${tenantPath(tenant)}/datasets`, { method: 'POST', body: form });
}

export interface ProfileSpanAnalysis {
  lookback_days: number;
  search_spans: number;
  columns: string[];
  profile_usage: Record<string, Record<string, number>>;
  quality: { column: string; statistics: Record<string, number | null> }[];
  profile_quality: {
    profile_column: string;
    quality_column: string;
    rows: { profile: string; mean: number | null; count: number }[];
  }[];
}

export function profileAnalysis(tenant: string, lookbackDays: number): Promise<ProfileSpanAnalysis> {
  return runtimeJson(`${tenantPath(tenant)}/profile-selection/analysis?lookback_days=${lookbackDays}`);
}

export interface TrainedRecommender {
  train_accuracy: number;
  test_accuracy: number;
  samples: number;
  features: number;
  profiles: string[];
  feature_importance: { feature: string; importance: number }[];
}

export function trainRecommender(tenant: string, lookbackDays: number): Promise<TrainedRecommender> {
  return runtimeJson(`${tenantPath(tenant)}/profile-selection/train`, {
    method: 'POST',
    body: { lookback_days: lookbackDays },
  });
}

export function recommenderState(
  tenant: string,
  signal?: AbortSignal,
): Promise<{ trained: boolean; profiles: string[] }> {
  return runtimeJson(`${tenantPath(tenant)}/profile-selection/model`, { signal });
}

export function predictProfile(
  tenant: string,
  query: string,
): Promise<{ profile: string; confidence: number; features: Record<string, number> }> {
  return runtimeJson(`${tenantPath(tenant)}/profile-selection/predict`, { method: 'POST', body: { query } });
}

export interface OptimizationMetrics {
  lookback_days: number;
  spans: number;
  routing: {
    accuracy: number;
    total_decisions: number;
    avg_latency_ms: number | null;
    confidence_calibration: number;
    per_agent: { agent: string; precision: number; recall: number; f1: number }[];
  } | null;
  evaluation: { spans: number; queries: number };
  training: { date: string; runs: number }[];
}

export function optimizationMetrics(
  tenant: string,
  lookbackDays: number,
  signal?: AbortSignal,
): Promise<OptimizationMetrics> {
  return runtimeJson(`${tenantPath(tenant)}/optimization-metrics?lookback_days=${lookbackDays}`, { signal });
}
