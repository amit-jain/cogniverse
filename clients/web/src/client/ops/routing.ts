/** A tenant's routing decisions as the runtime reports them, and the figures
 * the Routing view draws from them. */

export interface DecisionLabel {
  label: string;
  confidence: number | null;
  reasoning: string | null;
  suggested_agent: string | null;
  annotator: string | null;
  human_reviewed: boolean;
  requires_review: boolean;
  approved_by: string | null;
}

export interface RoutingDecision {
  span_id: string | null;
  trace_id: string | null;
  start_time: string;
  query: string | null;
  chosen_agent: string;
  confidence: number;
  outcome: string;
  reason: string;
  latency_ms: number;
  entity_extraction_failed: boolean;
  label: DecisionLabel | null;
}

export interface AgentRouting {
  agent: string;
  decisions: number;
  successes: number;
  failures: number;
  ambiguous: number;
  success_rate: number;
  mean_confidence: number;
  mean_latency_ms: number;
}

export interface RoutingDecisions {
  total: number;
  successes: number;
  failures: number;
  ambiguous: number;
  unreadable: number;
  accuracy: number | null;
  confidence_calibration: number | null;
  latency_ms: { mean: number | null; p50: number | null; p95: number | null };
  per_agent: AgentRouting[];
  decisions: RoutingDecision[];
}

export const ROUTING_LOOKBACKS = [
  { hours: 1, label: 'Last hour' },
  { hours: 6, label: 'Last 6 hours' },
  { hours: 24, label: 'Last day' },
  { hours: 168, label: 'Last week' },
  { hours: 720, label: 'Last 30 days' },
];

/** The labels a reviewer gives a decision. */
export const REVIEW_LABELS = ['correct', 'wrong', 'ambiguous', 'insufficient_info'];

export type LabelState = 'unlabelled' | 'llm' | 'reviewed';

/** Whether a decision has no label, an LLM label awaiting review, or a
 * reviewed one (a reviewer's own, or an approved LLM label). */
export function labelState(decision: RoutingDecision): LabelState {
  if (!decision.label) return 'unlabelled';
  return decision.label.human_reviewed ? 'reviewed' : 'llm';
}

export const LABEL_FILTERS: { value: LabelState | 'all'; label: string }[] = [
  { value: 'all', label: 'All decisions' },
  { value: 'llm', label: 'LLM labels to review' },
  { value: 'unlabelled', label: 'Unlabelled' },
  { value: 'reviewed', label: 'Reviewed' },
];

export interface CalibrationBin {
  range: string;
  decisions: number;
  /** The share of the bin's decisions that succeeded; ``null`` when empty. */
  success_rate: number | null;
}

/** Decisions grouped into ``bins`` equal confidence ranges over [0, 1], each
 * with its success rate; a confidence of 1 falls in the last range. */
export function calibrationBins(decisions: RoutingDecision[], bins = 5): CalibrationBin[] {
  return Array.from({ length: bins }, (_, index) => {
    const low = index / bins;
    const high = (index + 1) / bins;
    const inBin = decisions.filter((decision) => Math.min(Math.floor(decision.confidence * bins), bins - 1) === index);
    const successes = inBin.filter((decision) => decision.outcome === 'success').length;
    return {
      range: `${low.toFixed(1)}–${high.toFixed(1)}`,
      decisions: inBin.length,
      success_rate: inBin.length ? successes / inBin.length : null,
    };
  });
}

export interface HourCount {
  /** The start of the UTC hour, as ``YYYY-MM-DDTHH:00:00Z``. */
  hour: string;
  decisions: number;
  successes: number;
}

/** Decisions and successes per UTC hour, oldest hour first; hours without a
 * decision are left out. */
export function decisionsByHour(decisions: RoutingDecision[]): HourCount[] {
  const hours = new Map<string, HourCount>();
  for (const decision of decisions) {
    const hour = `${new Date(decision.start_time).toISOString().slice(0, 13)}:00:00Z`;
    const count = hours.get(hour) ?? { hour, decisions: 0, successes: 0 };
    count.decisions += 1;
    if (decision.outcome === 'success') count.successes += 1;
    hours.set(hour, count);
  }
  return [...hours.values()].sort((a, b) => a.hour.localeCompare(b.hour));
}

/** Who labelled a decision, as the view names it. */
export function labelledBy(label: DecisionLabel): string {
  if (label.annotator === 'llm') return label.approved_by ? `LLM, approved by ${label.approved_by}` : 'LLM';
  return label.annotator ?? 'unknown';
}

/** ``decisions`` with each one in ``reviewed`` (by span ID) replaced by it. */
export function withReviews(
  decisions: RoutingDecision[],
  reviewed: Record<string, RoutingDecision>,
): RoutingDecision[] {
  return decisions.map((decision) => (decision.span_id && reviewed[decision.span_id]) || decision);
}
