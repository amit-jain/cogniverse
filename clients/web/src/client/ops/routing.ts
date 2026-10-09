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
  precision: number;
  recall: number;
  f1: number;
}

export interface RoutingDecisions {
  /** The telemetry project the decisions were read from. */
  project: string;
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

/** The most hours a window may span: the runtime reads at most 30 days. */
export const MAX_LOOKBACK_HOURS = 720;

/** ``text`` as a window in whole hours, or ``null`` when it is not one from
 * 1 to ``MAX_LOOKBACK_HOURS``. */
export function lookbackHours(text: string): number | null {
  const hours = Number(text.trim());
  return text.trim() && Number.isInteger(hours) && hours >= 1 && hours <= MAX_LOOKBACK_HOURS ? hours : null;
}

/** The read-only labels the LLM annotator gives, with the reviewer label
 * each stands for. */
const LEGACY_LABELS: Record<string, string> = { correct_routing: 'correct', wrong_routing: 'wrong' };

/** The reviewer label a relabel form starts on: the decision's own label in
 * reviewer terms when it has one the reviewer may give, otherwise the first. */
export function initialReviewLabel(label: DecisionLabel | null, choices: string[]): string {
  const value = label ? (LEGACY_LABELS[label.label] ?? label.label) : '';
  return choices.includes(value) ? value : (choices[0] ?? '');
}

export type LabelState = 'unlabelled' | 'llm' | 'reviewed';

/** Whether a decision has no label, an LLM label awaiting review, or a
 * reviewed one (a reviewer's own, or an approved LLM label). */
export function labelState(decision: RoutingDecision): LabelState {
  if (!decision.label) return 'unlabelled';
  return decision.label.human_reviewed ? 'reviewed' : 'llm';
}

/** Whether a reviewer may approve the decision's label as it is: an LLM
 * label the LLM was sure of. One it flagged for review needs a reviewer's
 * own label. */
export function approvable(decision: RoutingDecision): boolean {
  return labelState(decision) === 'llm' && !decision.label!.requires_review;
}

export const LABEL_FILTERS: { value: LabelState | 'all'; label: string }[] = [
  { value: 'all', label: 'All decisions' },
  { value: 'llm', label: 'LLM labels to review' },
  { value: 'unlabelled', label: 'Unlabelled' },
  { value: 'reviewed', label: 'Reviewed' },
];

export interface CalibrationPoint {
  /** The mean confidence of the bin's decisions. */
  confidence: number;
  /** The share of the bin's decisions that succeeded. */
  success_rate: number;
  decisions: number;
}

/** Decisions in ``bins`` equal-width confidence bins spanning the lowest to
 * the highest confidence, as pandas ``cut`` makes them (each bin closed on
 * the right, the first widened by 0.1% of the span so it holds the lowest),
 * with each non-empty bin's mean confidence and success rate, lowest first. */
export function calibrationPoints(decisions: RoutingDecision[], bins = 10): CalibrationPoint[] {
  if (!decisions.length) return [];
  const confidences = decisions.map((decision) => decision.confidence);
  let low = Math.min(...confidences);
  let high = Math.max(...confidences);
  const span = high - low;
  if (span === 0) {
    const pad = low === 0 ? 0.001 : Math.abs(low) * 0.001;
    low -= pad;
    high += pad;
  }
  const edges = Array.from({ length: bins + 1 }, (_, index) =>
    index === bins ? high : low + ((high - low) * index) / bins,
  );
  if (span !== 0) edges[0] -= span * 0.001;
  const grouped = new Map<number, RoutingDecision[]>();
  for (const decision of decisions) {
    const index = edges.findIndex((edge, at) => at > 0 && decision.confidence <= edge) - 1;
    grouped.set(index, [...(grouped.get(index) ?? []), decision]);
  }
  return [...grouped.entries()]
    .sort(([a], [b]) => a - b)
    .map(([, members]) => ({
      confidence: members.reduce((sum, decision) => sum + decision.confidence, 0) / members.length,
      success_rate: members.filter((decision) => decision.outcome === 'success').length / members.length,
      decisions: members.length,
    }));
}

/** The start of ``iso``'s UTC hour, as ``YYYY-MM-DDTHH:00:00Z``. */
function hourOf(iso: string): string {
  return `${new Date(iso).toISOString().slice(0, 13)}:00:00Z`;
}

export interface AgentHours {
  agent: string;
  /** The UTC hours with a decision for the agent, oldest first. */
  hours: string[];
  decisions: number[];
}

/** Each agent's decisions per UTC hour, agents by name; an hour without a
 * decision for the agent is left out of its series. */
export function decisionsPerHourByAgent(decisions: RoutingDecision[]): AgentHours[] {
  const counts = new Map<string, Map<string, number>>();
  for (const decision of decisions) {
    const hours = counts.get(decision.chosen_agent) ?? new Map<string, number>();
    const hour = hourOf(decision.start_time);
    hours.set(hour, (hours.get(hour) ?? 0) + 1);
    counts.set(decision.chosen_agent, hours);
  }
  return [...counts.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([agent, hours]) => {
      const ordered = [...hours.entries()].sort(([a], [b]) => a.localeCompare(b));
      return { agent, hours: ordered.map(([hour]) => hour), decisions: ordered.map(([, count]) => count) };
    });
}

export interface HourRate {
  hour: string;
  /** The share of the hour's decisions that succeeded; ``null`` for an hour
   * without decisions. */
  success_rate: number | null;
}

/** The success rate of every UTC hour from the first decision's to the
 * last's, oldest first. */
export function successRatePerHour(decisions: RoutingDecision[]): HourRate[] {
  if (!decisions.length) return [];
  const tally = new Map<string, { decisions: number; successes: number }>();
  for (const decision of decisions) {
    const hour = hourOf(decision.start_time);
    const count = tally.get(hour) ?? { decisions: 0, successes: 0 };
    count.decisions += 1;
    if (decision.outcome === 'success') count.successes += 1;
    tally.set(hour, count);
  }
  const ordered = [...tally.keys()].sort();
  const rates: HourRate[] = [];
  const last = Date.parse(ordered[ordered.length - 1]);
  for (let at = Date.parse(ordered[0]); at <= last; at += 3_600_000) {
    const hour = hourOf(new Date(at).toISOString());
    const count = tally.get(hour);
    rates.push({ hour, success_rate: count ? count.successes / count.decisions : null });
  }
  return rates;
}

/** A ``value`` from 0 to 1 as a red-yellow-green background, red at 0. */
export function scoreColor(value: number): string {
  const stops = [
    [215, 48, 39],
    [255, 255, 191],
    [26, 152, 80],
  ];
  const at = Math.min(Math.max(value, 0), 1) * 2;
  const index = Math.min(Math.floor(at), 1);
  const share = at - index;
  const channel = (c: number) => Math.round(stops[index][c] + (stops[index + 1][c] - stops[index][c]) * share);
  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
}

export type Priority = 'high' | 'medium' | 'low';

export const PRIORITIES: Priority[] = ['high', 'medium', 'low'];

/** A decision the annotation agent says needs a reviewer. */
export interface AnnotationCandidate {
  span_id: string;
  start_time: string;
  query: string;
  chosen_agent: string;
  confidence: number;
  outcome: string;
  priority: Priority;
  reason: string;
}

/** The candidates of a chosen priority, without those the LLM has labelled
 * unless ``showLlmLabelled``; ``labels`` holds each decision's label by span
 * ID. */
export function shownCandidates(
  candidates: AnnotationCandidate[],
  priorities: Priority[],
  showLlmLabelled: boolean,
  labels: Record<string, DecisionLabel | null>,
): AnnotationCandidate[] {
  return candidates.filter(
    (candidate) =>
      priorities.includes(candidate.priority) &&
      (showLlmLabelled || labels[candidate.span_id]?.annotator !== 'llm'),
  );
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
