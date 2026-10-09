import { RuntimeRequestError } from './http';

/** One optimizer's example schema as ``GET /admin/tenant/training-example-templates`` serves it. */
export interface ExampleTemplate {
  schema: string;
  fields: string[];
  required: string[];
  example: Record<string, unknown>;
}

export interface ExampleTemplates {
  templates: Record<string, ExampleTemplate>;
  max_examples: number;
}

/** The file a template download saves for ``optimizer``. */
export function templateFile(optimizer: string, template: ExampleTemplate): { name: string; text: string } {
  return {
    name: `${optimizer}_examples.json`,
    text: JSON.stringify({ optimizer, examples: [template.example] }, null, 2),
  };
}

export type FileCheck =
  | { ok: true; optimizer: string; examples: Record<string, unknown>[]; summary: string; preview: string }
  | { ok: false; errors: string[]; preview?: string };

/**
 * Whether ``text`` is a training-examples file the runtime takes: a JSON
 * object naming a known ``optimizer`` and holding ``examples``, each an
 * object with every required field of that optimizer's schema and no other.
 * The runtime validates the values again on upload.
 */
export function checkTrainingFile(text: string, catalog: ExampleTemplates): FileCheck {
  let content: unknown;
  try {
    content = JSON.parse(text);
  } catch (error) {
    return { ok: false, errors: [`Invalid JSON: ${error instanceof Error ? error.message : String(error)}`] };
  }
  const preview = JSON.stringify(content, null, 2);
  const known = Object.keys(catalog.templates).sort();
  if (content === null || typeof content !== 'object' || Array.isArray(content))
    return { ok: false, preview, errors: ['Expected a JSON object with "optimizer" and "examples" keys.'] };
  const { optimizer, examples, ...rest } = content as Record<string, unknown>;
  const errors: string[] = [];
  const extra = Object.keys(rest);
  if (extra.length) errors.push(`Unexpected keys: ${extra.join(', ')}.`);
  const template = typeof optimizer === 'string' ? catalog.templates[optimizer] : undefined;
  if (!template) errors.push(`"optimizer" must be one of ${known.join(', ')}.`);
  if (!Array.isArray(examples) || examples.length === 0) errors.push('"examples" must be a non-empty list.');
  else if (examples.length > catalog.max_examples)
    errors.push(`A file holds at most ${catalog.max_examples} examples, not ${examples.length}.`);
  if (errors.length || !template || !Array.isArray(examples)) return { ok: false, preview, errors };
  examples.forEach((example, index) => {
    if (example === null || typeof example !== 'object' || Array.isArray(example)) {
      errors.push(`examples[${index}] must be a JSON object.`);
      return;
    }
    const keys = Object.keys(example);
    const missing = template.required.filter((field) => !keys.includes(field));
    const unknown = keys.filter((field) => !template.fields.includes(field));
    if (missing.length) errors.push(`examples[${index}] is missing ${missing.join(', ')}.`);
    if (unknown.length) errors.push(`examples[${index}] has unknown fields ${unknown.join(', ')}.`);
  });
  if (errors.length) return { ok: false, preview, errors };
  const count = examples.length;
  return {
    ok: true,
    optimizer: optimizer as string,
    examples: examples as Record<string, unknown>[],
    preview,
    summary: `Valid ${optimizer} examples file (${count} ${count === 1 ? 'example' : 'examples'}).`,
  };
}

/** What a run's 404 means: Argo deleted it when its time-to-live expired. */
export function runActionError(name: string, error: unknown): string {
  if (error instanceof RuntimeRequestError && error.status === 404)
    return `Run ${name} no longer exists; Argo deleted it when its time-to-live expired.`;
  return error instanceof Error ? error.message : String(error);
}

/** An agent event the report route streams. */
export interface ReportEvent {
  type: string;
  phase?: string;
  message?: string;
  data?: Record<string, unknown>;
}

export interface ReportProgress {
  status: string;
  text: string;
  report?: Record<string, unknown>;
  error?: string;
}

export const REPORT_START: ReportProgress = { status: 'Starting the report agent…', text: '' };

/** ``progress`` after the report stream's next ``event``. */
export function reportProgress(progress: ReportProgress, event: ReportEvent): ReportProgress {
  const data = event.data ?? {};
  if (event.type === 'status') return { ...progress, status: event.message ?? event.phase ?? progress.status };
  if (event.type === 'partial') {
    if (event.phase === 'token' && typeof data.accumulated === 'string') return { ...progress, text: data.accumulated };
    if (Array.isArray(data.themes))
      return { ...progress, status: `Themes: ${(data.themes as unknown[]).slice(0, 3).join(', ')}` };
    if (typeof data.summary === 'string') return { ...progress, text: data.summary };
    return progress;
  }
  if (event.type === 'final') return { ...progress, status: '', report: data };
  if (event.type === 'error')
    return { ...progress, status: '', error: event.message ?? 'The report agent failed.' };
  return progress;
}

/** The executive summary and recommendations of a finished report. */
export function reportSummary(report: Record<string, unknown>): { summary: string; recommendations: string[] } {
  const body = (
    typeof report.executive_summary === 'string' ? report : (report.result as Record<string, unknown> | undefined) ?? {}
  ) as Record<string, unknown>;
  const recommendations = Array.isArray(body.recommendations)
    ? body.recommendations.filter((item): item is string => typeof item === 'string')
    : [];
  return { summary: typeof body.executive_summary === 'string' ? body.executive_summary : '', recommendations };
}

/** ``optimization_report_YYYYMMDD_HHMMSS.json`` for local time ``at``. */
export function reportFilename(at: Date): string {
  const two = (n: number) => String(n).padStart(2, '0');
  return (
    `optimization_report_${at.getFullYear()}${two(at.getMonth() + 1)}${two(at.getDate())}_` +
    `${two(at.getHours())}${two(at.getMinutes())}${two(at.getSeconds())}.json`
  );
}

/** Saves ``text`` as a JSON file named ``name``. */
export function saveJson(name: string, text: string): void {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}
