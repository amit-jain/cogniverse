import { jsonText, type JsonObject } from './forms';

/** The create form's fields, as the text each input holds. */
export interface ProfileFields {
  type: string;
  description: string;
  schemaName: string;
  embeddingModel: string;
  embeddingType: string;
  modelLoader: string;
  processType: string;
  pipeline: string;
  strategies: string;
  schemaConfig: string;
  modelSpecific: string;
  extraConfig: string;
}

// The profile keys the create request takes as fields; any other key of a
// shipped profile goes in extra_config.
const NAMED_KEYS = new Set([
  'type',
  'description',
  'schema_name',
  'embedding_model',
  'embedding_type',
  'model_loader',
  'process_type',
  'pipeline_config',
  'strategies',
  'schema_config',
  'model_specific',
]);

const text = (value: unknown) => (typeof value === 'string' ? value : '');
const objectText = (value: unknown) =>
  value && typeof value === 'object' && Object.keys(value).length ? jsonText(value) : '';

/** The create form filled from a shipped profile's configuration. */
export function fieldsFromTemplate(config: JsonObject): ProfileFields {
  const extra = Object.fromEntries(Object.entries(config).filter(([key]) => !NAMED_KEYS.has(key)));
  return {
    type: text(config.type),
    description: text(config.description),
    schemaName: text(config.schema_name),
    embeddingModel: text(config.embedding_model),
    embeddingType: text(config.embedding_type),
    modelLoader: text(config.model_loader),
    processType: text(config.process_type),
    pipeline: objectText(config.pipeline_config),
    strategies: objectText(config.strategies),
    schemaConfig: objectText(config.schema_config),
    modelSpecific: objectText(config.model_specific),
    extraConfig: objectText(extra),
  };
}

/** The distinct non-empty values of ``key`` across shipped profiles, sorted. */
export function shippedValues(templates: { config: JsonObject }[], key: string): string[] {
  return [...new Set(templates.map((t) => text(t.config[key])).filter(Boolean))].sort();
}
