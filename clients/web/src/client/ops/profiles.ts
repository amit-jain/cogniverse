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

/** A blank profile's form: the shipped frame-based ColPali layout to edit
 * from, with an empty model-specific block. */
export const BLANK_PROFILE: ProfileFields = {
  type: 'video',
  description: '',
  schemaName: '',
  embeddingModel: '',
  embeddingType: 'multi_vector',
  modelLoader: 'colpali',
  processType: '',
  pipeline: jsonText({
    extract_keyframes: true,
    transcribe_audio: false,
    generate_descriptions: false,
    keyframe_fps: 0.5,
  }),
  strategies: jsonText({
    segmentation: { class: 'FrameSegmentationStrategy', params: { fps: 0.5, max_frames: 100 } },
    embedding: { class: 'MultiVectorEmbeddingStrategy', params: {} },
  }),
  schemaConfig: jsonText({
    schema_name: 'video_colpali_smol500_mv_frame',
    model_name: 'TomoroAI/tomoro-colqwen3-embed-4b',
    embedding_dim: 320,
    binary_dim: 40,
  }),
  modelSpecific: '{}',
  extraConfig: '',
};

/** The values the create form's choice fields take, as the runtime lists
 * them beside the shipped profiles. */
export interface ProfileChoices {
  profile_types: string[];
  embedding_types: string[];
  model_loaders: string[];
  process_types: string[];
}

/** ``options`` with ``current`` first when it is set and not one of them,
 * so a value a shipped profile carries stays selectable. */
export function withCurrent(options: string[], current: string): string[] {
  return current && !options.includes(current) ? [current, ...options] : options;
}
