import { describe, expect, it } from 'vitest';
import { BLANK_PROFILE, fieldsFromTemplate, withCurrent } from '../src/client/ops/profiles';

const documentText = {
  type: 'document',
  description: 'Document text',
  schema_name: 'document_text',
  result_granularity: 'source',
  embedding_model: 'lightonai/LateOn',
  pipeline_config: { generate_embeddings: true },
  strategies: { embedding: { class: 'DocumentTextEmbeddingStrategy', params: {} } },
  embedding_type: 'multi_vector',
  model_loader: 'colbert',
  inference_services: { embedding: 'colbert_pylate' },
  schema_config: { schema_name: 'document_text', embedding_dim: 128 },
};

describe('fieldsFromTemplate', () => {
  it('puts named keys in their fields and the rest in extra config', () => {
    expect(fieldsFromTemplate(documentText)).toEqual({
      type: 'document',
      description: 'Document text',
      schemaName: 'document_text',
      embeddingModel: 'lightonai/LateOn',
      embeddingType: 'multi_vector',
      modelLoader: 'colbert',
      processType: '',
      pipeline: JSON.stringify({ generate_embeddings: true }, null, 2),
      strategies: JSON.stringify({ embedding: { class: 'DocumentTextEmbeddingStrategy', params: {} } }, null, 2),
      schemaConfig: JSON.stringify({ schema_name: 'document_text', embedding_dim: 128 }, null, 2),
      modelSpecific: '',
      extraConfig: JSON.stringify(
        { result_granularity: 'source', inference_services: { embedding: 'colbert_pylate' } },
        null,
        2,
      ),
    });
  });

  it('leaves blank what a profile does not set', () => {
    expect(fieldsFromTemplate({ type: 'wiki', schema_name: 'wiki_page' })).toEqual({
      type: 'wiki',
      description: '',
      schemaName: 'wiki_page',
      embeddingModel: '',
      embeddingType: '',
      modelLoader: '',
      processType: '',
      pipeline: '',
      strategies: '',
      schemaConfig: '',
      modelSpecific: '',
      extraConfig: '',
    });
  });
});

describe('withCurrent', () => {
  it('keeps a value the choices lack selectable, first', () => {
    expect(withCurrent(['colbert', 'colpali'], 'videoprism')).toEqual(['videoprism', 'colbert', 'colpali']);
    expect(withCurrent(['colbert', 'colpali'], 'colpali')).toEqual(['colbert', 'colpali']);
    expect(withCurrent(['colbert', 'colpali'], '')).toEqual(['colbert', 'colpali']);
  });
});

describe('BLANK_PROFILE', () => {
  it('starts a blank profile from the frame-based ColPali layout', () => {
    expect({
      ...BLANK_PROFILE,
      pipeline: JSON.parse(BLANK_PROFILE.pipeline),
      strategies: JSON.parse(BLANK_PROFILE.strategies),
      schemaConfig: JSON.parse(BLANK_PROFILE.schemaConfig),
    }).toEqual({
      type: 'video',
      description: '',
      schemaName: '',
      embeddingModel: '',
      embeddingType: 'multi_vector',
      modelLoader: 'colpali',
      processType: '',
      pipeline: { extract_keyframes: true, transcribe_audio: false, generate_descriptions: false, keyframe_fps: 0.5 },
      strategies: {
        segmentation: { class: 'FrameSegmentationStrategy', params: { fps: 0.5, max_frames: 100 } },
        embedding: { class: 'MultiVectorEmbeddingStrategy', params: {} },
      },
      schemaConfig: {
        schema_name: 'video_colpali_smol500_mv_frame',
        model_name: 'TomoroAI/tomoro-colqwen3-embed-4b',
        embedding_dim: 320,
        binary_dim: 40,
      },
      modelSpecific: '{}',
      extraConfig: '',
    });
  });
});
