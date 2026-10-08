import { describe, expect, it } from 'vitest';
import { fieldsFromTemplate, shippedValues } from '../src/client/ops/profiles';

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

describe('shippedValues', () => {
  it('lists each value once, sorted, without blanks', () => {
    const templates = [
      { config: { model_loader: 'colpali' } },
      { config: { model_loader: 'colbert' } },
      { config: { model_loader: 'colpali' } },
      { config: {} },
    ];
    expect(shippedValues(templates, 'model_loader')).toEqual(['colbert', 'colpali']);
  });
});
