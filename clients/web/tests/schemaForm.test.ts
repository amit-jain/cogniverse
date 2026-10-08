import { describe, expect, it } from 'vitest';
import { formFields, fromFormState, toFormState, type JsonSchema } from '../src/client/ops/schemaForm';

// Excerpts of the runtime's telemetry and agent section schemas.
const schema: JsonSchema = {
  type: 'object',
  properties: {
    enabled: { type: 'boolean', default: true },
    level: { $ref: '#/$defs/TelemetryLevel', default: 'detailed' },
    provider: { anyOf: [{ type: 'string' }, { type: 'null' }], default: null },
    max_cached_tenants: { type: 'integer' },
    retirement_lease_timeout_seconds: { type: 'number' },
    batch_config: { $ref: '#/$defs/BatchExportConfig' },
    extra_resource_attributes: { type: 'object', additionalProperties: { type: 'string' } },
    capabilities: { type: 'array', items: { type: 'string' } },
    llm_max_tokens: { anyOf: [{ type: 'integer' }, { type: 'null' }], default: null },
    optimizer_config: { anyOf: [{ $ref: '#/$defs/OptimizerConfig' }, { type: 'null' }], default: null },
    llm_api_key: { anyOf: [{ type: 'string' }, { type: 'null' }], default: null, writeOnly: true },
  },
  $defs: {
    TelemetryLevel: { description: 'Telemetry collection levels.', enum: ['disabled', 'basic', 'detailed', 'verbose'], type: 'string' },
    BatchExportConfig: {
      type: 'object',
      properties: { max_queue_size: { type: 'integer' }, use_sync_export: { type: 'boolean' } },
    },
    OptimizerConfig: { type: 'object', properties: { optimizer_type: { type: 'string' } } },
  },
};

const value = {
  enabled: true,
  level: 'basic',
  provider: null,
  max_cached_tenants: 100,
  retirement_lease_timeout_seconds: 2.5,
  batch_config: { max_queue_size: 2048, use_sync_export: false },
  extra_resource_attributes: { team: 'search' },
  capabilities: ['video'],
  llm_max_tokens: null,
  optimizer_config: null,
  llm_api_key: null,
};

describe('formFields', () => {
  it('gives each property its input, resolving refs and nullable branches', () => {
    expect(formFields(schema)).toEqual([
      { name: 'enabled', kind: 'boolean', nullable: false, description: undefined },
      {
        name: 'level',
        kind: 'choice',
        nullable: false,
        options: ['disabled', 'basic', 'detailed', 'verbose'],
        description: 'Telemetry collection levels.',
      },
      { name: 'provider', kind: 'text', nullable: true, description: undefined },
      { name: 'max_cached_tenants', kind: 'integer', nullable: false, description: undefined },
      { name: 'retirement_lease_timeout_seconds', kind: 'number', nullable: false, description: undefined },
      {
        name: 'batch_config',
        kind: 'object',
        nullable: false,
        description: undefined,
        fields: [
          { name: 'max_queue_size', kind: 'integer', nullable: false, description: undefined },
          { name: 'use_sync_export', kind: 'boolean', nullable: false, description: undefined },
        ],
      },
      { name: 'extra_resource_attributes', kind: 'json', nullable: false, empty: {}, description: undefined },
      { name: 'capabilities', kind: 'json', nullable: false, empty: [], description: undefined },
      { name: 'llm_max_tokens', kind: 'integer', nullable: true, description: undefined },
      { name: 'optimizer_config', kind: 'json', nullable: true, empty: {}, description: undefined },
      { name: 'llm_api_key', kind: 'secret', nullable: true, description: undefined },
    ]);
  });
});

describe('toFormState and fromFormState', () => {
  const fields = formFields(schema);

  it('round-trips a value through the form unchanged', () => {
    expect(fromFormState(fields, toFormState(fields, value))).toEqual(value);
  });

  it('reads edited inputs back as typed values', () => {
    const state = toFormState(fields, value);
    state.provider = 'phoenix';
    state.max_cached_tenants = '250';
    state.llm_max_tokens = '';
    state.capabilities = '["video", "audio"]';
    (state.batch_config as { max_queue_size: string }).max_queue_size = '64';
    state.llm_api_key = { text: 'sk-1', clear: false };
    expect(fromFormState(fields, state)).toEqual({
      ...value,
      provider: 'phoenix',
      max_cached_tenants: 250,
      llm_max_tokens: null,
      capabilities: ['video', 'audio'],
      batch_config: { max_queue_size: 64, use_sync_export: false },
      llm_api_key: 'sk-1',
    });
  });

  it('sends an empty string for a secret marked to clear', () => {
    const state = toFormState(fields, value);
    state.llm_api_key = { text: 'ignored', clear: true };
    expect(fromFormState(fields, state).llm_api_key).toBe('');
  });

  it('names the field an input cannot be read as', () => {
    const state = toFormState(fields, value);
    (state.batch_config as { max_queue_size: string }).max_queue_size = '1.5';
    expect(() => fromFormState(fields, state)).toThrow('batch_config.max_queue_size must be a whole number.');
    const json = toFormState(fields, value);
    json.extra_resource_attributes = '{team:';
    expect(() => fromFormState(fields, json)).toThrow('extra_resource_attributes is not valid JSON.');
    const number = toFormState(fields, value);
    number.retirement_lease_timeout_seconds = 'soon';
    expect(() => fromFormState(fields, number)).toThrow('retirement_lease_timeout_seconds must be a number.');
  });

  it('reads a blank JSON input as its empty value or, when nullable, null', () => {
    const state = toFormState(fields, value);
    state.capabilities = '';
    state.optimizer_config = '';
    const read = fromFormState(fields, state);
    expect([read.capabilities, read.optimizer_config]).toEqual([[], null]);
  });
});
