import { describe, expect, it } from 'vitest';
import { formFields, fromFormState, toFormState, type JsonSchema } from '../src/client/ops/schemaForm';

// Excerpts of the runtime's telemetry and agent section schemas.
const schema: JsonSchema = {
  type: 'object',
  properties: {
    enabled: { type: 'boolean', default: true },
    level: { $ref: '#/$defs/TelemetryLevel', default: 'detailed' },
    provider: { anyOf: [{ type: 'string', enum: ['phoenix'] }, { type: 'null' }], default: null },
    max_cached_tenants: { type: 'integer', minimum: 1, maximum: 65535 },
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
    OptimizerConfig: {
      type: 'object',
      properties: {
        optimizer_type: { $ref: '#/$defs/OptimizerType' },
        num_trials: { type: 'integer', default: 10 },
        metric: { anyOf: [{ type: 'string' }, { type: 'null' }], default: null },
        teacher_settings: { type: 'object', additionalProperties: true },
      },
      required: ['optimizer_type'],
    },
    OptimizerType: { enum: ['bootstrap_few_shot', 'mipro_v2'], type: 'string' },
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
      { name: 'enabled', kind: 'boolean', nullable: false, description: undefined, default: true },
      {
        name: 'level',
        kind: 'choice',
        nullable: false,
        options: ['disabled', 'basic', 'detailed', 'verbose'],
        description: 'Telemetry collection levels.',
        default: 'detailed',
      },
      { name: 'provider', kind: 'choice', nullable: true, options: ['phoenix'], description: undefined, default: null },
      { name: 'max_cached_tenants', kind: 'integer', nullable: false, description: undefined, min: 1, max: 65535 },
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
      { name: 'llm_max_tokens', kind: 'integer', nullable: true, description: undefined, default: null },
      {
        name: 'optimizer_config',
        kind: 'optional',
        nullable: true,
        description: undefined,
        default: null,
        fields: [
          {
            name: 'optimizer_type',
            kind: 'choice',
            nullable: false,
            options: ['bootstrap_few_shot', 'mipro_v2'],
            description: undefined,
          },
          { name: 'num_trials', kind: 'integer', nullable: false, description: undefined, default: 10 },
          { name: 'metric', kind: 'text', nullable: true, description: undefined, default: null },
          { name: 'teacher_settings', kind: 'json', nullable: false, empty: {}, description: undefined },
        ],
      },
      { name: 'llm_api_key', kind: 'secret', nullable: true, description: undefined, default: null },
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

  it('reads a blank JSON input as its empty value', () => {
    const state = toFormState(fields, value);
    state.capabilities = '';
    expect(fromFormState(fields, state).capabilities).toEqual([]);
  });

  it('sets an optional object from its defaults and leaves it null when unset', () => {
    const state = toFormState(fields, value);
    expect(state.optimizer_config).toEqual({
      set: false,
      fields: { optimizer_type: 'bootstrap_few_shot', num_trials: '10', metric: '', teacher_settings: '{}' },
    });
    expect(fromFormState(fields, state).optimizer_config).toBeNull();
    const optimizer = state.optimizer_config as { set: boolean; fields: Record<string, string> };
    optimizer.set = true;
    optimizer.fields.optimizer_type = 'mipro_v2';
    optimizer.fields.teacher_settings = '{"temperature": 0.2}';
    expect(fromFormState(fields, state).optimizer_config).toEqual({
      optimizer_type: 'mipro_v2',
      num_trials: 10,
      metric: null,
      teacher_settings: { temperature: 0.2 },
    });
  });

  it('reads a stored optional object back as set, with its values', () => {
    const stored = {
      ...value,
      optimizer_config: { optimizer_type: 'mipro_v2', num_trials: 3, metric: 'f1', teacher_settings: {} },
    };
    const state = toFormState(fields, stored);
    expect(state.optimizer_config).toEqual({
      set: true,
      fields: { optimizer_type: 'mipro_v2', num_trials: '3', metric: 'f1', teacher_settings: '{}' },
    });
    expect(fromFormState(fields, state)).toEqual(stored);
  });

  it('refuses a number outside its bounds, naming them', () => {
    const state = toFormState(fields, value);
    state.max_cached_tenants = '70000';
    expect(() => fromFormState(fields, state)).toThrow('max_cached_tenants must be between 1 and 65535.');
    state.max_cached_tenants = '0';
    expect(() => fromFormState(fields, state)).toThrow('max_cached_tenants must be between 1 and 65535.');
  });

  it('reads an unset nullable choice as null', () => {
    const state = toFormState(fields, value);
    expect(state.provider).toBe('');
    expect(fromFormState(fields, state).provider).toBeNull();
    state.provider = 'phoenix';
    expect(fromFormState(fields, state).provider).toBe('phoenix');
  });
});
