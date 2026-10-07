import { jsonText, type JsonObject } from './forms';

/** The subset of JSON Schema the runtime's config sections use. */
export interface JsonSchema {
  type?: string | string[];
  properties?: Record<string, JsonSchema>;
  required?: string[];
  enum?: unknown[];
  anyOf?: JsonSchema[];
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  items?: JsonSchema;
  additionalProperties?: boolean | JsonSchema;
  writeOnly?: boolean;
  description?: string;
  default?: unknown;
}

export type FieldKind = 'text' | 'integer' | 'number' | 'boolean' | 'choice' | 'object' | 'json' | 'secret';

/** One input of a generated form; ``fields`` for a nested object. */
export interface FormField {
  name: string;
  kind: FieldKind;
  nullable: boolean;
  options?: string[];
  fields?: FormField[];
  /** For ``json`` fields: what a blank input stands for when not nullable. */
  empty?: unknown;
  description?: string;
}

/** What a form holds: input text, checkbox state, or a nested object's state. */
export interface SecretInput {
  text: string;
  clear: boolean;
}
export type FormState = { [name: string]: string | boolean | SecretInput | FormState };

/** The fields of an object schema, ``$ref``s and nullable branches resolved. */
export function formFields(schema: JsonSchema, defs: Record<string, JsonSchema> = schema.$defs ?? {}): FormField[] {
  return Object.entries(schema.properties ?? {}).map(([name, property]) => field(name, property, defs));
}

function resolve(schema: JsonSchema, defs: Record<string, JsonSchema>): JsonSchema {
  let node = schema;
  while (node.$ref) node = defs[node.$ref.split('/').pop() ?? ''] ?? {};
  return node;
}

function field(name: string, property: JsonSchema, defs: Record<string, JsonSchema>): FormField {
  let node = resolve(property, defs);
  let nullable = false;
  if (node.anyOf) {
    const branches = node.anyOf.map((branch) => resolve(branch, defs));
    nullable = branches.some((branch) => branch.type === 'null');
    const others = branches.filter((branch) => branch.type !== 'null');
    node = others.length === 1 ? others[0] : { type: 'json' };
  }
  const base = { name, nullable, description: property.description ?? node.description };
  if (property.writeOnly) return { ...base, kind: 'secret' };
  if (node.enum) return { ...base, kind: 'choice', options: node.enum.map(String) };
  // A nullable object is edited as JSON, where null is expressible.
  if (node.properties && !nullable) return { ...base, kind: 'object', fields: formFields(node, defs) };
  if (node.properties) return { ...base, kind: 'json', empty: {} };
  switch (node.type) {
    case 'string':
      return { ...base, kind: 'text' };
    case 'integer':
      return { ...base, kind: 'integer' };
    case 'number':
      return { ...base, kind: 'number' };
    case 'boolean':
      return { ...base, kind: 'boolean' };
    case 'array':
      return { ...base, kind: 'json', empty: [] };
    default:
      return { ...base, kind: 'json', empty: {} };
  }
}

/** ``value`` as the form's inputs hold it. */
export function toFormState(fields: FormField[], value: JsonObject): FormState {
  return Object.fromEntries(
    fields.map((f) => {
      const item = value[f.name];
      switch (f.kind) {
        case 'boolean':
          return [f.name, Boolean(item)];
        case 'object':
          return [f.name, toFormState(f.fields ?? [], (item ?? {}) as JsonObject)];
        case 'json':
          return [f.name, item === null || item === undefined ? '' : jsonText(item)];
        case 'secret':
          return [f.name, { text: '', clear: false }];
        default:
          return [f.name, item === null || item === undefined ? '' : String(item)];
      }
    }),
  );
}

/**
 * The value a form's inputs describe. A blank input is null for a nullable
 * field; a secret left blank is null (kept) and one marked to clear is
 * ``""``. Throws naming the field for an input that is not a number or not
 * JSON.
 */
export function fromFormState(fields: FormField[], state: FormState, path = ''): JsonObject {
  return Object.fromEntries(
    fields.map((f) => {
      const at = path ? `${path}.${f.name}` : f.name;
      const input = state[f.name];
      switch (f.kind) {
        case 'boolean':
          return [f.name, Boolean(input)];
        case 'object':
          return [f.name, fromFormState(f.fields ?? [], input as FormState, at)];
        case 'secret': {
          const secret = input as SecretInput;
          return [f.name, secret.clear ? '' : secret.text || null];
        }
        default:
          return [f.name, scalar(f, String(input ?? ''), at)];
      }
    }),
  );
}

function scalar(f: FormField, text: string, at: string): unknown {
  const blank = !text.trim();
  if (blank && f.nullable) return null;
  switch (f.kind) {
    case 'integer': {
      const value = Number(text);
      if (blank || !Number.isInteger(value)) throw new Error(`${at} must be a whole number.`);
      return value;
    }
    case 'number': {
      const value = Number(text);
      if (blank || Number.isNaN(value)) throw new Error(`${at} must be a number.`);
      return value;
    }
    case 'json':
      if (blank) return f.empty;
      try {
        return JSON.parse(text);
      } catch {
        throw new Error(`${at} is not valid JSON.`);
      }
    default:
      return text;
  }
}
