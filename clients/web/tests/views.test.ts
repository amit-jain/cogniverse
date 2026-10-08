import { describe, expect, it } from 'vitest';
import { OPS_VIEWS } from '../src/client/ops/views';

describe('OPS_VIEWS', () => {
  it('gives every view one sentence under its heading', () => {
    const notOneSentence = OPS_VIEWS.filter((view) => !/^[A-Z][^.]*\.$/.test(view.description ?? ''));
    expect(notOneSentence.map((view) => view.id)).toEqual([]);
    expect(OPS_VIEWS.find((view) => view.id === 'ingestion')?.description).toBe(
      'Interactive testing and configuration of ingestion pipelines with different processing profiles.',
    );
  });
});
