import { describe, expect, it } from 'vitest';
import { agentHue, agentLabel } from '../src/client/api';
import { clock, resultsOf, searchSpanOf } from '../src/client/ResultCards';

describe('resultsOf', () => {
  it('reads public-shaped single-profile hits in order', () => {
    const state = {
      agent: 'search_agent',
      result: {
        query: 'cats',
        results: [
          {
            id: 'v1_seg_3',
            document_id: 'id:video:video::v1_seg_3',
            score: 0.91,
            metadata: { video_id: 'v1', video_title: 'Cats', audio_transcript: 'a cat jumps' },
            temporal_info: { start_time: 75.4, end_time: 81 },
          },
          {
            id: 'v2_seg_0',
            document_id: 'id:video:video::v2_seg_0',
            score: 0.42,
            metadata: { video_id: 'v2', description: 'a dog' },
          },
        ],
      },
    };
    expect(resultsOf(state)).toEqual([
      {
        id: 'v1_seg_3',
        ratingId: 'id:video:video::v1_seg_3',
        score: 0.91,
        title: 'Cats',
        snippet: 'a cat jumps',
        start: 75.4,
        end: 81,
      },
      {
        id: 'v2_seg_0',
        ratingId: 'id:video:video::v2_seg_0',
        score: 0.42,
        title: undefined,
        snippet: 'a dog',
        start: undefined,
        end: undefined,
      },
    ]);
  });

  it('ranks an ensemble hit by its rrf_score, not the per-profile score', () => {
    const state = {
      result: { results: [{ id: 'a', score: 12.5, rrf_score: 0.033, metadata: {} }] },
    };
    expect(resultsOf(state)).toEqual([
      { id: 'a', ratingId: 'a', score: 0.033, title: undefined, snippet: undefined, start: undefined, end: undefined },
    ]);
  });

  it('falls back through document_id and metadata.video_id for identity', () => {
    const state = {
      result: {
        results: [
          { id: '', document_id: 'doc-7', metadata: {} },
          { metadata: { video_id: 'v9' } },
          { metadata: { title: 'no identity' } },
          null,
          'junk',
        ],
      },
    };
    expect(resultsOf(state).map((hit) => hit.id)).toEqual(['doc-7', 'v9']);
  });

  it('rates a hit by the id its search span records it under', () => {
    const state = {
      result: {
        results: [
          { id: 'seg_1', document_id: 'doc_1', video_id: 'v1' },
          { id: 'seg_2', documentid: 'doc_2' },
          { id: 'seg_3', video_id: 'v3' },
          { source_id: 'src_4', video_id: 'v4', metadata: { video_id: 'v4' } },
          { video_id: 'v5' },
          { metadata: { video_id: 'v6' } },
        ],
      },
    };
    expect(resultsOf(state).map((hit) => [hit.id, hit.ratingId])).toEqual([
      ['seg_1', 'doc_1'],
      ['seg_2', 'doc_2'],
      ['seg_3', 'seg_3'],
      ['v4', 'src_4'],
      ['v5', 'v5'],
      ['v6', undefined],
    ]);
  });

  it('returns no hits for a payload without a results array', () => {
    expect(resultsOf(undefined)).toEqual([]);
    expect(resultsOf({})).toEqual([]);
    expect(resultsOf({ result: { answer: 'hi' } })).toEqual([]);
    expect(resultsOf({ result: { results: 'nope' } })).toEqual([]);
  });
});

describe('searchSpanOf', () => {
  it("reads the search's span id from the final payload", () => {
    expect(searchSpanOf({ result: { span_id: '00000000000000ab', results: [] } })).toBe('00000000000000ab');
    expect(searchSpanOf({ result: { span_id: null, results: [] } })).toBeUndefined();
    expect(searchSpanOf({ result: { results: [] } })).toBeUndefined();
    expect(searchSpanOf(undefined)).toBeUndefined();
  });
});

describe('labels', () => {
  it('turns registry names into Dot labels', () => {
    expect(agentLabel('search_agent')).toBe('Search');
    expect(agentLabel('detailed_report_agent')).toBe('Detailed report');
    expect(agentLabel('coding')).toBe('Coding');
    expect(agentLabel('agent_router')).toBe('Agent router');
  });

  it('gives each agent a stable hue in range', () => {
    expect(agentHue('search_agent')).toBe(agentHue('search_agent'));
    expect(agentHue('search_agent')).not.toBe(agentHue('summarizer_agent'));
    for (const name of ['a', 'search_agent', 'x'.repeat(200)]) {
      const hue = agentHue(name);
      expect(Number.isInteger(hue) && hue >= 0 && hue < 360).toBe(true);
    }
  });

  it('formats segment times as m:ss', () => {
    expect(clock(0)).toBe('0:00');
    expect(clock(75.9)).toBe('1:15');
    expect(clock(3601)).toBe('60:01');
  });
});
