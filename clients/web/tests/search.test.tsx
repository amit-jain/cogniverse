import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_AGENT, chosenAgent } from '../src/client/App';
import { turnRecord } from '../src/client/AgentWorkspace';
import {
  ResultPanel,
  entitiesOf,
  enhancementsOf,
  foundLine,
  keyPointsOf,
  orchestrationSummaryOf,
  profileSelectionsOf,
} from '../src/client/ResultCards';
import { summarize } from '../src/client/SearchSession';
import {
  annotationExport,
  loadAnnotations,
  loadSettings,
  loadTurns,
  messageText,
  saveAnnotations,
  saveTurns,
  spanIdsOf,
  withAnnotation,
} from '../src/client/session';

const HIT_A = {
  id: 'v1_seg_0',
  document_id: 'id:video:video::v1_seg_0',
  score: 0.91,
  metadata: { video_id: 'v1', video_title: 'Tower at night' },
};
const HIT_B = {
  id: 'v2_seg_4',
  document_id: 'id:video:video::v2_seg_4',
  score: 0.31,
  metadata: { video_id: 'v2', video_title: 'Tower by day' },
};

const SEARCH = {
  agent: 'search_agent',
  tenant_id: 'acme:prod',
  result: {
    span_id: '00000000000000ab',
    results: [HIT_A, HIT_B],
    profile: null,
    profiles: ['video_colpali_smol500_mv_frame', 'video_videoprism_base_mv_chunk_30s'],
    search_mode: 'ensemble',
    degraded_profiles: [{ profile: 'video_videoprism_base_mv_chunk_30s', reason: 'encoder unavailable' }],
  },
};

describe('foundLine', () => {
  it('counts what a search found for its question', () => {
    expect(foundLine(2, 'tower at night')).toBe("Found 2 results for 'tower at night'.");
    expect(foundLine(1, 'tower')).toBe("Found 1 result for 'tower'.");
    expect(foundLine(0, 'zebra')).toBe("No results for 'zebra'.");
    expect(foundLine(0)).toBe('No results.');
  });
});

describe('ResultPanel: a search', () => {
  it('states what it found, its metrics, its partial ensemble and the hits above the minimum score', () => {
    const html = renderToStaticMarkup(
      <ResultPanel state={SEARCH} tenant="acme:prod" run={{ query: 'tower at night', latencyMs: 1234.4, minScore: 0.5 }} />,
    );
    expect(html).toBe(
      '<aside class="results" aria-label="Results">' +
        '<section class="search-group" aria-label="Search by Search">' +
        '<p class="result-found">Found 2 results for &#x27;tower at night&#x27;.</p>' +
        '<dl class="result-metrics">' +
        '<div><dt>Results</dt><dd>2</dd></div>' +
        '<div><dt>Latency</dt><dd>1234 ms</dd></div>' +
        '<div><dt>Profile</dt><dd>video_colpali_smol500_mv_frame, video_videoprism_base_mv_chunk_30s</dd></div>' +
        '<div><dt>Search mode</dt><dd>ensemble</dd></div>' +
        '</dl>' +
        '<p class="alert warning" role="alert">Partial results: video_videoprism_base_mv_chunk_30s did not run (encoder unavailable).</p>' +
        '<p class="muted">Showing 1 of 2; the rest score below 0.5.</p>' +
        '<ol class="result-list"><li class="result-card">' +
        '<div class="result-head"><span class="result-title">Tower at night</span><span class="result-score">0.910</span></div>' +
        '<div class="result-id">v1_seg_0</div>' +
        '<div class="result-id">Video v1 · Document id:video:video::v1_seg_0</div>' +
        '<div class="relevance" role="group" aria-label="Relevance of id:video:video::v1_seg_0">' +
        '<button type="button" aria-pressed="false">Highly Relevant</button>' +
        '<button type="button" aria-pressed="false">Somewhat Relevant</button>' +
        '<button type="button" aria-pressed="false">Not Relevant</button>' +
        '</div></li></ol>' +
        '</section></aside>',
    );
  });

  it('says a search that found nothing found nothing', () => {
    const html = renderToStaticMarkup(
      <ResultPanel
        state={{
          agent: 'search_agent',
          tenant_id: 'acme:prod',
          result: { results: [], profile: 'video_colpali_smol500_mv_frame', search_mode: 'single_profile' },
        }}
        tenant="acme:prod"
        run={{ query: 'zebra' }}
      />,
    );
    expect(html).toBe(
      '<aside class="results" aria-label="Results">' +
        '<section class="search-group" aria-label="Search by Search">' +
        '<p class="result-found">No results for &#x27;zebra&#x27;.</p>' +
        '<dl class="result-metrics"><div><dt>Results</dt><dd>0</dd></div>' +
        '<div><dt>Profile</dt><dd>video_colpali_smol500_mv_frame</dd></div>' +
        '<div><dt>Search mode</dt><dd>single_profile</dd></div></dl>' +
        '<ol class="result-list"></ol></section></aside>',
    );
  });

  it("shows an answer agent's key points and an orchestration's account", () => {
    const state = {
      agent: 'orchestrator_agent',
      tenant_id: 'acme:prod',
      result: {
        key_points: ['Three clips show the tower', '', 7, 'The clearest is at 0:42'],
        orchestration_result: { execution_summary: 'Ran search_agent then summarizer_agent.' },
      },
    };
    expect(keyPointsOf(state)).toEqual(['Three clips show the tower', 'The clearest is at 0:42']);
    expect(orchestrationSummaryOf(state)).toBe('Ran search_agent then summarizer_agent.');
    expect(renderToStaticMarkup(<ResultPanel state={state} tenant="acme:prod" />)).toBe(
      '<aside class="results" aria-label="Results">' +
        '<section class="orchestration-summary" aria-label="Orchestration summary">' +
        '<h2 class="results-heading">Orchestration</h2><p>Ran search_agent then summarizer_agent.</p></section>' +
        '<section class="key-points" aria-label="Key points"><h2 class="results-heading">Key points</h2>' +
        '<ul><li>Three clips show the tower</li><li>The clearest is at 0:42</li></ul></section>' +
        '</aside>',
    );
  });
});

const ENTITY_STATE = {
  agent: 'entity_extraction_agent',
  tenant_id: 'acme:prod',
  result: {
    status: 'success',
    agent: 'entity_extraction_agent',
    query: "Daenerys burned the castle at King's Landing",
    entities: [
      { text: 'Daenerys', type: 'PERSON', context: 'Daenerys burned' },
      { text: "King's Landing", type: 'PLACE' },
      { text: '', type: 'CONCEPT' },
    ],
    relationships: [
      { subject: 'Daenerys', relation: 'burn', object: 'castle', confidence: 0.8 },
      { subject: 'Daenerys', relation: '', object: 'castle' },
    ],
    entity_count: 2,
    has_entities: true,
    answer: "Found 2 entities: Daenerys (person), King's Landing (place).",
  },
};

const ENHANCEMENT_STATE = {
  agent: 'query_enhancement_agent',
  tenant_id: 'acme:prod',
  result: {
    status: 'success',
    agent: 'query_enhancement_agent',
    original_query: 'fire castle video',
    enhanced_query: 'burning castle video footage',
    expansion_terms: ['footage', 'blaze'],
    synonyms: ['clip'],
    context_additions: [],
    query_variants: ['burning castle video footage', 'fire castle video footage blaze'],
    confidence: 0.8,
    reasoning: 'Spelled out the scene.',
    path_used: 'lm',
  },
};

const PROFILE_STATE = {
  agent: 'profile_selection_agent',
  tenant_id: 'acme:prod',
  result: {
    status: 'success',
    agent: 'profile_selection_agent',
    query: 'Which profile for a burning castle video?',
    selected_profile: 'video_colpali_smol500_mv_frame',
    confidence: 0.95,
    reasoning: 'Frame-level video search.',
    query_intent: 'video_search',
    modality: 'video',
    complexity: 'simple',
    alternatives: [{ profile_name: 'video_videoprism_base_mv_chunk_30s', score: 0.4, reasoning: 'Chunk level.' }],
  },
};

describe('enrichment panels', () => {
  it("shows an entity extraction's entities and relationships", () => {
    expect(entitiesOf(ENTITY_STATE)).toEqual([
      {
        agent: 'entity_extraction_agent',
        entities: [
          { text: 'Daenerys', type: 'PERSON' },
          { text: "King's Landing", type: 'PLACE' },
        ],
        relationships: [{ subject: 'Daenerys', relation: 'burn', object: 'castle' }],
      },
    ]);
    expect(renderToStaticMarkup(<ResultPanel state={ENTITY_STATE} tenant="acme:prod" />)).toBe(
      '<aside class="results" aria-label="Results">' +
        '<section class="entities" aria-label="Entities"><h2 class="results-heading">Entities</h2>' +
        '<ul><li>Daenerys<span class="entity-type"> person</span></li>' +
        '<li>King&#x27;s Landing<span class="entity-type"> place</span></li></ul>' +
        '<h3 class="results-subheading">Relationships</h3><ul><li>Daenerys → burn → castle</li></ul>' +
        '</section></aside>',
    );
  });

  it('shows the query as asked, the query searched and how it was enhanced', () => {
    expect(renderToStaticMarkup(<ResultPanel state={ENHANCEMENT_STATE} tenant="acme:prod" />)).toBe(
      '<aside class="results" aria-label="Results">' +
        '<section class="query-enhancement" aria-label="Query enhancement">' +
        '<h2 class="results-heading">Query enhancement</h2><dl class="enrichment-facts">' +
        '<div><dt>Asked</dt><dd>fire castle video</dd></div>' +
        '<div><dt>Searched</dt><dd>burning castle video footage</dd></div>' +
        '<div><dt>Expansion terms</dt><dd>footage, blaze</dd></div>' +
        '<div><dt>Synonyms</dt><dd>clip</dd></div>' +
        '<div><dt>Variants</dt><dd>burning castle video footage · fire castle video footage blaze</dd></div>' +
        '<div><dt>Path</dt><dd>Language model</dd></div></dl>' +
        '<p class="enrichment-reasoning">Spelled out the scene.</p></section></aside>',
    );
  });

  it('says a query searched as asked is unchanged', () => {
    const state = {
      ...ENHANCEMENT_STATE,
      result: {
        ...ENHANCEMENT_STATE.result,
        enhanced_query: 'fire castle video',
        expansion_terms: [],
        synonyms: [],
        query_variants: [],
        reasoning: 'Fallback enhancement with heuristic expansion',
        path_used: 'heuristic_fallback',
      },
    };
    expect(enhancementsOf(state)).toEqual([
      {
        agent: 'query_enhancement_agent',
        original: 'fire castle video',
        enhanced: 'fire castle video',
        expansionTerms: [],
        synonyms: [],
        variants: [],
        path: 'heuristic_fallback',
        reasoning: 'Fallback enhancement with heuristic expansion',
      },
    ]);
    expect(renderToStaticMarkup(<ResultPanel state={state} tenant="acme:prod" />)).toContain(
      '<div><dt>Searched</dt><dd>Unchanged</dd></div><div><dt>Path</dt><dd>heuristic fallback</dd></div></dl>',
    );
  });

  it('shows the profile chosen, why, and the runners-up', () => {
    expect(profileSelectionsOf(PROFILE_STATE)).toEqual([
      {
        agent: 'profile_selection_agent',
        profile: 'video_colpali_smol500_mv_frame',
        confidence: 0.95,
        intent: 'video_search',
        modality: 'video',
        complexity: 'simple',
        reasoning: 'Frame-level video search.',
        alternatives: [{ profile: 'video_videoprism_base_mv_chunk_30s', score: 0.4, reasoning: 'Chunk level.' }],
      },
    ]);
    expect(renderToStaticMarkup(<ResultPanel state={PROFILE_STATE} tenant="acme:prod" />)).toBe(
      '<aside class="results" aria-label="Results">' +
        '<section class="profile-selection" aria-label="Profile selection">' +
        '<h2 class="results-heading">Profile selection</h2><dl class="enrichment-facts">' +
        '<div><dt>Profile</dt><dd>video_colpali_smol500_mv_frame</dd></div>' +
        '<div><dt>Confidence</dt><dd>0.95</dd></div>' +
        '<div><dt>Intent</dt><dd>video search</dd></div>' +
        '<div><dt>Modality</dt><dd>video</dd></div>' +
        '<div><dt>Complexity</dt><dd>simple</dd></div></dl>' +
        '<p class="enrichment-reasoning">Frame-level video search.</p>' +
        '<h3 class="results-subheading">Alternatives</h3>' +
        '<ul><li>video_videoprism_base_mv_chunk_30s (0.40): Chunk level.</li></ul></section></aside>',
    );
  });

  it("hides another tenant's enrichment results", () => {
    for (const state of [ENTITY_STATE, ENHANCEMENT_STATE, PROFILE_STATE])
      expect(renderToStaticMarkup(<ResultPanel state={state} tenant="beta:dev" />)).toBe(
        '<aside class="results" aria-label="Results"><p class="alert error" role="alert">' +
          'These results belong to tenant acme:prod, not beta:dev; they are not shown.</p></aside>',
      );
  });
});

describe('turnRecord', () => {
  it("records a run's question, its hits and the spans its searches recorded them under", () => {
    expect(turnRecord('tower at night', SEARCH, '2026-10-08T10:00:00.000Z')).toEqual({
      query: 'tower at night',
      at: '2026-10-08T10:00:00.000Z',
      results: 2,
      spanIds: ['00000000000000ab'],
      profile: 'video_colpali_smol500_mv_frame, video_videoprism_base_mv_chunk_30s',
    });
  });

  it('records a run that did not search, or that failed, without a result count', () => {
    expect(turnRecord('hello', { agent: 'chat', result: { answer: 'hi' } }, 't')).toEqual({
      query: 'hello',
      at: 't',
      results: undefined,
      spanIds: [],
      profile: undefined,
    });
    expect(turnRecord('hello', undefined, 't').results).toBe(undefined);
  });
});

describe('session records', () => {
  const rating = { spanId: '00000000000000ab', resultId: 'doc_1', relevance: 'Not Relevant', score: 0, query: 'q', at: 't1' };

  it('keeps one rating per hit of a search, the latest', () => {
    const first = withAnnotation([], rating);
    const second = withAnnotation(first, { ...rating, resultId: 'doc_2', relevance: 'Highly Relevant', score: 1 });
    const again = withAnnotation(second, { ...rating, relevance: 'Somewhat Relevant', score: 0.5, at: 't2' });
    expect(again).toEqual([
      { ...rating, resultId: 'doc_2', relevance: 'Highly Relevant', score: 1 },
      { ...rating, relevance: 'Somewhat Relevant', score: 0.5, at: 't2' },
    ]);
  });

  it('exports the ratings with the tenant and conversation they were made in', () => {
    const turns = [
      { query: 'tower', at: 't0', results: 2, spanIds: ['00000000000000ab'], profile: 'p1' },
      { query: 'tower at night', at: 't1', results: 1, spanIds: ['00000000000000cd', '00000000000000ab'], profile: 'p2' },
    ];
    expect(spanIdsOf(turns)).toEqual(['00000000000000ab', '00000000000000cd']);
    expect(annotationExport('acme:prod', 'thread-1', 'search_agent', turns, [rating], '2026-10-08T10:00:00.000Z')).toEqual({
      search_session: {
        tenant_id: 'acme:prod',
        thread_id: 'thread-1',
        agent: 'search_agent',
        query: 'tower at night',
        profile: 'p2',
        exported_at: '2026-10-08T10:00:00.000Z',
      },
      annotations: [
        { query: 'q', span_id: '00000000000000ab', result_id: 'doc_1', relevance: 'Not Relevant', score: 0, rated_at: 't1' },
      ],
    });
  });

  it("keeps each tenant's records of a thread apart", () => {
    const stored = new Map<string, string>();
    vi.stubGlobal('localStorage', {
      getItem: (key: string) => stored.get(key) ?? null,
      setItem: (key: string, value: string) => stored.set(key, value),
    });
    try {
      const turn = { query: 'tower', at: 't0', results: 1, spanIds: ['00000000000000ab'] };
      saveTurns('acme:prod', 'thread-1', [turn]);
      saveAnnotations('acme:prod', 'thread-1', [rating]);
      expect(loadTurns('acme:prod', 'thread-1')).toEqual([turn]);
      expect(loadAnnotations('acme:prod', 'thread-1')).toEqual([rating]);
      expect(loadTurns('beta:dev', 'thread-1')).toEqual([]);
      expect(loadAnnotations('beta:dev', 'thread-1')).toEqual([]);
      expect([...stored.keys()].sort()).toEqual([
        'cogniverse.annotations.acme:prod/thread-1',
        'cogniverse.turns.acme:prod/thread-1',
      ]);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("reads a message's text from a string or its text parts", () => {
    expect(messageText('cats')).toBe('cats');
    expect(messageText([{ type: 'text', text: 'cats' }, { type: 'image' }, { type: 'text', text: 'dogs' }])).toBe(
      'cats dogs',
    );
    expect(messageText(undefined)).toBe('');
  });
});

describe('loadSettings', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('keeps only valid stored settings', () => {
    const stored = new Map<string, string>();
    vi.stubGlobal('localStorage', { getItem: (key: string) => stored.get(key) ?? null });
    expect(loadSettings()).toEqual({ topK: 10, minScore: 0 });
    stored.set('cogniverse.search-settings', JSON.stringify({ topK: 4, minScore: 0.25 }));
    expect(loadSettings()).toEqual({ topK: 4, minScore: 0.25 });
    stored.set('cogniverse.search-settings', JSON.stringify({ topK: 21, minScore: -1 }));
    expect(loadSettings()).toEqual({ topK: 10, minScore: 0 });
    stored.set('cogniverse.search-settings', '{not json');
    expect(loadSettings()).toEqual({ topK: 10, minScore: 0 });
  });
});

describe('chosenAgent', () => {
  it('opens the named agent, else the gateway, else the first agent', () => {
    expect(DEFAULT_AGENT).toBe('gateway_agent');
    expect(chosenAgent('search_agent', ['search_agent', 'gateway_agent'])).toBe('search_agent');
    expect(chosenAgent(undefined, ['search_agent', 'gateway_agent'])).toBe('gateway_agent');
    expect(chosenAgent('gone_agent', ['search_agent', 'gateway_agent'])).toBe('gateway_agent');
    expect(chosenAgent(undefined, ['search_agent', 'coding_agent'])).toBe('search_agent');
    expect(chosenAgent(undefined, [])).toBe(undefined);
  });
});

function sse(events: object[]): Response {
  const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('');
  return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } });
}

describe('summarize', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('sends the hits as the run grounding and reads the reply, status and key points', async () => {
    const fetchFn = vi.fn(async () =>
      sse([
        { type: 'RUN_STARTED', threadId: 't', runId: 'r' },
        { type: 'CUSTOM', name: 'cogniverse.status', value: { phase: 'summarization', message: 'Generating summary...' } },
        { type: 'TEXT_MESSAGE_START', messageId: 'm', role: 'assistant' },
        { type: 'TEXT_MESSAGE_CONTENT', messageId: 'm', delta: 'Two clips ' },
        { type: 'TEXT_MESSAGE_CONTENT', messageId: 'm', delta: 'show the tower.' },
        { type: 'TEXT_MESSAGE_END', messageId: 'm' },
        { type: 'STATE_SNAPSHOT', snapshot: { agent: 'summarizer_agent', result: { key_points: ['Night', 'Day'] } } },
        { type: 'RUN_FINISHED', threadId: 't', runId: 'r' },
      ]),
    );
    vi.stubGlobal('fetch', fetchFn);
    const updates: object[] = [];
    await summarize('acme:prod', 'tower', [HIT_A, HIT_B], (update) => updates.push(update));

    expect(updates).toEqual([
      { status: 'Generating summary...' },
      { text: 'Two clips ' },
      { text: 'Two clips show the tower.' },
      { keyPoints: ['Night', 'Day'] },
    ]);
    const [url, init] = fetchFn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/ui-api/runtime/ag-ui/summarizer_agent');
    expect(init.headers).toEqual({
      'content-type': 'application/json',
      accept: 'text/event-stream',
      'x-cogniverse-tenant': 'acme:prod',
    });
    const body = JSON.parse(init.body as string);
    expect(body.messages.map((m: { role: string; content: string }) => [m.role, m.content])).toEqual([
      ['user', "Summarize the search results for 'tower'"],
    ]);
    expect(body.forwardedProps).toEqual({ cogniverse: { search_results: [HIT_A, HIT_B] } });
  });

  it("fails with the run's error, and with a refused request's reason", async () => {
    vi.stubGlobal('fetch', async () =>
      sse([{ type: 'RUN_ERROR', message: "Agent 'summarizer_agent' failed with ValueError.", code: 'internal_error' }]),
    );
    await expect(summarize('acme:prod', 'tower', [HIT_A], () => undefined)).rejects.toThrow(
      "Agent 'summarizer_agent' failed with ValueError.",
    );
    vi.stubGlobal(
      'fetch',
      async () =>
        new Response(JSON.stringify({ error: { message: "Agent 'summarizer_agent' is not registered." } }), {
          status: 404,
        }),
    );
    await expect(summarize('acme:prod', 'tower', [HIT_A], () => undefined)).rejects.toThrow(
      "Agent 'summarizer_agent' is not registered.",
    );
  });
});
