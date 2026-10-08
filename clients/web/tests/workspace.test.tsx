import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { noticeMessage, runNotice } from '../src/client/AgentWorkspace';
import { NoticeView } from '../src/client/Notice';
import { ResultPanel, codingOf, resultGroupsOf, resultsOf } from '../src/client/ResultCards';

const VIDEO_HIT = {
  id: 'v1_seg_3',
  document_id: 'id:video:video::v1_seg_3',
  score: 5.29,
  rrf_score: 0.0333,
  metadata: {
    video_id: 'v1',
    video_title: 'match.mp4',
    audio_transcript: '- Yeah.',
    segment_description: 'A man throws a ball on a grassy field while spectators watch.',
  },
  temporal_info: { start_time: 5.9, end_time: 6.9 },
};

const DOCUMENT_HIT = {
  document_id: 'doc_7',
  document_url: 's3://docs/report.pdf',
  title: 'Quarterly report',
  page_number: 3,
  document_type: 'pdf',
  content_preview: 'Revenue grew 12% on the back of video search.',
  relevance_score: 0.82,
  strategy_used: 'text',
  metadata: {},
};

const IMAGE_HIT = {
  image_id: 'img_2',
  image_url: 's3://images/cat.png',
  title: 'Cat on a mat',
  description: 'A tabby cat asleep on a red mat.',
  relevance_score: 0.71,
  detected_objects: [],
  detected_scenes: [],
  metadata: {},
};

const AUDIO_HIT = {
  audio_id: 'aud_4',
  audio_url: 's3://audio/talk.mp3',
  title: 'Keynote',
  transcript: 'Welcome everyone to the keynote.',
  duration: 61,
  relevance_score: 0.64,
  metadata: {},
};

const CODING_RESULT = {
  plan: 'Write reverse().',
  code_changes: [
    { file_path: '/workspace/solution.py', content: 'def reverse(s):\n    return s[::-1]\n', change_type: 'create' },
  ],
  execution_results: [
    { command: 'python /workspace/solution.py', exit_code: 0, stdout: 'olleh\n', stderr: '', success: true },
  ],
  summary: 'Completed coding task in 1 iteration(s). Generated 1 file(s). Final execution: exit_code=0',
  iterations_used: 1,
  files_modified: ['/workspace/solution.py'],
};

describe('resultsOf: each agent hit shape', () => {
  it('reads a video segment by its description, rated and ranked by rrf', () => {
    expect(resultsOf({ result: { results: [VIDEO_HIT] } })).toEqual([
      {
        id: 'v1_seg_3',
        ratingId: 'id:video:video::v1_seg_3',
        score: 0.0333,
        title: 'match.mp4',
        snippet: 'A man throws a ball on a grassy field while spectators watch.',
        start: 5.9,
        end: 6.9,
      },
    ]);
  });

  it('reads a document hit by its relevance score and content preview', () => {
    expect(resultsOf({ result: { results: [DOCUMENT_HIT] } })).toEqual([
      {
        id: 'doc_7',
        ratingId: 'doc_7',
        score: 0.82,
        title: 'Quarterly report',
        snippet: 'Revenue grew 12% on the back of video search.',
        start: undefined,
        end: undefined,
      },
    ]);
  });

  it('reads image and audio hits by their own ids and text', () => {
    expect(resultsOf({ result: { results: [IMAGE_HIT, AUDIO_HIT] } })).toEqual([
      {
        id: 'img_2',
        ratingId: 'img_2',
        score: 0.71,
        title: 'Cat on a mat',
        snippet: 'A tabby cat asleep on a red mat.',
        start: undefined,
        end: undefined,
      },
      {
        id: 'aud_4',
        ratingId: 'aud_4',
        score: 0.64,
        title: 'Keynote',
        snippet: 'Welcome everyone to the keynote.',
        start: undefined,
        end: undefined,
      },
    ]);
  });
});

describe('resultGroupsOf', () => {
  it('gives a single agent run one group with its span', () => {
    expect(
      resultGroupsOf({ agent: 'search_agent', result: { span_id: '00000000000000ab', results: [VIDEO_HIT] } }),
    ).toEqual([{ agent: 'search_agent', spanId: '00000000000000ab', items: resultsOf({ result: { results: [VIDEO_HIT] } }) }]);
  });

  it("reads an orchestration's hits per agent, in plan order, each with its own span", () => {
    const state = {
      agent: 'orchestrator_agent',
      result: {
        status: 'success',
        orchestration_result: {
          agent_results: {
            query_enhancement_agent: { status: 'success', enhanced_query: 'cats' },
            search_agent: { status: 'success', span_id: '00000000000000cd', results: [VIDEO_HIT] },
            document_agent: { status: 'success', results: [DOCUMENT_HIT] },
          },
        },
      },
    };
    expect(resultGroupsOf(state)).toEqual([
      { agent: 'search_agent', spanId: '00000000000000cd', items: resultsOf({ result: { results: [VIDEO_HIT] } }) },
      { agent: 'document_agent', spanId: undefined, items: resultsOf({ result: { results: [DOCUMENT_HIT] } }) },
    ]);
  });

  it('has no groups for a payload without hits', () => {
    expect(resultGroupsOf({ agent: 'summarizer_agent', result: { summary: 'hi' } })).toEqual([]);
    expect(resultGroupsOf(undefined)).toEqual([]);
  });
});

describe('codingOf', () => {
  const coding = {
    agent: 'coding_agent',
    summary: CODING_RESULT.summary,
    files: [{ path: '/workspace/solution.py', content: 'def reverse(s):\n    return s[::-1]\n', change: 'create' }],
    runs: [{ command: 'python /workspace/solution.py', exitCode: 0, stdout: 'olleh\n', stderr: '' }],
  };

  it('reads the dispatcher envelope, the streamed output and an orchestration step alike', () => {
    expect(
      codingOf({ agent: 'coding_agent', result: { status: 'success', agent: 'coding_agent', result: CODING_RESULT } }),
    ).toEqual([coding]);
    expect(codingOf({ agent: 'coding_agent', result: CODING_RESULT })).toEqual([coding]);
    expect(
      codingOf({
        agent: 'orchestrator_agent',
        result: {
          orchestration_result: {
            agent_results: { coding_agent: { status: 'success', agent: 'coding_agent', result: CODING_RESULT } },
          },
        },
      }),
    ).toEqual([coding]);
  });

  it('has nothing for a payload without code', () => {
    expect(codingOf({ agent: 'search_agent', result: { results: [] } })).toEqual([]);
  });
});

describe('ResultPanel', () => {
  it("renders the code and its run output", () => {
    const html = renderToStaticMarkup(
      <ResultPanel state={{ agent: 'coding_agent', result: { status: 'success', result: CODING_RESULT } }} />,
    );
    expect(html).toBe(
      '<aside class="results" aria-label="Results">' +
        '<section class="code-result" aria-label="Code from Coding">' +
        '<p class="code-summary">Completed coding task in 1 iteration(s). Generated 1 file(s). Final execution: exit_code=0</p>' +
        '<figure class="code-file"><figcaption>/workspace/solution.py (create)</figcaption>' +
        '<pre><code>def reverse(s):\n    return s[::-1]\n</code></pre></figure>' +
        '<figure class="code-run"><figcaption>python /workspace/solution.py — exit code 0</figcaption>' +
        '<pre aria-label="Output">olleh\n</pre></figure>' +
        '</section></aside>',
    );
  });

  it('labels each agent of an orchestration and rates only hits with a span', () => {
    const html = renderToStaticMarkup(
      <ResultPanel
        state={{
          agent: 'orchestrator_agent',
          result: {
            orchestration_result: {
              agent_results: {
                search_agent: { span_id: '00000000000000cd', results: [VIDEO_HIT] },
                document_agent: { results: [DOCUMENT_HIT] },
              },
            },
          },
        }}
      />,
    );
    expect(html.match(/<h2 class="result-group">[^<]*<\/h2>/g)).toEqual([
      '<h2 class="result-group">Search</h2>',
      '<h2 class="result-group">Document</h2>',
    ]);
    expect(html.match(/aria-label="Relevance of [^"]*"/g)).toEqual([
      'aria-label="Relevance of id:video:video::v1_seg_3"',
    ]);
    expect(html.match(/<p class="result-snippet">[^<]*<\/p>/g)).toEqual([
      '<p class="result-snippet">A man throws a ball on a grassy field while spectators watch.</p>',
      '<p class="result-snippet">Revenue grew 12% on the back of video search.</p>',
    ]);
    expect(html.match(/<span class="result-score">[^<]*<\/span>/g)).toEqual([
      '<span class="result-score">0.033</span>',
      '<span class="result-score">0.820</span>',
    ]);
  });

  it('renders nothing for a payload with neither hits nor code', () => {
    expect(renderToStaticMarkup(<ResultPanel state={{ result: { summary: 'x' } }} />)).toBe('');
  });
});

describe('run notices', () => {
  it("states a failed run's reason", () => {
    expect(runNotice({ message: 'profile_selection_agent failed with ValueError.', code: 'internal_error' }, false)).toEqual({
      tone: 'error',
      text: 'The run failed: profile_selection_agent failed with ValueError.',
    });
  });

  it('reads a stopped run as cancelled, whatever the abort reported', () => {
    for (const failure of [
      { message: 'This operation was aborted', code: 'abort' },
      { message: 'Run stopped by user', code: 'STOPPED' },
      undefined,
    ])
      expect(runNotice(failure, true)).toEqual({ tone: 'cancelled', text: 'You cancelled this run.' });
    expect(runNotice({ message: 'This operation was aborted', code: 'abort' }, false)).toEqual({
      tone: 'cancelled',
      text: 'The run was cancelled.',
    });
  });

  it('has nothing to say about a run that finished', () => {
    expect(runNotice(undefined, false)).toBeUndefined();
  });

  it('becomes an activity message the agent never reads', () => {
    expect(noticeMessage('n1', { tone: 'error', text: 'The run failed: x' })).toEqual({
      id: 'n1',
      role: 'activity',
      activityType: 'cogniverse.notice',
      content: { tone: 'error', text: 'The run failed: x' },
    });
  });

  it('renders an error as an alert and a cancellation as a status', () => {
    expect(renderToStaticMarkup(<NoticeView content={{ tone: 'error', text: 'The run failed: x' }} />)).toBe(
      '<p class="run-notice error" role="alert">The run failed: x</p>',
    );
    expect(renderToStaticMarkup(<NoticeView content={{ tone: 'cancelled', text: 'You cancelled this run.' }} />)).toBe(
      '<p class="run-notice cancelled" role="status">You cancelled this run.</p>',
    );
    expect(
      renderToStaticMarkup(<NoticeView content={{ tone: 'warning', text: 'Part of this conversation was not saved: x.' }} />),
    ).toBe('<p class="run-notice warning" role="status">Part of this conversation was not saved: x.</p>');
  });
});
