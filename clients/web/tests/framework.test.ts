import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import {
  cellText,
  compactDate,
  compactTimestamp,
  confidenceBand,
  entityText,
  exampleColumns,
  expectedReviewRate,
  fileSafe,
  formatRunAge,
  goldenSample,
  millis,
  pageNumbers,
  pageOf,
  ratingText,
} from '../src/client/ops/framework';
import { InlineReview, ReviewItem } from '../src/client/ops/framework/SyntheticDataTab';

describe('formatRunAge', () => {
  const now = new Date('2026-10-08T12:00:00Z');
  it('reads a run start as whole minutes, hours or days ago', () => {
    expect(formatRunAge('2026-10-08T11:59:01Z', now)).toBe('0m ago');
    expect(formatRunAge('2026-10-08T11:01:00Z', now)).toBe('59m ago');
    expect(formatRunAge('2026-10-08T11:00:00Z', now)).toBe('1h ago');
    expect(formatRunAge('2026-10-07T12:00:01Z', now)).toBe('23h ago');
    expect(formatRunAge('2026-10-05T12:00:00Z', now)).toBe('3d ago');
  });
  it('says when there is no start, when it does not parse, and never goes negative', () => {
    expect(formatRunAge(null, now)).toBe('not started');
    expect(formatRunAge('yesterday', now)).toBe('unknown');
    expect(formatRunAge('2026-10-08T12:05:00Z', now)).toBe('0m ago');
    expect(formatRunAge('2026-10-08T10:00:00', now)).toBe('2h ago');
  });
});

describe('confidenceBand', () => {
  it('bands below-threshold confidence at 0.75 and 0.6', () => {
    expect([0.84, 0.75, 0.74, 0.6, 0.59, 0].map((c) => confidenceBand(c))).toEqual([
      { label: 'Medium-low', tone: 'ok' },
      { label: 'Medium-low', tone: 'ok' },
      { label: 'Low', tone: 'warn' },
      { label: 'Low', tone: 'warn' },
      { label: 'Very low', tone: 'error' },
      { label: 'Very low', tone: 'error' },
    ]);
  });
});

describe('expectedReviewRate', () => {
  it('is the whole percent below the threshold, free of float noise', () => {
    expect([0.85, 0.9, 0.7, 0.855, 1, 0].map(expectedReviewRate)).toEqual([
      '~15%',
      '~10%',
      '~30%',
      '~14%',
      '~0%',
      '~100%',
    ]);
  });
});

describe('ratingText', () => {
  it('names each kind of rating', () => {
    expect([
      ratingText('thumbs', 1),
      ratingText('thumbs', 0),
      ratingText('stars', 1),
      ratingText('stars', 4),
      ratingText('relevance', 0.5),
    ]).toEqual(['Thumbs up', 'Thumbs down', '1 star', '4 stars', '0.5 relevance']);
  });
});

describe('annotation pages', () => {
  it('pages ten at a time and always has a first page', () => {
    expect(pageNumbers(0)).toEqual([1]);
    expect(pageNumbers(10)).toEqual([1]);
    expect(pageNumbers(11)).toEqual([1, 2]);
    const items = Array.from({ length: 23 }, (_, index) => index);
    expect(pageOf(items, 3)).toEqual([20, 21, 22]);
    expect(pageOf(items, 2)).toEqual([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
  });
});

describe('goldenSample', () => {
  it('keeps the first ten queries in order with their sizes', () => {
    const entry = (videos: string[], avg: number) => ({
      expected_videos: videos,
      relevance_scores: {},
      avg_relevance: avg,
      profile: 'p',
      timestamp: '',
    });
    const dataset = Object.fromEntries(
      Array.from({ length: 12 }, (_, index) => [`q${index}`, entry(['a', 'b'].slice(0, (index % 2) + 1), index / 10)]),
    );
    const sample = goldenSample(dataset);
    expect(sample.map((row) => row.query)).toEqual(['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9']);
    expect(sample.slice(0, 3)).toEqual([
      { query: 'q0', expectedVideos: 1, avgRelevance: 0 },
      { query: 'q1', expectedVideos: 2, avgRelevance: 0.1 },
      { query: 'q2', expectedVideos: 1, avgRelevance: 0.2 },
    ]);
  });
});

describe('export names', () => {
  it('stamps UTC dates and keeps tenant IDs file safe', () => {
    const date = new Date('2026-10-08T09:05:07Z');
    expect(compactDate(date)).toBe('20261008');
    expect(compactTimestamp(date)).toBe('20261008_090507');
    expect(fileSafe('acme:prod/eu')).toBe('acme_prod_eu');
  });
});

describe('generated example cells', () => {
  it('lists every column in first-seen order and renders values', () => {
    expect(exampleColumns([{ query: 'a', entities: [] }, { query: 'b', reasoning: 'r' }])).toEqual([
      'query',
      'entities',
      'reasoning',
    ]);
    expect([cellText('text'), cellText(['x']), cellText(null), cellText(0.5)]).toEqual(['text', '["x"]', '', '0.5']);
    expect([
      entityText({ text: 'Marie Curie', type: 'PERSON' }),
      entityText({ text: 'radium' }),
      entityText('plain'),
    ]).toEqual(['Marie Curie (PERSON)', 'radium', 'plain']);
    expect([millis(1250.4), millis(null)]).toEqual(['1250ms', '—']);
  });
});

describe('inline review of generated items', () => {
  const item = {
    item_id: 'synthetic_routing_ab_0',
    status: 'pending_review',
    confidence: 0.65,
    query: 'find the lecture on gradient descent',
    reasoning: 'Routed to video search for a lecture',
    entities: [{ text: 'gradient descent', type: 'CONCEPT' }, 'lecture'],
    schema_name: 'RoutingExperienceSchema',
    retry_count: 2,
    generation_metadata: { retry_count: 2, generator: 'RoutingGenerator' },
    data: {},
  };

  it('shows the query, reasoning, entities, confidence, retries, band and generation details', () => {
    const html = renderToStaticMarkup(createElement(ReviewItem, { item, position: 1, total: 3 }));
    expect(html).toContain(
      '<summary>Item 1/3 - Confidence: 0.65 - find the lecture on gradient descent</summary>',
    );
    expect(html).toContain('<dt>Generated query</dt><dd>find the lecture on gradient descent</dd>');
    expect(html).toContain('<dt>Reasoning</dt><dd>Routed to video search for a lecture</dd>');
    expect(html).toContain('<dt>Entities</dt><dd>gradient descent (CONCEPT), lecture</dd>');
    expect(html).toContain('<dt>Confidence</dt><dd>0.65</dd><dt>Retries</dt><dd>2</dd>');
    expect(html).toContain('<dt>Band</dt><dd class="band warn">Low</dd>');
    expect(html).toContain(
      '<summary>Generation details</summary><dl class="facts"><dt>retry_count</dt><dd>2</dd><dt>generator</dt><dd>RoutingGenerator</dd></dl>',
    );
  });

  it('shows five pending items and points to the Approvals view for the rest', () => {
    const pending = Array.from({ length: 7 }, (_, index) => ({
      ...item,
      item_id: `i${index}`,
      query: `q${index}`,
    }));
    const html = renderToStaticMarkup(createElement(InlineReview, { tenant: 'acme:prod', pending }));
    expect(html.match(/<details class="result-card"/g)).toHaveLength(5);
    expect(html).toContain('<strong>7 items</strong> need your review');
    expect(html).toContain(
      'Showing 5 of 7 items. Approve or reject them in the <a href="#/ops/approvals">Approvals</a> view under tenant acme:prod.',
    );
    expect(renderToStaticMarkup(createElement(InlineReview, { tenant: 'acme:prod', pending: [] }))).toBe(
      '<p role="status">All items were approved automatically; no review is needed.</p>',
    );
  });
});
