import type { BaseEvent } from '@ag-ui/client';
import { firstValueFrom, lastValueFrom, toArray } from 'rxjs';
import { describe, expect, it } from 'vitest';
import type { ServerConfig } from '../src/server/config';
import { cogniverseAgents } from '../src/server/runtime';
import { TENANT_HEADER, TenantKeys } from '../src/server/tenants';
import { CogniverseThreadRunner, NOTICE_ACTIVITY } from '../src/server/threads';
import { deadUrl, json, runtimeServer, withAdmin, type Admin } from './fakeRuntime';

function config(runtimeUrl: string): ServerConfig {
  return { runtimeUrl, port: 4000, host: '127.0.0.1', clientDir: '/nonexistent' };
}

const TENANT = 'acme:prod';

/** A runner over a runtime answering thread reads with ``handler`` for
 * the registered ``acme:prod`` and ``beta:dev``. */
async function runnerFor(
  handler: Parameters<typeof withAdmin>[1],
): Promise<[CogniverseThreadRunner, Admin]> {
  const [route, admin] = withAdmin([TENANT, 'beta:dev'], handler);
  const url = await runtimeServer(route);
  return [new CogniverseThreadRunner(config(url), new TenantKeys(config(url))), admin];
}

const connect = (runner: CogniverseThreadRunner, threadId: string, tenant: string | null = TENANT): Promise<BaseEvent[]> =>
  lastValueFrom(
    runner.connect({ threadId, agentId: 'search_agent', headers: tenant ? { [TENANT_HEADER]: tenant } : {} }).pipe(toArray()),
  );

describe('CogniverseThreadRunner.connect', () => {
  it("restores a thread from the turns the runtime saved, with the tenant's key", async () => {
    const seen: string[] = [];
    const [runner, admin] = await runnerFor((req, res, admin) => {
      seen.push(`${req.method} ${req.url} ${admin.tenantOf(req.headers.authorization)}`);
      json(res, 200, {
        thread_id: 't/1',
        state: 'loaded',
        reason: null,
        turns: [
          { role: 'user', content: 'cats' },
          { role: 'assistant', content: 'two clips' },
          { role: 'user', content: 'dogs?' },
        ],
      });
    });
    expect(await connect(runner, 't/1')).toEqual([
      { type: 'RUN_STARTED', threadId: 't/1', runId: 'restore-t/1' },
      {
        type: 'MESSAGES_SNAPSHOT',
        messages: [
          { id: 't/1:0', role: 'user', content: 'cats' },
          { id: 't/1:1', role: 'assistant', content: 'two clips' },
          { id: 't/1:2', role: 'user', content: 'dogs?' },
        ],
      },
      { type: 'RUN_FINISHED', threadId: 't/1', runId: 'restore-t/1' },
    ]);
    expect(seen).toEqual([`GET /ag-ui/threads/t%2F1 ${TENANT}`]);
    expect(admin.calls).toEqual([`probe ${TENANT}`, `mint ${TENANT} ${new TenantKeys(config('')).keyName}`]);
  });

  it('says so when part of the thread was not saved', async () => {
    const [runner] = await runnerFor((_req, res) =>
      json(res, 200, {
        thread_id: 't2',
        state: 'incomplete',
        reason: 'a turn was not saved (RuntimeError)',
        turns: [{ role: 'user', content: 'lost reply' }],
      }),
    );
    const events = await connect(runner, 't2');
    expect(events[1]).toEqual({
      type: 'MESSAGES_SNAPSHOT',
      messages: [
        { id: 't2:0', role: 'user', content: 'lost reply' },
        {
          id: 't2:incomplete',
          role: 'activity',
          activityType: NOTICE_ACTIVITY,
          content: {
            tone: 'warning',
            text: 'Part of this conversation was not saved: a turn was not saved (RuntimeError).',
          },
        },
      ],
    });
  });

  it("fails the restore with the runtime's reason, never as an empty thread", async () => {
    const [runner] = await runnerFor((_req, res) =>
      json(res, 503, {
        error: {
          message: 'The conversation store is unavailable (HTTPError). See server logs for detail.',
          type: 'server_error',
          code: 'service_unavailable',
        },
      }),
    );
    expect(await connect(runner, 't3')).toEqual([
      { type: 'RUN_STARTED', threadId: 't3', runId: 'restore-t3' },
      {
        type: 'RUN_ERROR',
        message:
          'This conversation could not be restored: The conversation store is unavailable (HTTPError). ' +
          'See server logs for detail.',
        code: 'thread_unavailable',
      },
    ]);
  });

  it('names the runtime that did not answer', async () => {
    const url = await deadUrl();
    expect(await connect(new CogniverseThreadRunner(config(url), new TenantKeys(config(url))), 't4')).toEqual([
      { type: 'RUN_STARTED', threadId: 't4', runId: 'restore-t4' },
      {
        type: 'RUN_ERROR',
        message: `This conversation could not be restored: The Cogniverse runtime at ${url} did not answer (TypeError).`,
        code: 'thread_unavailable',
      },
    ]);
  });

  it('restores concurrent threads each from their own turns', async () => {
    const [runner] = await runnerFor((req, res) => {
      const thread = decodeURIComponent(req.url!.split('/').pop()!);
      // Answer in reverse arrival order so no restore sees another's reply.
      setTimeout(
        () => json(res, 200, { thread_id: thread, state: 'loaded', reason: null, turns: [{ role: 'user', content: thread }] }),
        (10 - Number(thread.slice(1))) * 15,
      );
    });
    const threads = Array.from({ length: 8 }, (_, i) => `c${i}`);
    const restored = await Promise.all(threads.map((thread) => connect(runner, thread)));
    expect(restored.map((events) => (events[1] as unknown as { messages: unknown[] }).messages)).toEqual(
      threads.map((thread) => [{ id: `${thread}:0`, role: 'user', content: thread }]),
    );
  });

  it("reads each tenant's own thread store for the same thread id", async () => {
    const [runner] = await runnerFor((req, res, admin) =>
      json(res, 200, {
        thread_id: 'shared',
        state: 'loaded',
        reason: null,
        turns: [{ role: 'user', content: `asked as ${admin.tenantOf(req.headers.authorization)}` }],
      }),
    );
    const [acme, beta] = await Promise.all([connect(runner, 'shared', TENANT), connect(runner, 'shared', 'beta:dev')]);
    expect([acme[1], beta[1]]).toEqual([
      { type: 'MESSAGES_SNAPSHOT', messages: [{ id: 'shared:0', role: 'user', content: `asked as ${TENANT}` }] },
      { type: 'MESSAGES_SNAPSHOT', messages: [{ id: 'shared:0', role: 'user', content: 'asked as beta:dev' }] },
    ]);
  });

  it('refuses a restore that names no tenant without calling the runtime', async () => {
    const [runner, admin] = await runnerFor((_req, res) => json(res, 500, {}));
    expect(await connect(runner, 't5', null)).toEqual([
      { type: 'RUN_STARTED', threadId: 't5', runId: 'restore-t5' },
      { type: 'RUN_ERROR', message: 'Choose a tenant before opening a conversation.', code: 'no_tenant' },
    ]);
    expect(admin.calls).toEqual([]);
  });
});

describe('CogniverseThreadRunner.run', () => {
  it("keeps a running thread to the tenant that started it", async () => {
    let release: () => void = () => {};
    const [route, admin] = withAdmin([TENANT, 'beta:dev'], (req, res) => {
      res.setHeader('content-type', 'text/event-stream');
      res.write(`data: ${JSON.stringify({ type: 'RUN_STARTED', threadId: 'live', runId: 'r1' })}\n\n`);
      release = () =>
        res.end(
          `data: ${JSON.stringify({ type: 'TEXT_MESSAGE_START', messageId: 'm', role: 'assistant' })}\n\n` +
            `data: ${JSON.stringify({ type: 'TEXT_MESSAGE_CONTENT', messageId: 'm', delta: `for ${admin.tenantOf(req.headers.authorization)}` })}\n\n` +
            `data: ${JSON.stringify({ type: 'TEXT_MESSAGE_END', messageId: 'm' })}\n\n` +
            `data: ${JSON.stringify({ type: 'RUN_FINISHED', threadId: 'live', runId: 'r1' })}\n\n`,
        );
    });
    const url = await runtimeServer(route);
    const keys = new TenantKeys(config(url));
    const runner = new CogniverseThreadRunner(config(url), keys);
    const agentOf = (tenant: string) => cogniverseAgents(config(url), ['search_agent'], keys, tenant).search_agent;
    const input = (runId: string) => ({ threadId: 'live', runId, messages: [], tools: [], context: [], state: {}, forwardedProps: {} });

    const running = lastValueFrom(
      runner.run({ threadId: 'live', agent: agentOf(TENANT), input: input('r1') }).pipe(toArray()),
    );
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(await runner.isRunning({ threadId: 'live' })).toBe(true);

    expect(await connect(runner, 'live', 'beta:dev')).toEqual([
      { type: 'RUN_STARTED', threadId: 'live', runId: 'restore-live' },
      { type: 'RUN_ERROR', message: 'This conversation belongs to another tenant.', code: 'thread_of_another_tenant' },
    ]);
    expect(
      await lastValueFrom(runner.run({ threadId: 'live', agent: agentOf('beta:dev'), input: input('r2') }).pipe(toArray())),
    ).toEqual([
      { type: 'RUN_STARTED', threadId: 'live', runId: 'r2' },
      { type: 'RUN_ERROR', message: 'This conversation belongs to another tenant.', code: 'thread_of_another_tenant' },
    ]);
    const joined = firstValueFrom(runner.connect({ threadId: 'live', headers: { [TENANT_HEADER]: TENANT } }));
    expect(await joined).toMatchObject({ type: 'RUN_STARTED', threadId: 'live' });

    release();
    const events = await running;
    expect(events.filter((event) => event.type === 'TEXT_MESSAGE_CONTENT')).toEqual([
      { type: 'TEXT_MESSAGE_CONTENT', messageId: 'm', delta: `for ${TENANT}` },
    ]);
    // Once the run is over the thread is no longer held for its tenant.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(await runner.isRunning({ threadId: 'live' })).toBe(false);
  });
});
