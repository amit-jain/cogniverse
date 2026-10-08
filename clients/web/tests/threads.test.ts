import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import type { BaseEvent } from '@ag-ui/client';
import { lastValueFrom, toArray } from 'rxjs';
import { afterEach, describe, expect, it } from 'vitest';
import type { ServerConfig } from '../src/server/config';
import { CogniverseThreadRunner, NOTICE_ACTIVITY } from '../src/server/threads';

const servers: Server[] = [];
afterEach(async () => {
  await Promise.all(servers.splice(0).map((s) => new Promise((r) => s.close(r))));
});

async function runtimeServer(handler: (req: IncomingMessage, res: ServerResponse) => void): Promise<string> {
  const server = createServer(handler);
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

async function deadUrl(): Promise<string> {
  const server = createServer();
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  await new Promise((r) => server.close(r));
  return `http://127.0.0.1:${port}`;
}

function config(runtimeUrl: string): ServerConfig {
  return { runtimeUrl, apiKey: 'sk-test', port: 4000, host: '127.0.0.1', clientDir: '/nonexistent' };
}

function json(res: ServerResponse, status: number, body: unknown) {
  res.statusCode = status;
  res.setHeader('content-type', 'application/json');
  res.end(JSON.stringify(body));
}

const connect = (runner: CogniverseThreadRunner, threadId: string): Promise<BaseEvent[]> =>
  lastValueFrom(runner.connect({ threadId, agentId: 'search_agent' }).pipe(toArray()));

describe('CogniverseThreadRunner.connect', () => {
  it("restores a thread from the turns the runtime saved, with the server's key", async () => {
    const seen: string[] = [];
    const url = await runtimeServer((req, res) => {
      seen.push(`${req.method} ${req.url} ${req.headers.authorization}`);
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
    expect(await connect(new CogniverseThreadRunner(config(url)), 't/1')).toEqual([
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
    expect(seen).toEqual(['GET /ag-ui/threads/t%2F1 Bearer sk-test']);
  });

  it('says so when part of the thread was not saved', async () => {
    const url = await runtimeServer((_req, res) =>
      json(res, 200, {
        thread_id: 't2',
        state: 'incomplete',
        reason: 'a turn was not saved (RuntimeError)',
        turns: [{ role: 'user', content: 'lost reply' }],
      }),
    );
    const events = await connect(new CogniverseThreadRunner(config(url)), 't2');
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
    const url = await runtimeServer((_req, res) =>
      json(res, 503, {
        error: {
          message: 'The conversation store is unavailable (HTTPError). See server logs for detail.',
          type: 'server_error',
          code: 'service_unavailable',
        },
      }),
    );
    expect(await connect(new CogniverseThreadRunner(config(url)), 't3')).toEqual([
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
    expect(await connect(new CogniverseThreadRunner(config(url)), 't4')).toEqual([
      { type: 'RUN_STARTED', threadId: 't4', runId: 'restore-t4' },
      {
        type: 'RUN_ERROR',
        message: `This conversation could not be restored: The Cogniverse runtime at ${url} did not answer (TypeError).`,
        code: 'thread_unavailable',
      },
    ]);
  });

  it('restores concurrent threads each from their own turns', async () => {
    const url = await runtimeServer((req, res) => {
      const thread = decodeURIComponent(req.url!.split('/').pop()!);
      // Answer in reverse arrival order so no restore sees another's reply.
      setTimeout(
        () => json(res, 200, { thread_id: thread, state: 'loaded', reason: null, turns: [{ role: 'user', content: thread }] }),
        (10 - Number(thread.slice(1))) * 15,
      );
    });
    const runner = new CogniverseThreadRunner(config(url));
    const threads = Array.from({ length: 8 }, (_, i) => `c${i}`);
    const restored = await Promise.all(threads.map((thread) => connect(runner, thread)));
    expect(restored.map((events) => (events[1] as unknown as { messages: unknown[] }).messages)).toEqual(
      threads.map((thread) => [{ id: `${thread}:0`, role: 'user', content: thread }]),
    );
  });
});
