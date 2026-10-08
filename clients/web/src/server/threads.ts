import { EventType, type BaseEvent, type Message } from '@ag-ui/client';
import { InMemoryAgentRunner, type AgentRunnerConnectRequest } from '@copilotkit/runtime/v2';
import { Observable } from 'rxjs';
import type { ServerConfig } from './config.js';

/** The activity type of the notes the client shows in a conversation. */
export const NOTICE_ACTIVITY = 'cogniverse.notice';

interface Turn {
  role: 'user' | 'assistant';
  content: string;
}

interface ThreadRead {
  thread_id: string;
  state: 'loaded' | 'incomplete';
  reason: string | null;
  turns: Turn[];
}

/** A thread's saved turns as the chat's messages, ids stable per position. */
export function threadMessages(threadId: string, thread: ThreadRead): Message[] {
  const messages: Message[] = thread.turns.map((turn, index) => ({
    id: `${threadId}:${index}`,
    role: turn.role,
    content: turn.content,
  }));
  if (thread.state === 'incomplete')
    messages.push({
      id: `${threadId}:incomplete`,
      role: 'activity',
      activityType: NOTICE_ACTIVITY,
      content: {
        tone: 'warning',
        text: `Part of this conversation was not saved: ${thread.reason ?? 'no reason given'}.`,
      },
    });
  return messages;
}

function errorText(body: unknown, status: number): string {
  const error = (body as { error?: { message?: unknown } } | null)?.error;
  return typeof error?.message === 'string' ? error.message : `HTTP ${status}`;
}

/**
 * Runs agents in memory, and restores a thread that is not running from the
 * turns the Cogniverse runtime saved for it (``GET /ag-ui/threads/{id}``), so
 * a conversation survives reloads and server restarts. A thread still running
 * here joins its live run.
 */
export class CogniverseThreadRunner extends InMemoryAgentRunner {
  constructor(
    private readonly config: ServerConfig,
    private readonly fetchFn: typeof fetch = fetch,
  ) {
    super();
  }

  override connect(request: AgentRunnerConnectRequest): Observable<BaseEvent> {
    return new Observable<BaseEvent>((subscriber) => {
      let inner: { unsubscribe(): void } | undefined;
      const controller = new AbortController();
      const runId = `restore-${request.threadId}`;
      const emitAll = (events: BaseEvent[]) => {
        for (const event of events) subscriber.next(event);
        subscriber.complete();
      };
      const restore = async () => {
        if (await this.isRunning({ threadId: request.threadId })) {
          inner = super.connect(request).subscribe(subscriber);
          return;
        }
        const started = { type: EventType.RUN_STARTED, threadId: request.threadId, runId } as BaseEvent;
        let message: string;
        try {
          const response = await this.fetchFn(
            `${this.config.runtimeUrl}/ag-ui/threads/${encodeURIComponent(request.threadId)}`,
            {
              headers: { Authorization: `Bearer ${this.config.apiKey}` },
              signal: AbortSignal.any([controller.signal, AbortSignal.timeout(30000)]),
            },
          );
          const body = (await response.json().catch(() => null)) as unknown;
          if (response.ok) {
            emitAll([
              started,
              {
                type: EventType.MESSAGES_SNAPSHOT,
                messages: threadMessages(request.threadId, body as ThreadRead),
              } as BaseEvent,
              { type: EventType.RUN_FINISHED, threadId: request.threadId, runId } as BaseEvent,
            ]);
            return;
          }
          message = errorText(body, response.status);
        } catch (error) {
          if (controller.signal.aborted) return;
          message = `The Cogniverse runtime at ${this.config.runtimeUrl} did not answer (${
            error instanceof Error ? error.name : 'unknown error'
          }).`;
        }
        emitAll([
          started,
          {
            type: EventType.RUN_ERROR,
            message: `This conversation could not be restored: ${message}`,
            code: 'thread_unavailable',
          } as BaseEvent,
        ]);
      };
      restore().catch((error: unknown) => subscriber.error(error));
      return () => {
        controller.abort();
        inner?.unsubscribe();
      };
    });
  }
}
