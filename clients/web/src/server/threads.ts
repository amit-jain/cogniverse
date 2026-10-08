import { EventType, type BaseEvent, type Message } from '@ag-ui/client';
import {
  InMemoryAgentRunner,
  type AgentRunnerConnectRequest,
  type AgentRunnerRunRequest,
} from '@copilotkit/runtime/v2';
import { Observable, finalize } from 'rxjs';
import type { ServerConfig } from './config.js';
import { requestTenant, tenantFetch, type TenantKeys } from './tenants.js';

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
  const error = (body as { error?: unknown } | null)?.error;
  if (typeof error === 'string') return error;
  const message = (error as { message?: unknown } | null | undefined)?.message;
  return typeof message === 'string' ? message : `HTTP ${status}`;
}

/** The run's events as one ``RUN_ERROR``, after its ``RUN_STARTED``. */
function refused(threadId: string, runId: string, message: string, code: string): Observable<BaseEvent> {
  return new Observable<BaseEvent>((subscriber) => {
    subscriber.next({ type: EventType.RUN_STARTED, threadId, runId } as BaseEvent);
    subscriber.next({ type: EventType.RUN_ERROR, message, code } as BaseEvent);
    subscriber.complete();
  });
}

/**
 * Runs agents in memory, and restores a thread that is not running from the
 * turns the Cogniverse runtime saved for it (``GET /ag-ui/threads/{id}``) with
 * the requesting tenant's key, so a conversation survives reloads and server
 * restarts. A thread still running here joins its live run, but only for the
 * tenant that started it.
 */
export class CogniverseThreadRunner extends InMemoryAgentRunner {
  /** The tenant of each thread running here. */
  private readonly owners = new Map<string, string>();

  constructor(
    private readonly config: ServerConfig,
    private readonly keys: TenantKeys,
    private readonly fetchFn: typeof fetch = fetch,
  ) {
    super();
  }

  override run(request: AgentRunnerRunRequest): Observable<BaseEvent> {
    const tenant = requestTenant((request.agent as { headers?: Record<string, string> }).headers);
    const owner = this.owners.get(request.threadId);
    if (owner !== undefined && owner !== tenant)
      return refused(
        request.threadId,
        request.input.runId,
        'This conversation belongs to another tenant.',
        'thread_of_another_tenant',
      );
    if (tenant) this.owners.set(request.threadId, tenant);
    return super.run(request).pipe(
      finalize(() => {
        if (this.owners.get(request.threadId) === tenant) this.owners.delete(request.threadId);
      }),
    );
  }

  override connect(request: AgentRunnerConnectRequest): Observable<BaseEvent> {
    const tenant = requestTenant(request.headers);
    const runId = `restore-${request.threadId}`;
    if (!tenant)
      return refused(request.threadId, runId, 'Choose a tenant before opening a conversation.', 'no_tenant');
    const send = tenantFetch(this.keys, tenant, this.fetchFn);
    return new Observable<BaseEvent>((subscriber) => {
      let inner: { unsubscribe(): void } | undefined;
      const controller = new AbortController();
      const emitAll = (events: BaseEvent[]) => {
        for (const event of events) subscriber.next(event);
        subscriber.complete();
      };
      const restore = async () => {
        if (await this.isRunning({ threadId: request.threadId })) {
          inner = (
            this.owners.get(request.threadId) === tenant
              ? super.connect(request)
              : refused(request.threadId, runId, 'This conversation belongs to another tenant.', 'thread_of_another_tenant')
          ).subscribe(subscriber);
          return;
        }
        const started = { type: EventType.RUN_STARTED, threadId: request.threadId, runId } as BaseEvent;
        let message: string;
        try {
          const response = await send(
            `${this.config.runtimeUrl}/ag-ui/threads/${encodeURIComponent(request.threadId)}`,
            { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(30000)]) },
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
