import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import { once } from "node:events";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { resolve } from "node:path";
import {
  discoverAndLoadExtensions,
  SessionManager,
  createAgentSession,
  AuthStorage,
  ModelRegistry,
  SettingsManager,
  DefaultResourceLoader,
  type ExtensionCommandContext,
  type ExtensionUIContext,
  type ExtensionHandler,
  type SessionStartEvent,
  type ToolCallEvent,
  type ToolCallEventResult,
  type ExtensionRuntime,
} from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const packageDir = resolve(import.meta.dirname, "..");
const extensionPath = resolve(packageDir, "src/index.ts");
let scratch: string;
let url: string;
let server: ReturnType<typeof createServer>;
let requests: {
  method: string | undefined;
  url: string | undefined;
  authorization: string | undefined;
  body: string;
}[];
let respond: (req: IncomingMessage, res: ServerResponse) => void;

beforeEach(async () => {
  await mkdir(resolve(packageDir, ".test-artifacts"), { recursive: true });
  scratch = await mkdtemp(resolve(packageDir, ".test-artifacts/run-"));
  await mkdir(resolve(scratch, "cwd"));
  await mkdir(resolve(scratch, "agent"));
  requests = [];
  respond = (_req, res) =>
    res.end(
      JSON.stringify({
        data: [{ id: "cogniverse" }, { id: "cogniverse/custom" }],
      }),
    );
  server = createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    requests.push({
      method: req.method,
      url: req.url,
      authorization: req.headers.authorization,
      body,
    });
    respond(req, res);
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  if (!address || typeof address === "string")
    throw new Error("Expected TCP address");
  url = `http://127.0.0.1:${address.port}/v1`;
  vi.stubEnv("COGNIVERSE_BASE_URL", url);
  vi.stubEnv("COGNIVERSE_API_KEY", "test-client-key");
});

afterEach(async () => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  server.closeAllConnections();
  await new Promise<void>((done, reject) =>
    server.close((err) => (err ? reject(err) : done())),
  );
  await rm(scratch, { recursive: true, force: true });
});

async function discover() {
  return discoverAndLoadExtensions(
    [packageDir],
    resolve(scratch, "cwd"),
    resolve(scratch, "agent"),
  );
}

async function load() {
  const result = await discover();
  expect(result.errors).toEqual([]);
  expect(result.extensions.map((ext) => ext.path)).toEqual([extensionPath]);
  const ext = result.extensions[0];
  const start = ext.handlers.get(
    "session_start",
  )![0] as ExtensionHandler<SessionStartEvent>;
  const tool = ext.handlers.get("tool_call")![0] as ExtensionHandler<
    ToolCallEvent,
    ToolCallEventResult
  >;
  const search = ext.commands.get("cogniverse-search")!.handler;
  return { ...result, start, tool, search };
}

function context(manager = SessionManager.inMemory(scratch)) {
  const select = vi
    .fn<ExtensionUIContext["select"]>()
    .mockResolvedValue("Deny");
  const notify = vi.fn<ExtensionUIContext["notify"]>();
  // The real host supplies ExtensionAPI; only these UI methods are substituted.
  // Handlers use a real SessionManager and no other context services.
  const surface = {
    ui: { select, notify } satisfies Pick<
      ExtensionUIContext,
      "select" | "notify"
    >,
    sessionManager: manager,
  };
  return {
    ctx: surface as unknown as ExtensionCommandContext,
    select,
    notify,
    manager,
  };
}

const write: ToolCallEvent = {
  type: "tool_call",
  toolCallId: "write-1",
  toolName: "write",
  input: { path: "notes.md", content: "robots" },
};
const prompt = [
  "cogniverse: allow write? notes.md",
  ["Allow once", "Always allow write (this session)", "Deny"],
];
function grants(manager: SessionManager) {
  return manager
    .getEntries()
    .filter((e) => e.type === "custom")
    .map((e) => ({ customType: e.customType, data: e.data }));
}
function persist(runtime: ExtensionRuntime, active: () => SessionManager) {
  const append = vi.fn<ExtensionRuntime["appendEntry"]>((kind, data) => {
    active().appendCustomEntry(kind, data);
  });
  runtime.appendEntry = append;
  return append;
}

describe("model discovery", () => {
  it("registers exactly the ordered OpenAI catalog returned at load", async () => {
    const loaded = await load();
    expect(requests).toEqual([
      {
        method: "GET",
        url: "/v1/models",
        authorization: "Bearer test-client-key",
        body: "",
      },
    ]);
    expect(loaded.runtime.pendingProviderRegistrations).toEqual([
      {
        name: "cogniverse",
        extensionPath,
        config: {
          name: "Cogniverse",
          baseUrl: url,
          apiKey: "$COGNIVERSE_API_KEY",
          api: "openai-completions",
          models: ["cogniverse", "cogniverse/custom"].map((id) => ({
            id,
            name: id === "cogniverse" ? "Cogniverse (auto-routing)" : id,
            reasoning: false,
            input: ["text"],
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: 16384,
            maxTokens: 4096,
          })),
        },
      },
    ]);
  });

  it("reports the exact model HTTP failure", async () => {
    respond = (_req, res) => {
      res.statusCode = 503;
      res.end("catalog unavailable");
    };
    const result = await discover();
    expect(result.errors).toEqual([
      {
        path: extensionPath,
        error: `Failed to load extension: cogniverse models failed (503) at ${url}/models: catalog unavailable`,
      },
    ]);
    expect(result.extensions).toEqual([]);
    expect(result.runtime.pendingProviderRegistrations).toEqual([]);
  });

  it("reports the exact unreachable model endpoint", async () => {
    respond = (req) => req.socket.destroy();
    const result = await discover();
    expect(result.errors).toEqual([
      {
        path: extensionPath,
        error: `Failed to load extension: cogniverse models unreachable at ${url}: fetch failed`,
      },
    ]);
    expect(result.extensions).toEqual([]);
    expect(result.runtime.pendingProviderRegistrations).toEqual([]);
  });

  it.each([
    {},
    { data: [{ id: 42 }] },
    { data: [{ id: "" }] },
    { data: [{ id: "x" }, { id: "x" }] },
  ])("rejects malformed catalogs %j", async (body) => {
    respond = (_req, res) => res.end(JSON.stringify(body));
    const result = await discover();
    expect(result.errors).toEqual([
      {
        path: extensionPath,
        error: `Failed to load extension: cogniverse models invalid at ${url}/models: expected unique non-empty ids in {data:[{id}]}`,
      },
    ]);
    expect(result.extensions).toEqual([]);
  });

  it("reports invalid model JSON with its endpoint", async () => {
    respond = (_req, res) => res.end("");
    const result = await discover();
    expect(result.errors).toEqual([
      {
        path: extensionPath,
        error: `Failed to load extension: cogniverse models invalid JSON at ${url}/models`,
      },
    ]);
    expect(result.extensions).toEqual([]);
  });

  it("bounds a hung model request and names the endpoint", async () => {
    respond = () => {};
    const result = await discover();
    expect(result.errors).toEqual([
      {
        path: extensionPath,
        error: `Failed to load extension: cogniverse models unreachable at ${url}: The operation was aborted due to timeout`,
      },
    ]);
    expect(result.extensions).toEqual([]);
  }, 15_000);

  it("preserves an empty catalog", async () => {
    respond = (_req, res) => res.end('{"data":[]}');
    const result = await load();
    expect(
      result.runtime.pendingProviderRegistrations[0].config.models,
    ).toEqual([]);
  });
});

describe("search", () => {
  it("reports the exact HTTP failure and sends nothing", async () => {
    const { search, runtime } = await load();
    const send = vi.fn<ExtensionRuntime["sendMessage"]>();
    runtime.sendMessage = send;
    const { ctx, notify } = context();
    respond = (_req, res) => {
      res.statusCode = 502;
      res.end("search unavailable");
    };
    await search("robots", ctx);
    expect(notify.mock.calls).toEqual([
      ["cogniverse search failed (502): search unavailable", "error"],
    ]);
    expect(send.mock.calls).toEqual([]);
  });
  it("reports the exact empty answer and sends nothing", async () => {
    const { search, runtime } = await load();
    const send = vi.fn<ExtensionRuntime["sendMessage"]>();
    runtime.sendMessage = send;
    const { ctx, notify } = context();
    respond = (_req, res) =>
      res.end('{"choices":[{"message":{"content":""}}]}');
    await search("robots", ctx);
    expect(notify.mock.calls).toEqual([
      ["cogniverse search returned no content", "warning"],
    ]);
    expect(send.mock.calls).toEqual([]);
  });
  it("reports the exact rejected fetch and sends nothing", async () => {
    const { search, runtime } = await load();
    const send = vi.fn<ExtensionRuntime["sendMessage"]>();
    runtime.sendMessage = send;
    const { ctx, notify } = context();
    respond = (req) => req.socket.destroy();
    await search("robots", ctx);
    expect(notify.mock.calls).toEqual([
      [`cogniverse search unreachable at ${url}: fetch failed`, "error"],
    ]);
    expect(send.mock.calls).toEqual([]);
  });
  it("sends the exact visible custom message without a turn", async () => {
    const { search, runtime } = await load();
    const send = vi.fn<ExtensionRuntime["sendMessage"]>();
    runtime.sendMessage = send;
    const user = vi.fn<ExtensionRuntime["sendUserMessage"]>();
    runtime.sendUserMessage = user;
    const { ctx, notify } = context();
    respond = (_req, res) =>
      res.end('{"choices":[{"message":{"content":"Robot video A"}}]}');
    await search("  robots  ", ctx);
    expect(send.mock.calls).toEqual([
      [
        {
          customType: "cogniverse-search-results",
          content: '[cogniverse search results for "robots"]\nRobot video A',
          display: true,
          details: { query: "robots", baseUrl: url },
        },
        { triggerTurn: false },
      ],
    ]);
    expect(user.mock.calls).toEqual([]);
    expect(notify.mock.calls).toEqual([
      ["cogniverse search results added to context", "info"],
    ]);
    expect(requests.at(-1)).toEqual({
      method: "POST",
      url: "/v1/chat/completions",
      authorization: "Bearer test-client-key",
      body: JSON.stringify({
        model: "cogniverse/search",
        messages: [{ role: "user", content: "robots" }],
      }),
    });
  });
  it("warns on an empty query without making a search request", async () => {
    const { search } = await load();
    const { ctx, notify } = context();
    const before = requests.slice();
    await search("   ", ctx);
    expect(requests).toEqual(before);
    expect(notify.mock.calls).toEqual([
      ["usage: /cogniverse-search <query>", "warning"],
    ]);
  });
});

describe("permissions", () => {
  it("allows once and prompts for the next write without persistence", async () => {
    const { start, tool, runtime } = await load();
    const { ctx, select, manager } = context();
    const append = persist(runtime, () => manager);
    await start({ type: "session_start", reason: "startup" }, ctx);
    select.mockResolvedValue("Allow once");
    expect(await tool(write, ctx)).toEqual(undefined);
    expect(await tool(write, ctx)).toEqual(undefined);
    expect(select.mock.calls).toEqual([prompt, prompt]);
    expect(append.mock.calls).toEqual([]);
    expect(grants(manager)).toEqual([]);
  });
  it("always allows only the selected tool and restores its persisted grant", async () => {
    const { start, tool, runtime } = await load();
    const { ctx, select, manager } = context();
    persist(runtime, () => manager);
    await start({ type: "session_start", reason: "startup" }, ctx);
    select.mockResolvedValueOnce("Always allow write (this session)");
    expect(await tool(write, ctx)).toEqual(undefined);
    expect(await tool(write, ctx)).toEqual(undefined);
    expect(grants(manager)).toEqual([
      { customType: "cogniverse-permission", data: { tool: "write" } },
    ]);
    const restored = await load();
    await restored.start({ type: "session_start", reason: "startup" }, ctx);
    expect(await restored.tool(write, ctx)).toEqual(undefined);
    expect(select.mock.calls).toEqual([prompt]);
    expect(
      await tool(
        {
          type: "tool_call",
          toolCallId: "bash-1",
          toolName: "bash",
          input: { command: "pwd" },
        },
        ctx,
      ),
    ).toEqual({ block: true, reason: "bash denied by user" });
    expect(select.mock.calls).toEqual([
      prompt,
      [
        "cogniverse: allow bash? pwd",
        ["Allow once", "Always allow bash (this session)", "Deny"],
      ],
    ]);
  });
  it.each(["Deny", undefined])(
    "blocks denied or dismissed writes (%s)",
    async (choice) => {
      const { start, tool, runtime } = await load();
      const { ctx, select, manager } = context();
      persist(runtime, () => manager);
      await start({ type: "session_start", reason: "startup" }, ctx);
      select.mockResolvedValue(choice);
      expect(await tool(write, ctx)).toEqual({
        block: true,
        reason: "write denied by user",
      });
      expect(select.mock.calls).toEqual([prompt]);
      expect(grants(manager)).toEqual([]);
    },
  );
  it("does not prompt for read", async () => {
    const { start, tool } = await load();
    const { ctx, select } = context();
    await start({ type: "session_start", reason: "startup" }, ctx);
    expect(
      await tool(
        {
          type: "tool_call",
          toolCallId: "read-1",
          toolName: "read",
          input: { path: "notes.md" },
        },
        ctx,
      ),
    ).toEqual(undefined);
    expect(select.mock.calls).toEqual([]);
  });
  it.each(["Always allow write (this session)", "Allow once"])(
    "discards session A dialog after session B starts (%s)",
    async (choice) => {
      const { start, tool, runtime } = await load();
      const a = context();
      const b = context();
      let active = a.manager;
      const append = persist(runtime, () => active);
      await start({ type: "session_start", reason: "startup" }, a.ctx);
      let release!: (value: string) => void;
      let opened!: () => void;
      const ready = new Promise<void>((done) => {
        opened = done;
      });
      a.select.mockImplementation(() => {
        opened();
        return new Promise((done) => {
          release = done;
        });
      });
      const pending = tool(write, a.ctx);
      await ready;
      active = b.manager;
      await start({ type: "session_start", reason: "startup" }, b.ctx);
      release(choice);
      const oldResult = await pending;
      const newResult = await tool(write, b.ctx);
      expect(oldResult).toEqual({
        block: true,
        reason: "write permission discarded: session changed",
      });
      expect(newResult).toEqual({
        block: true,
        reason: "write denied by user",
      });
      expect(a.notify.mock.calls).toEqual([
        [
          "cogniverse: discarded write permission because the session changed",
          "warning",
        ],
      ]);
      expect(a.select.mock.calls).toEqual([prompt]);
      expect(b.select.mock.calls).toEqual([prompt]);
      expect(append.mock.calls).toEqual([]);
      expect(grants(a.manager)).toEqual([]);
      expect(grants(b.manager)).toEqual([]);
      console.log(
        JSON.stringify({
          case: "GRANT_AFTER_SESSION_SWITCH",
          choice,
          promptsA: a.select.mock.calls.length,
          promptsB: b.select.mock.calls.length,
          oldResult,
          newResult,
          persistedEntries: grants(b.manager),
        }),
      );
    },
  );
});

it("inserts and persists search context through real AgentSession.sendCustomMessage", async () => {
  const { search, runtime } = await load();
  const cwd = resolve(scratch, "cwd");
  const agentDir = resolve(scratch, "agent");
  const manager = SessionManager.create(cwd, resolve(scratch, "sessions"));
  // Pi flushes session files after an assistant message exists.
  manager.appendMessage({
    role: "assistant",
    content: [{ type: "text", text: "Ready" }],
    api: "openai-completions",
    provider: "cogniverse",
    model: "cogniverse",
    timestamp: 1,
    stopReason: "stop",
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
  });
  const settingsManager = SettingsManager.inMemory();
  const authStorage = AuthStorage.inMemory();
  const resourceLoader = new DefaultResourceLoader({
    cwd,
    agentDir,
    settingsManager,
    noExtensions: true,
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
    noContextFiles: true,
  });
  await resourceLoader.reload();
  const { session } = await createAgentSession({
    cwd,
    agentDir,
    sessionManager: manager,
    settingsManager,
    authStorage,
    modelRegistry: ModelRegistry.inMemory(authStorage),
    resourceLoader,
    tools: [],
  });
  try {
    const { ctx } = context(manager);
    let delivery: Promise<void> = Promise.resolve();
    runtime.sendMessage = (message, options) => {
      delivery = session.sendCustomMessage(message, options);
    };
    const events: unknown[] = [];
    session.subscribe((event) => events.push(event));
    const timestamp = 1234567890000;
    vi.spyOn(Date, "now").mockReturnValue(timestamp);
    respond = (_req, res) =>
      res.end('{"choices":[{"message":{"content":"Robot video A"}}]}');
    await search("robots", ctx);
    await delivery;
    const message = {
      role: "custom",
      customType: "cogniverse-search-results",
      content: '[cogniverse search results for "robots"]\nRobot video A',
      display: true,
      details: { query: "robots", baseUrl: url },
      timestamp,
    };
    expect(session.messages.filter((m) => m.role === "custom")).toEqual([
      message,
    ]);
    expect(events).toEqual([
      { type: "message_start", message },
      { type: "message_end", message },
    ]);
    expect(session.isStreaming).toEqual(false);
    const file = manager.getSessionFile();
    if (!file) throw new Error("Expected a persisted session file");
    const restored = SessionManager.open(file, resolve(scratch, "sessions"));
    expect(
      restored
        .getEntries()
        .filter((e) => e.type === "custom_message")
        .map(({ type, customType, content, display, details }) => ({
          type,
          customType,
          content,
          display,
          details,
        })),
    ).toEqual([
      {
        type: "custom_message",
        customType: "cogniverse-search-results",
        content: message.content,
        display: true,
        details: message.details,
      },
    ]);
    expect(requests.map((r) => r.url)).toEqual([
      "/v1/models",
      "/v1/chat/completions",
    ]);
  } finally {
    session.dispose();
  }
});
