import type {
  ExtensionAPI,
  ProviderModelConfig,
} from "@earendil-works/pi-coding-agent";

const PERMISSION_ENTRY = "cogniverse-permission";
const SEARCH_RESULTS = "cogniverse-search-results";
const GATED_TOOLS = new Set(["bash", "write", "edit"]);
const REQUEST_TIMEOUT_MS = 10_000;

const MODEL_DEFAULTS = {
  reasoning: false,
  input: ["text"] as ProviderModelConfig["input"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 16384,
  maxTokens: 4096,
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function discoverModels(baseUrl: string): Promise<ProviderModelConfig[]> {
  let response: Response;
  let content: string;
  try {
    response = await fetch(`${baseUrl}/models`, {
      headers: {
        Authorization: `Bearer ${process.env.COGNIVERSE_API_KEY || ""}`,
      },
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    content = await response.text();
  } catch (error) {
    throw new Error(
      `cogniverse models unreachable at ${baseUrl}: ${error instanceof Error ? error.message : String(error)}`,
      { cause: error },
    );
  }
  if (!response.ok) {
    throw new Error(
      `cogniverse models failed (${response.status}) at ${baseUrl}/models: ${content.slice(0, 200)}`,
    );
  }
  let body: unknown;
  try {
    body = JSON.parse(content);
  } catch (error) {
    throw new Error(`cogniverse models invalid JSON at ${baseUrl}/models`, {
      cause: error,
    });
  }
  if (
    !isRecord(body) ||
    !Array.isArray(body.data) ||
    !body.data.every(
      (model: unknown) =>
        isRecord(model) &&
        typeof model.id === "string" &&
        model.id.trim() !== "",
    ) ||
    new Set(body.data.map((model) => model.id)).size !== body.data.length
  ) {
    throw new Error(
      `cogniverse models invalid at ${baseUrl}/models: expected unique non-empty ids in {data:[{id}]}`,
    );
  }
  return body.data.map(({ id }: { id: string }) => ({
    ...MODEL_DEFAULTS,
    id,
    name: id === "cogniverse" ? "Cogniverse (auto-routing)" : id,
  }));
}

export default async function (pi: ExtensionAPI) {
  const baseUrl = process.env.COGNIVERSE_BASE_URL || "http://localhost:8000/v1";
  pi.registerProvider("cogniverse", {
    name: "Cogniverse",
    baseUrl,
    apiKey: "$COGNIVERSE_API_KEY",
    api: "openai-completions",
    models: await discoverModels(baseUrl),
  });

  let currentSession: { id: string; grants: Set<string> } | undefined;
  pi.on("session_start", (_event, ctx) => {
    currentSession = {
      id: ctx.sessionManager.getSessionId(),
      grants: new Set(),
    };
    for (const entry of ctx.sessionManager.getEntries()) {
      if (
        entry.type === "custom" &&
        entry.customType === PERMISSION_ENTRY &&
        isRecord(entry.data) &&
        typeof entry.data.tool === "string" &&
        GATED_TOOLS.has(entry.data.tool)
      ) {
        currentSession.grants.add(entry.data.tool);
      }
    }
  });

  pi.on("tool_call", async (event, ctx) => {
    const tool = event.toolName;
    if (!GATED_TOOLS.has(tool)) return undefined;
    const session = currentSession;
    const sessionId = ctx.sessionManager.getSessionId();
    if (!session || session.id !== sessionId) {
      return {
        block: true,
        reason: `${tool} permission unavailable: session not initialized`,
      };
    }
    if (session.grants.has(tool)) return undefined;
    const summary =
      tool === "bash"
        ? String(event.input.command ?? "")
        : String("path" in event.input ? event.input.path : "");
    const choice = await ctx.ui.select(
      `cogniverse: allow ${tool}? ${summary}`.slice(0, 200),
      ["Allow once", `Always allow ${tool} (this session)`, "Deny"],
    );
    if (
      currentSession !== session ||
      ctx.sessionManager.getSessionId() !== sessionId
    ) {
      ctx.ui.notify(
        `cogniverse: discarded ${tool} permission because the session changed`,
        "warning",
      );
      return {
        block: true,
        reason: `${tool} permission discarded: session changed`,
      };
    }
    if (choice === "Allow once") return undefined;
    if (choice === `Always allow ${tool} (this session)`) {
      pi.appendEntry(PERMISSION_ENTRY, { tool });
      session.grants.add(tool);
      return undefined;
    }
    return { block: true, reason: `${tool} denied by user` };
  });

  pi.registerCommand("cogniverse-search", {
    description: "Search cogniverse directly; inject results as context",
    handler: async (args, ctx) => {
      const query = args.trim();
      if (!query) {
        ctx.ui.notify("usage: /cogniverse-search <query>", "warning");
        return;
      }
      try {
        const resp = await fetch(`${baseUrl}/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${process.env.COGNIVERSE_API_KEY || ""}`,
          },
          body: JSON.stringify({
            model: "cogniverse/search",
            messages: [{ role: "user", content: query }],
          }),
          signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
        });
        if (!resp.ok) {
          const detail = await resp.text();
          ctx.ui.notify(
            `cogniverse search failed (${resp.status}): ${detail.slice(0, 200)}`,
            "error",
          );
          return;
        }
        const body = await resp.json();
        const answer = body?.choices?.[0]?.message?.content;
        if (!answer) {
          ctx.ui.notify("cogniverse search returned no content", "warning");
          return;
        }
        pi.sendMessage(
          {
            customType: SEARCH_RESULTS,
            content: `[cogniverse search results for "${query}"]\n${answer}`,
            display: true,
            details: { query, baseUrl },
          },
          { triggerTurn: false },
        );
        ctx.ui.notify("cogniverse search results added to context", "info");
      } catch (error) {
        ctx.ui.notify(
          `cogniverse search unreachable at ${baseUrl}: ${error instanceof Error ? error.message : String(error)}`,
          "error",
        );
      }
    },
  });
}
