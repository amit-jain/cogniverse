# Messaging Gateway — Telegram Integration

## Context

Cogniverse agents are accessible only via REST API and the Streamlit dashboard. Users need a conversational interface through messaging platforms to query, search, summarize, and get reports without opening a browser. Telegram is the first platform, with Slack planned later.

This is Hermes Gap 3 — the final gap identified from the Hermes agent comparison.

## Decisions

- **Deployment**: Separate service (own pod), same container image, different entrypoint. Calls runtime via HTTP API. No agent logic in gateway.
- **Telegram mode**: Webhook for production (k3d/cloud), long polling for local dev. Configurable via `GATEWAY_MODE` env var.
- **Auth**: Invite token. Admin generates token via `POST /admin/messaging/invite`, user sends `/start <token>` to register. Token maps Telegram user ID to tenant_id.
- **Conversation storage**: Mem0 via existing `Mem0MemoryManager`. Same as dashboard. Per-chat history stored with `agent_name="_messaging_gateway"`.
- **Agent access**: Full — slash commands target specific agents, plain text goes to routing_agent, media (images/videos) forwarded with file URL.

## Architecture

```
Telegram User → Telegram Bot API → Messaging Gateway (separate pod)
                                        │
                                        ├─ TelegramHandler (parse, format, chunk)
                                        ├─ CommandRouter (/search, /summarize, /report, plain text)
                                        ├─ ConversationManager (Mem0 history)
                                        ├─ RuntimeClient (HTTP to runtime API)
                                        └─ Auth (invite tokens, user→tenant mapping)
                                        │
                                        ▼ HTTP
                                    Runtime (existing, localhost:28000)
                                    /agents/{name}/process
                                    /events/{id}
                                    /admin/tenants
```

## Message Flow

### Inbound (Telegram → Runtime)

1. User sends message (text, image, video, or command)
2. Gateway receives via webhook or polling
3. TelegramHandler parses message type and extracts content
4. Auth looks up `telegram_user_id → tenant_id` mapping from Mem0. If not mapped, rejects with registration instructions.
5. ConversationManager loads recent history from Mem0 (`search_memory`)
6. CommandRouter determines target agent:
   - `/search <query>` → search_agent
   - `/summarize <query>` → summarizer_agent
   - `/report <query>` → detailed_report_agent
   - `/research <query>` → deep_research_agent
   - `/help` → list available commands
   - `/start <token>` → registration flow
   - plain text → routing_agent (auto-routes)
   - image/video attachment → search_agent with media URL in context
7. RuntimeClient posts to `/agents/{agent_name}/process` with query, tenant_id, context_id (chat_id), conversation_history
8. For streaming agents: subscribe to `/events/{task_id}` SSE, send typing indicator, update message on each chunk
9. For non-streaming agents: wait for response

### Outbound (Runtime → Telegram)

1. Parse response: message, results, results_count
2. Send main message, chunked at 4096 chars if needed
3. Format search results as numbered list with titles and descriptions
4. Store assistant turn in Mem0 via `add_memory`
5. If results contain media URLs, send as Telegram media group

### Registration (/start)

1. Admin calls `POST /admin/messaging/invite` with `tenant_id` and optional `expires_in_hours`
2. Runtime generates UUID token, stores in ConfigStore with tenant_id and expiry
3. Admin shares token with user
4. User sends `/start <token>` to bot
5. Gateway validates token (exists, not expired, not used)
6. Maps `telegram_user_id → tenant_id` in Mem0 with `agent_name="_messaging_gateway"`
7. Marks token as used
8. Replies "Registered as {tenant_name}. Send /help for commands."

## Files

### New files

| File | Purpose |
|------|---------|
| `libs/messaging/cogniverse_messaging/__init__.py` | Package init |
| `libs/messaging/cogniverse_messaging/gateway.py` | Main gateway: start/stop, webhook/polling mode selection |
| `libs/messaging/cogniverse_messaging/telegram_handler.py` | Parse updates, format responses, chunk messages, media handling |
| `libs/messaging/cogniverse_messaging/command_router.py` | Command parsing, agent mapping, help text |
| `libs/messaging/cogniverse_messaging/conversation.py` | ConversationManager: Mem0 history load/store per chat_id |
| `libs/messaging/cogniverse_messaging/runtime_client.py` | HTTP client: agent process, events SSE, health check |
| `libs/messaging/cogniverse_messaging/auth.py` | Invite token generation, validation, user-tenant mapping |
| `libs/messaging/pyproject.toml` | Package config, deps: python-telegram-bot, httpx |
| `tests/messaging/unit/test_command_router.py` | Command parsing, agent mapping |
| `tests/messaging/unit/test_telegram_handler.py` | Response formatting, chunking |
| `tests/messaging/unit/test_auth.py` | Token generation, validation, expiry |
| `tests/messaging/integration/test_gateway_integration.py` | Real Mem0, mock Telegram updates, runtime API calls |
| `tests/e2e/test_messaging_e2e.py` | k3d: pod running, health endpoint, invite token API |

### Modified files

| File | Change |
|------|--------|
| `configs/config.json` | Add `messaging` section |
| `charts/cogniverse/values.yaml` | Add `messaging` section |
| `charts/cogniverse/templates/all-resources.yaml` | Add messaging gateway Deployment + Service |
| `libs/runtime/cogniverse_runtime/routers/admin.py` | Add `POST /admin/messaging/invite` endpoint |
| `pyproject.toml` | Add `cogniverse-messaging` to workspace members |

## Configuration

### config.json

```json
{
  "messaging": {
    "telegram": {
      "enabled": true,
      "max_message_length": 4096,
      "max_results_per_message": 5
    }
  }
}
```

### Environment variables (secrets, not in config.json)

- `TELEGRAM_BOT_TOKEN` — Bot token from @BotFather
- `GATEWAY_MODE` — `webhook` or `polling` (default: `polling`)
- `TELEGRAM_WEBHOOK_URL` — Public URL for webhook mode
- `RUNTIME_URL` — Runtime API URL (default: `http://cogniverse-runtime:28000`)
- `BACKEND_URL` / `BACKEND_PORT` — For Mem0 initialization

### Helm values

```yaml
messaging:
  enabled: true
  image:
    repository: cogniverse/runtime  # Same image
    tag: "2.0.0"
  replicaCount: 1
  mode: polling  # webhook for production
  resources:
    limits:
      cpu: "1"
      memory: "1Gi"
    requests:
      cpu: "250m"
      memory: "256Mi"
  env:
    GATEWAY_MODE: "polling"
```

Bot token stored as Kubernetes Secret, referenced via `secretKeyRef`.

## Dependencies

- `python-telegram-bot>=20.0` — async Telegram Bot API (v20+ uses asyncio natively)
- `httpx` — async HTTP client for runtime API calls (already in workspace)

## Testing

### Unit tests (no external services)

- `test_command_router.py` — parse commands, map to agent names, extract query text, handle media, unknown commands
- `test_telegram_handler.py` — format responses, chunk at 4096 chars, format search results, media groups
- `test_auth.py` — token generation, validation, expiry, user mapping CRUD

### Integration tests (real services)

- `test_gateway_integration.py` — mock Telegram updates fed to gateway, verify runtime API calls via httpx mock transport. Real Mem0 for conversation history using `shared_memory_vespa` fixture. Real invite token flow: generate, validate, map, dispatch.

### E2E k3d tests

- `test_messaging_e2e.py` — verify messaging gateway pod running and healthy, invite token creation via admin API works. No actual Telegram interaction (would need a test bot).

## Future Enhancements (Not in Scope)

- Slack integration (same gateway pattern, add SlackHandler)
- Inline keyboard buttons for result interaction
- File upload processing (PDF, documents)
- Voice message transcription
- Group chat support with @mention filtering
