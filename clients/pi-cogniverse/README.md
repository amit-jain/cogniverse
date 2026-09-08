# pi-cogniverse

Pi 0.80.6 extension for the Cogniverse runtime's OpenAI-compatible `/v1` API.
The provider discovers its model catalog from `GET /v1/models` at load time.
The response is `{ "data": [{ "id": "cogniverse" }] }`; ids must be unique,
non-empty strings. Discovery failures stop loading with an error. Requests
have a ten-second timeout.

## Setup

Use Node.js 22 and a runtime exposing `/v1/models` and `/v1/chat/completions`.
Set the runtime's harness key and base URL:

```bash
export COGNIVERSE_API_KEY='<harness-key>'
export COGNIVERSE_BASE_URL=http://localhost:8000/v1
```

The base URL defaults to `http://localhost:8000/v1`. In Pi's global
`~/.pi/agent/settings.json` or project `.pi/settings.json`, reference the
local package using its absolute path:

```json
{ "packages": ["/absolute/path/to/cogniverse/clients/pi-cogniverse"] }
```

Select the `cogniverse` provider in `/model`. Pi executes the workspace tool
calls returned by the runtime.

## Permissions and search

`bash`, `write`, and `edit` prompt for Allow once, Always allow this tool for
this session, or Deny. Grants persist in that session's entries and are
restored when the session loads. Switching sessions discards outstanding
dialog results with a notification.

`/cogniverse-search <query>` requests the `cogniverse/search` model and adds
its answer as a visible `cogniverse-search-results` custom message without
starting a model turn. Message details record the query and runtime base URL.
HTTP errors, empty answers, and connection failures produce notifications.

## Tests

```bash
cd clients/pi-cogniverse
npm ci
npm run typecheck
npm test
```

Vitest loads this package through Pi's real `discoverAndLoadExtensions` host
and uses real `SessionManager` instances. UI `select` and `notify` are typed
recorders; the handlers use no other UI services. Tests serve the HTTP
responses over a local TCP socket. Message delivery also exercises the real
`AgentSession.sendCustomMessage` method and reopens its persisted session file.
