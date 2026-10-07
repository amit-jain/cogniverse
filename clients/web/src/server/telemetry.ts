// CopilotKit reports usage to its own servers unless told not to. A
// self-hosted Cogniverse sends nothing out unless the operator sets
// COPILOTKIT_TELEMETRY_DISABLED=false. Imported before any CopilotKit module,
// which reads the flag when it loads.
process.env.COPILOTKIT_TELEMETRY_DISABLED ??= 'true';
