{{/*
Expand the name of the chart.
*/}}
{{- define "cogniverse.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "cogniverse.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "cogniverse.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "cogniverse.labels" -}}
helm.sh/chart: {{ include "cogniverse.chart" . }}
{{ include "cogniverse.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | replace "+" "_" | trunc 63 | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "cogniverse.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cogniverse.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Component labels
*/}}
{{- define "cogniverse.componentLabels" -}}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "cogniverse.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "cogniverse.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "cogniverse.image" -}}
{{- $registryName := .imageRoot.registry -}}
{{- $repositoryName := .imageRoot.repository -}}
{{- $tag := .imageRoot.tag | toString -}}
{{- if .global }}
    {{- if .global.imageRegistry }}
     {{- $registryName = .global.imageRegistry -}}
    {{- end -}}
{{- end -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else }}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end -}}
{{- end -}}

{{/*
LLM endpoint URL. The chart talks to every LLM backend through the
OpenAI-compatible HTTP shape (``/v1/chat/completions``,
``/v1/embeddings``); modern Ollama, vLLM, and external SaaS providers
all expose it. This helper picks the backend host:port based on
``llm.engine`` plus the ``llm.external.enabled`` switch:

  - ``external`` (engine)       → ``llm.external.url`` (required)
  - ``external.enabled = true`` → ``llm.external.url`` (overrides in-cluster)
  - ``vllm`` / ``ollama``       → in-cluster ``-llm`` service on the
                                   matching backend port

The ``/v1`` suffix is appended so litellm's openai provider can post
chat/completions directly. Single source of truth for every consumer
(runtime, agents, init jobs, optimization workflows) so the URL never
drifts between sites.
*/}}
{{- define "cogniverse.llmEndpoint" -}}
{{- $engine := .Values.llm.engine | default "ollama" -}}
{{- if .Values.llm.external.enabled -}}
{{- required "llm.external.url is required when llm.external.enabled=true" .Values.llm.external.url -}}
{{- else if eq $engine "external" -}}
{{- required "llm.external.url is required when llm.engine=external" .Values.llm.external.url -}}
{{- else if eq $engine "vllm" -}}
{{- printf "http://%s-llm:%d/v1" (include "cogniverse.fullname" .) (int .Values.llm.vllm.service.port) -}}
{{- else if eq $engine "ollama" -}}
{{- printf "http://%s-llm:%d/v1" (include "cogniverse.fullname" .) (int .Values.llm.ollama.service.port) -}}
{{- else -}}
{{- fail (printf "llm.engine must be one of [ollama, vllm, external], got %q" $engine) -}}
{{- end -}}
{{- end -}}

{{/*
Whether the chart should deploy an in-cluster LLM pod (true for engine
ollama / vllm; false for engine external).
*/}}
{{- define "cogniverse.llmDeploysPod" -}}
{{- $engine := .Values.llm.engine | default "ollama" -}}
{{- if or (eq $engine "ollama") (eq $engine "vllm") -}}true{{- end -}}
{{- end -}}

{{/*
vLLM student LLM endpoint — runtime hot path. Where the student is actually
served, with the ``/v1`` suffix so the generic OpenAI-compatible client can hit
/v1/chat/completions. ``runtime.primaryLLM.apiBase`` wins when set: it is how
the student moves off-cluster, and building the Service URL unconditionally
pointed every consumer at a Service that no longer exists. Reads the value
directly rather than delegating to primaryLLMEndpoint, which falls back here.
*/}}
{{- define "cogniverse.llmStudentEndpoint" -}}
{{- if and .Values.runtime.primaryLLM .Values.runtime.primaryLLM.apiBase -}}
{{- .Values.runtime.primaryLLM.apiBase -}}
{{- else -}}
{{- include "cogniverse.llmStudentServiceUrl" . -}}
{{- end -}}
{{- end -}}

{{/* The in-cluster vllm-llm-student Service's /v1 URL. */}}
{{- define "cogniverse.llmStudentServiceUrl" -}}
{{- printf "http://%s-vllm-llm-student:%d/v1" (include "cogniverse.fullname" .) (int .Values.inference.vllm_llm_student.service.port) -}}
{{- end -}}

{{/*
host:port/path an in-cluster URL resolves to, so every spelling of one
Service compares equal: case, a trailing slash, the scheme's default port,
and the release namespace's .<ns>, .<ns>.svc and .<ns>.svc.cluster.local
suffixes (with or without the root dot) are normalised away.
Call with (dict "url" $url "root" $).
*/}}
{{- define "cogniverse.clusterEndpointKey" -}}
{{- $url := .url | lower | trimSuffix "/" -}}
{{- $https := hasPrefix "https://" $url -}}
{{- $rest := $url | trimPrefix "https://" | trimPrefix "http://" -}}
{{- $parts := splitn "/" 2 $rest -}}
{{- $hostPort := splitList ":" $parts._0 -}}
{{- $host := index $hostPort 0 | trimSuffix "." -}}
{{- $port := ternary "443" "80" $https -}}
{{- if gt (len $hostPort) 1 -}}
{{- $port = index $hostPort 1 -}}
{{- end -}}
{{- $namespace := .root.Release.Namespace | lower -}}
{{- range list ".svc.cluster.local" ".svc" -}}
{{- $suffix := printf ".%s%s" $namespace . -}}
{{- if hasSuffix $suffix $host -}}
{{- $host = trimSuffix $suffix $host -}}
{{- end -}}
{{- end -}}
{{- $host = trimSuffix (printf ".%s" $namespace) $host -}}
{{- printf "%s:%s/%s" $host $port ($parts._1 | default "" | trimSuffix "/") -}}
{{- end -}}

{{/*
vLLM teacher LLM endpoint: the DSPy compile-time teacher and the backend
the semantic router serves pro-reasoning from. Resolves to the teacher
entry in ``cogniverse.inferenceServiceUrls`` when that service is rendered
with ``externalUrl``; otherwise to the in-cluster vllm-llm-teacher service,
which ships at scale-to-zero (replicaCount: 0) until a device overlay or an
optimization_cli run scales it.
*/}}
{{- define "cogniverse.llmTeacherEndpoint" -}}
{{- $urls := include "cogniverse.inferenceServiceUrls" . | fromJson -}}
{{- $teacher := index $urls "vllm_llm_teacher" -}}
{{- if $teacher -}}
{{- printf "%s/v1" $teacher -}}
{{- else -}}
{{- printf "http://%s-vllm-llm-teacher:%d/v1" (include "cogniverse.fullname" .) (int .Values.inference.vllm_llm_teacher.service.port) -}}
{{- end -}}
{{- end -}}

{{/*
litellm provider prefix the runtime emits for every model id. The
chart talks to all backends through the OpenAI-compatible wire
contract (vLLM, modern Ollama, OpenAI proper, Modal, Anyscale,
Together…) so one prefix routes them all; the actual destination is
selected by ``api_base``. Centralised here so any future provider
swap is a one-line change.
*/}}
{{- define "cogniverse.llmProviderPrefix" -}}openai{{- end -}}

{{/*
Default api_key value for endpoints that don't require authentication
(in-cluster vLLM/Ollama). litellm's openai-compatible client refuses
to dispatch with a null/empty key, so we hand it a sentinel string;
the receiving server ignores Authorization headers it doesn't check.
External SaaS providers should override via runtime.primaryLLM.apiKey
or by injecting the real secret into the runtime pod's environment.
*/}}
{{- define "cogniverse.llmPlaceholderApiKey" -}}placeholder-no-auth-needed{{- end -}}

{{/*
Secret holding the inference bearer, or empty when every enabled inference
service is in-cluster. The one place the Secret name and the
externalUrl test live; both the env source below and the spawned-workflow
wiring resolve the credential through it.
*/}}
{{- define "cogniverse.inferenceApiKeySecret" -}}
{{- $needsExternalKey := false -}}
{{- range $name, $cfg := .Values.inference -}}
{{- if $cfg.externalUrl -}}
{{- $needsExternalKey = true -}}
{{- end -}}
{{- end -}}
{{- if $needsExternalKey -}}cogniverse-inference-api-key{{- end -}}
{{- end -}}

{{/*
Runtime inference API key source. When any enabled inference service
is rendered with a non-empty externalUrl, the runtime and ingestor
consume the synced cogniverse-inference-api-key Secret. Fully in-cluster
renders keep the no-auth placeholder so local-only stacks do not need the
Secret at all.
*/}}
{{- define "cogniverse.inferenceApiKeyEnv" -}}
{{- $secret := include "cogniverse.inferenceApiKeySecret" . -}}
{{- if $secret -}}
valueFrom:
  secretKeyRef:
    name: {{ $secret }}
    key: COGNIVERSE_INFERENCE_API_KEY
    optional: false
{{- else -}}
value: {{ include "cogniverse.llmPlaceholderApiKey" . | quote }}
{{- end -}}
{{- end -}}

{{/*
Name of the shared optimization WorkflowTemplate. The runtime route, the
quality monitor and the annotation-feedback cron all submit Workflows that
reference it by this name, so it is resolved in one place.
*/}}
{{- define "cogniverse.optimizationWorkflowTemplateName" -}}
{{ include "cogniverse.fullname" . }}-optimization-runner
{{- end -}}

{{/*
Env for a pod that submits optimization Workflows: the name of the shared
WorkflowTemplate the submitted Workflow references. That template owns the
container spec, the backend/inference/LLM env, the resource requests and
limits, the config mount and the per-tenant mutex.
*/}}
{{- define "cogniverse.optimizationWorkflowEnv" -}}
{{- if .Values.argo.enabled -}}
- name: OPTIMIZATION_WORKFLOW_TEMPLATE
  value: {{ include "cogniverse.optimizationWorkflowTemplateName" . }}
{{- end -}}
{{- end -}}

{{/*
Teacher LLM model id passed to litellm. ``inference.vllm_llm_teacher.model``
is the bare model name vLLM serves; the prefix is supplied by
``cogniverse.llmProviderPrefix``. config.json's teacher.model and the
pod's ``args[1]`` must reference the same bare model string —
vLLM's served-model-name defaults to whatever path is passed to
``vllm serve``.
*/}}
{{- define "cogniverse.teacherLLMModel" -}}
{{- printf "%s/%s" (include "cogniverse.llmProviderPrefix" .) .Values.inference.vllm_llm_teacher.model -}}
{{- end -}}

{{/* Same as ``cogniverse.teacherLLMModel`` WITHOUT the provider prefix: the
served model name an OAI-compatible endpoint expects. */}}
{{- define "cogniverse.teacherLLMModelBare" -}}
{{- .Values.inference.vllm_llm_teacher.model -}}
{{- end -}}

{{/*
Runtime primary LLM model — the model id the runtime hands to
``dspy.LM`` verbatim. ``create_dspy_lm`` does no string manipulation
on it, so the chart is the single place that picks the model id.

Resolution order:
  1. ``runtime.primaryLLM.model`` if explicitly set — escape hatch for
     ops who want to override the chart's default.
  2. Bare model id derived from the chosen engine:
       - ``vllm``     → ``inference.vllm_llm_student.model``
       - else         → ``llm.model``
     prefixed with ``cogniverse.llmProviderPrefix``.
*/}}
{{- define "cogniverse.primaryLLMModel" -}}
{{- if and .Values.runtime.primaryLLM .Values.runtime.primaryLLM.model -}}
{{- .Values.runtime.primaryLLM.model -}}
{{- else -}}
{{- $prefix := include "cogniverse.llmProviderPrefix" . -}}
{{- $engine := .Values.llm.engine | default "ollama" -}}
{{- if eq $engine "vllm" -}}
{{- printf "%s/%s" $prefix .Values.inference.vllm_llm_student.model -}}
{{- else -}}
{{- printf "%s/%s" $prefix .Values.llm.model -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Same as ``cogniverse.primaryLLMModel`` but WITHOUT the provider prefix.
Use this when sending the model id directly to an OAI-compatible
``/v1/chat/completions`` endpoint (vLLM, llama.cpp server, etc.) —
those servers reject ``openai/google/gemma-4-e4b-it`` with 404 because
the actual served model name is ``google/gemma-4-e4b-it``. DSPy /
litellm needs the prefix for provider routing; raw HTTP does not.
*/}}
{{- define "cogniverse.primaryLLMModelBare" -}}
{{- if and .Values.runtime.primaryLLM .Values.runtime.primaryLLM.model -}}
{{- .Values.runtime.primaryLLM.model -}}
{{- else -}}
{{- $engine := .Values.llm.engine | default "ollama" -}}
{{- if eq $engine "vllm" -}}
{{- .Values.inference.vllm_llm_student.model -}}
{{- else -}}
{{- .Values.llm.model -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Runtime primary LLM api_base. Resolves to ``runtime.primaryLLM.apiBase``
when set, otherwise to the endpoint matching ``llm.engine``:
  - ollama   → ``cogniverse.llmEndpoint`` (the in-cluster Ollama service)
  - vllm     → ``cogniverse.llmStudentEndpoint`` (in-cluster vllm-llm-student)
  - external → ``cogniverse.llmEndpoint`` (the configured external URL)
*/}}
{{- define "cogniverse.primaryLLMEndpoint" -}}
{{- if and .Values.runtime.primaryLLM .Values.runtime.primaryLLM.apiBase -}}
{{- .Values.runtime.primaryLLM.apiBase -}}
{{- else -}}
{{- $engine := .Values.llm.engine | default "ollama" -}}
{{- if eq $engine "vllm" -}}
{{- include "cogniverse.llmStudentEndpoint" . -}}
{{- else -}}
{{- include "cogniverse.llmEndpoint" . -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Runtime service URL. All agents share the runtime pod, so the orchestrator
dispatches to them via path routing on this single URL
(e.g. http://<runtime>:8000/agents/<name>/process).
*/}}
{{- define "cogniverse.runtimeUrl" -}}
{{- printf "http://%s-runtime:%d" (include "cogniverse.fullname" .) (int .Values.runtime.service.port) -}}
{{- end -}}

{{/*
Vespa backend base URL (scheme + host, no port). Used by backend.url in
config.json so VespaBackend constructed from config (e.g. inside ingestion
worker tasks) reaches the in-cluster service rather than localhost.
*/}}
{{- define "cogniverse.vespaUrl" -}}
{{- printf "http://%s-vespa" (include "cogniverse.fullname" .) -}}
{{- end -}}

{{/*
Semantic-router entry URL — the Envoy data plane the runtime routes its
agent LLM calls through when ``semanticRouter.enabled``. The runtime reads
this from SEMANTIC_ROUTER_URL and rewrites each agent's api_base to it.
*/}}
{{- define "cogniverse.semanticRouterUrl" -}}
{{- printf "http://%s-semantic-router-envoy:%d/v1" (include "cogniverse.fullname" .) (int .Values.semanticRouter.envoy.service.port) -}}
{{- end -}}

{{/* host:port of an OpenAI-compatible endpoint URL, scheme and /v1 stripped.
The srEndpoint* helpers take the endpoint URL as their context. */}}
{{- define "cogniverse.srEndpointHostPort" -}}
{{- . | trimPrefix "http://" | trimPrefix "https://" | trimSuffix "/" | trimSuffix "/v1" | trimSuffix "/" -}}
{{- end -}}

{{- define "cogniverse.srEndpointHost" -}}
{{- index (include "cogniverse.srEndpointHostPort" . | splitList ":") 0 -}}
{{- end -}}

{{/* https when the endpoint is TLS, otherwise http. Envoy dials the backend
itself, so this decides both the default port and the transport it speaks. */}}
{{- define "cogniverse.srEndpointProtocol" -}}
{{- if hasPrefix "https://" . }}https{{ else }}http{{ end -}}
{{- end -}}

{{- define "cogniverse.srEndpointPort" -}}
{{- $hp := include "cogniverse.srEndpointHostPort" . | splitList ":" -}}
{{- if gt (len $hp) 1 -}}
{{- index $hp 1 -}}
{{- else if eq (include "cogniverse.srEndpointProtocol" .) "https" -}}
443
{{- else -}}
80
{{- end -}}
{{- end -}}

{{/* The student backend Envoy serves basic-chat from: the runtime's own LLM
endpoint. */}}
{{- define "cogniverse.srUpstreamHost" -}}
{{- include "cogniverse.srEndpointHost" (include "cogniverse.primaryLLMEndpoint" .) -}}
{{- end -}}

{{- define "cogniverse.srUpstreamPort" -}}
{{- include "cogniverse.srEndpointPort" (include "cogniverse.primaryLLMEndpoint" .) -}}
{{- end -}}

{{- define "cogniverse.srUpstreamProtocol" -}}
{{- include "cogniverse.srEndpointProtocol" (include "cogniverse.primaryLLMEndpoint" .) -}}
{{- end -}}

{{/* A missing teacher serves pro decisions on the student and names why. */}}
{{- define "cogniverse.srTierDegraded" -}}
{{- if and .Values.semanticRouter.enabled (not .Values.inference.vllm_llm_teacher.enabled) (not .Values.inference.vllm_llm_teacher.externalUrl) -}}
pro_model_unavailable
{{- end -}}
{{- end -}}

{{- define "cogniverse.srTierWarning" -}}
{{- if include "cogniverse.srTierDegraded" . -}}
WARNING: pro_model_unavailable: pro-reasoning uses the student because inference.vllm_llm_teacher.enabled=false and inference.vllm_llm_teacher.externalUrl is empty.
{{- end -}}
{{- end -}}

{{- define "cogniverse.srProCluster" -}}
{{- if include "cogniverse.srTierDegraded" . -}}
llm_upstream
{{- else -}}
llm_teacher
{{- end -}}
{{- end -}}

{{- define "cogniverse.srProEndpoint" -}}
{{- if include "cogniverse.srTierDegraded" . -}}
{{- include "cogniverse.primaryLLMEndpoint" . -}}
{{- else -}}
{{- include "cogniverse.llmTeacherEndpoint" . -}}
{{- end -}}
{{- end -}}

{{- define "cogniverse.srProModelBare" -}}
{{- if include "cogniverse.srTierDegraded" . -}}
{{- include "cogniverse.primaryLLMModelBare" . -}}
{{- else -}}
{{- include "cogniverse.teacherLLMModelBare" . -}}
{{- end -}}
{{- end -}}

{{/* The teacher backend Envoy serves pro-reasoning from. */}}
{{- define "cogniverse.srTeacherHost" -}}
{{- include "cogniverse.srEndpointHost" (include "cogniverse.srProEndpoint" .) -}}
{{- end -}}

{{- define "cogniverse.srTeacherPort" -}}
{{- include "cogniverse.srEndpointPort" (include "cogniverse.srProEndpoint" .) -}}
{{- end -}}

{{- define "cogniverse.srTeacherProtocol" -}}
{{- include "cogniverse.srEndpointProtocol" (include "cogniverse.srProEndpoint" .) -}}
{{- end -}}

{{/*
Return the proper Docker Image Registry Secret Names
*/}}
{{- define "cogniverse.imagePullSecrets" -}}
{{- $llmImage := dict -}}
{{- if eq (.Values.llm.engine | default "ollama") "vllm" -}}
{{- $llmImage = .Values.llm.vllm.image -}}
{{- else if eq (.Values.llm.engine | default "ollama") "ollama" -}}
{{- $llmImage = .Values.llm.ollama.image -}}
{{- end -}}
{{- include "common.images.pullSecrets" (dict "images" (list .Values.image .Values.vespa.image .Values.phoenix.image $llmImage) "global" .Values.global) -}}
{{- end -}}

{{/*
Compile all warnings into a single message.
*/}}
{{- define "cogniverse.validateValues" -}}
{{- $messages := list -}}
{{- $messages := append $messages (include "cogniverse.validateValues.ingress" .) -}}
{{- $messages := without $messages "" -}}
{{- $message := join "\n" $messages -}}

{{- if $message -}}
{{-   printf "\nVALUES VALIDATION:\n%s" $message -}}
{{- end -}}
{{- end -}}

{{/*
Validate ingress configuration
*/}}
{{- define "cogniverse.validateValues.ingress" -}}
{{- if and .Values.ingress.enabled (not .Values.ingress.className) -}}
cogniverse: ingress.className
    You must specify an ingress class name when ingress is enabled
{{- end -}}
{{- end -}}

{{/*
PyTorch TunableOp env for ROCm inference pods. On gfx1151 the default
hipBLASLt kernel heuristic mistunes many GEMM shapes; TunableOp benchmarks
candidates once per shape and reuses the winner. The results file lives in
the persistent model-cache mount (per-service name avoids collisions when
services share one hostPath cache; %d is the device id) so tuning survives
restarts and rollouts. Call with (dict "device" $device "name" $name "root" $).

Services on the pylate engine are excluded: their server faults the GPU
(exit 139) while TunableOp benchmarks an untuned GEMM shape, and real
traffic supplies a new shape on nearly every request. An individual
service overrides the engine-derived default with `tunableOp`.
*/}}
{{- define "cogniverse.tunableOpEnv" -}}
{{- $svc := (index .root.Values.inference .name) | default dict -}}
{{- $enabled := ne ($svc.engine | default "") "pylate" -}}
{{- if hasKey $svc "tunableOp" -}}{{- $enabled = $svc.tunableOp -}}{{- end -}}
{{- if and (eq .device "rocm") .root.Values.runtime.tunableOp $enabled }}
- name: PYTORCH_TUNABLEOP_ENABLED
  value: "1"
- name: PYTORCH_TUNABLEOP_TUNING
  value: "1"
- name: PYTORCH_TUNABLEOP_FILENAME
  value: /root/.cache/huggingface/tunableop_{{ .name | kebabcase }}_%d.csv
{{- end }}
{{- end -}}

{{/*
INFERENCE_SERVICE_URLS JSON body — one {service_key: url} entry per enabled
inference service. A non-empty ``externalUrl`` (an absolute https URL, e.g. a
Modal endpoint service root) replaces the cluster-internal Service URL. The
LLM services use their own helpers: the student is overridden via
``runtime.primaryLLM``, while the teacher keeps its key in this map and can
point at Modal.
*/}}
{{- define "cogniverse.inferenceServiceUrls" -}}
{{- $fullName := include "cogniverse.fullname" . -}}
{{- range $name, $cfg := .Values.inference -}}
{{- if and $cfg.externalUrl (eq $name "vllm_llm_student") -}}
{{- fail (printf "inference.%s.externalUrl is unsupported: the LLM endpoint is derived by runtime.primaryLLM instead" $name) -}}
{{- end -}}
{{- if $cfg.externalUrl -}}
{{- if or (hasSuffix "/v1" $cfg.externalUrl) (hasSuffix "/" $cfg.externalUrl) -}}
{{- fail (printf "inference.%s.externalUrl must be the service root URL (no trailing / or /v1)" $name) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{
{{- $first := true -}}
{{- range $name, $cfg := .Values.inference -}}
{{- if or $cfg.enabled $cfg.externalUrl }}
{{- if not $first }},{{ end -}}
{{- $first = false }}
  "{{ $name }}": "{{ if $cfg.externalUrl }}{{ $cfg.externalUrl }}{{ else }}http://{{ $fullName }}-{{ $name | kebabcase }}:{{ $cfg.service.port }}{{ end }}"
{{- end -}}
{{- end -}}
}
{{- end -}}

{{/*
Fails the render when config.defaultProfiles.video selects a profile this
composition cannot serve. Reads the profile from the rendered config.json:
every inference_services key it binds needs an enabled service or an
externalUrl, and a description vlm_endpoint naming the in-cluster student
Service, in any spelling (cogniverse.clusterEndpointKey), needs
inference.vllm_llm_student enabled.
*/}}
{{- define "cogniverse.validateDefaultVideoProfile" -}}
{{- $name := .Values.config.defaultProfiles.video -}}
{{- if $name -}}
{{- $config := tpl (.Files.Get "files/config.json") . | fromJson -}}
{{- $profile := index $config.backend.profiles $name -}}
{{- if not $profile -}}
{{- fail (printf "config.defaultProfiles.video=%s is not a profile in backend.profiles" $name) -}}
{{- end -}}
{{- range $role, $key := $profile.inference_services -}}
{{- if not (hasKey $.Values.inference $key) -}}
{{- fail (printf "config.defaultProfiles.video=%s binds inference_services.%s to %s, which is not a service under inference" $name $role $key) -}}
{{- end -}}
{{- $svc := index $.Values.inference $key -}}
{{- if not (or $svc.enabled $svc.externalUrl) -}}
{{- fail (printf "config.defaultProfiles.video=%s binds inference_services.%s to %s: set inference.%s.enabled=true or inference.%s.externalUrl" $name $role $key $key $key) -}}
{{- end -}}
{{- end -}}
{{- $vlmEndpoint := dig "strategies" "description" "params" "vlm_endpoint" "" $profile -}}
{{- $vlmKey := include "cogniverse.clusterEndpointKey" (dict "url" $vlmEndpoint "root" .) -}}
{{- $studentKey := include "cogniverse.clusterEndpointKey" (dict "url" (include "cogniverse.llmStudentServiceUrl" .) "root" .) -}}
{{- if and $vlmEndpoint (eq $vlmKey $studentKey) (not .Values.inference.vllm_llm_student.enabled) -}}
{{- fail (printf "config.defaultProfiles.video=%s describes frames with VLMDescriptionStrategy at %s, the in-cluster vllm_llm_student Service: set inference.vllm_llm_student.enabled=true or point runtime.primaryLLM.apiBase at a served endpoint" $name $vlmEndpoint) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
REDIS_URL for containers running cogniverse application code.

cogniverse_dashboard.tabs.approval_queue raises when it is absent, which
aborts a Streamlit render partway through and silently drops every widget
after that point. Defined once so a new deployment cannot pick up a stale
copy of the auth branch.
*/}}
{{- define "cogniverse.redisUrlEnv" -}}
{{- $fullName := include "cogniverse.fullname" . -}}
{{- if .Values.redis.enabled }}
{{- if .Values.redis.auth.enabled }}
- name: REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ $fullName }}-redis-auth
      key: redis-password
- name: REDIS_URL
  value: "redis://:$(REDIS_PASSWORD)@{{ $fullName }}-redis:{{ .Values.redis.service.port }}/0"
{{- else }}
- name: REDIS_URL
  value: "redis://{{ $fullName }}-redis:{{ .Values.redis.service.port }}/0"
{{- end }}
{{- else }}
{{- fail "redis.enabled=false is unsupported: the runtime requires REDIS_URL, which the chart sets only from the in-cluster Redis" }}
{{- end }}
{{- end }}

{{/*
Public URL prefix the ingress publishes the runtime under.

The runtime is reached two ways: in-cluster on its Service, where the path
carries no prefix, and through the ingress, where every path starts with
this prefix. FastAPI strips a ``root_path`` that a request actually carries
and leaves every other path alone, so one value serves both. It is read
from the ingress rules rather than configured twice, and publishing the
runtime under two different prefixes has no single answer, so it fails.
*/}}
{{- define "cogniverse.apiPrefix" -}}
{{- $prefixes := list -}}
{{- if .Values.ingress.enabled -}}
{{- range .Values.ingress.hosts -}}
{{- range .paths -}}
{{- if eq .service "runtime" -}}
{{- $prefixes = append $prefixes (trimSuffix "/" .path) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- $distinct := uniq $prefixes -}}
{{- if gt (len $distinct) 1 -}}
{{- fail (printf "runtime ingress paths must share one prefix, got %v" $distinct) -}}
{{- end -}}
{{- first $distinct | default "" -}}
{{- end -}}
