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
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
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
vLLM student LLM endpoint — runtime hot path. Resolves to the
in-cluster vllm-llm-student service with the ``/v1`` suffix so the
generic OpenAI-compatible client can hit /v1/chat/completions.
*/}}
{{- define "cogniverse.llmStudentEndpoint" -}}
{{- printf "http://%s-vllm-llm-student:%d/v1" (include "cogniverse.fullname" .) (int .Values.inference.vllm_llm_student.service.port) -}}
{{- end -}}

{{/*
vLLM teacher LLM endpoint — DSPy compile-time only. Resolves to the
in-cluster vllm-llm-teacher service. Scale-to-zero by default
(replicaCount: 0); spun up on-demand by optimization_cli runs.
*/}}
{{- define "cogniverse.llmTeacherEndpoint" -}}
{{- printf "http://%s-vllm-llm-teacher:%d/v1" (include "cogniverse.fullname" .) (int .Values.inference.vllm_llm_teacher.service.port) -}}
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
*/}}
{{- define "cogniverse.tunableOpEnv" -}}
{{- if and (eq .device "rocm") .root.Values.runtime.tunableOp }}
- name: PYTORCH_TUNABLEOP_ENABLED
  value: "1"
- name: PYTORCH_TUNABLEOP_TUNING
  value: "1"
- name: PYTORCH_TUNABLEOP_FILENAME
  value: /root/.cache/huggingface/tunableop_{{ .name | kebabcase }}_%d.csv
{{- end }}
{{- end -}}
