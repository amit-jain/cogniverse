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
LLM endpoint URL. Resolves to the right host:port based on llm.engine:
  - external → llm.external.url (required)
  - vllm     → in-cluster service on llm.vllm.service.port
  - ollama   → in-cluster service on llm.ollama.service.port
Single source of truth for every consumer (runtime, agents, init jobs,
optimization workflows) so the URL never drifts between sites.
*/}}
{{- define "cogniverse.llmEndpoint" -}}
{{- $engine := .Values.llm.engine | default "ollama" -}}
{{- if eq $engine "external" -}}
{{- required "llm.external.url is required when llm.engine=external" .Values.llm.external.url -}}
{{- else if eq $engine "vllm" -}}
{{- printf "http://%s-llm:%d" (include "cogniverse.fullname" .) (int .Values.llm.vllm.service.port) -}}
{{- else if eq $engine "ollama" -}}
{{- printf "http://%s-llm:%d" (include "cogniverse.fullname" .) (int .Values.llm.ollama.service.port) -}}
{{- else -}}
{{- fail (printf "llm.engine must be one of [ollama, vllm, external], got %q" $engine) -}}
{{- end -}}
{{- end -}}

{{/*
DSPy / litellm prefix for ``llm.model`` based on engine. The runtime joins
this with llm.model to construct the dspy.LM model string. ollama needs
the ``ollama_chat/`` prefix specifically (not ``ollama/`` — that targets
the legacy non-chat completion API).
*/}}
{{- define "cogniverse.llmDspyPrefix" -}}
{{- $engine := .Values.llm.engine | default "ollama" -}}
{{- if eq $engine "vllm" -}}hosted_vllm
{{- else if eq $engine "external" -}}openai
{{- else -}}ollama_chat
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
vLLM student LLM endpoint — runtime hot path (low-latency Gemma 4 E4B).
Resolves to the in-cluster vllm-llm-student service. Always /v1 suffix
because litellm hosted_vllm provider expects an OpenAI-compatible base.
*/}}
{{- define "cogniverse.llmStudentEndpoint" -}}
{{- printf "http://%s-vllm-llm-student:%d/v1" (include "cogniverse.fullname" .) (int .Values.inference.vllm_llm_student.service.port) -}}
{{- end -}}

{{/*
vLLM teacher LLM endpoint — DSPy compile-time only (Gemma 4 26B-A4B).
Resolves to the in-cluster vllm-llm-teacher service. Scale-to-zero by
default (replicaCount: 0); spun up on-demand by optimization_cli runs.
*/}}
{{- define "cogniverse.llmTeacherEndpoint" -}}
{{- printf "http://%s-vllm-llm-teacher:%d/v1" (include "cogniverse.fullname" .) (int .Values.inference.vllm_llm_teacher.service.port) -}}
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
