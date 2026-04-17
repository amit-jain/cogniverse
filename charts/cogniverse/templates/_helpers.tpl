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
LLM endpoint URL. Prefers llm.external.url when external is enabled,
otherwise derives the in-cluster service URL from the builtin LLM.
Used by the config.json ConfigMap so DSPy/Ollama clients hit the
deployed endpoint rather than the localhost default baked into config.json.
*/}}
{{- define "cogniverse.llmEndpoint" -}}
{{- if .Values.llm.external.enabled -}}
{{- required "llm.external.url is required when llm.external.enabled=true" .Values.llm.external.url -}}
{{- else -}}
{{- printf "http://%s-llm:%d" (include "cogniverse.fullname" .) (int .Values.llm.builtin.service.port) -}}
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
{{- include "common.images.pullSecrets" (dict "images" (list .Values.image .Values.vespa.image .Values.phoenix.image .Values.llm.builtin.image) "global" .Values.global) -}}
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
