import type { ComponentType } from 'react';
import { AnalyticsView } from './AnalyticsView';
import { AnnotationsView } from './AnnotationsView';
import { ConfigView } from './ConfigView';
import { EmbeddingAtlasView } from './EmbeddingAtlasView';
import { ApprovalsView } from './ApprovalsView';
import { EvaluationView } from './EvaluationView';
import { IngestionView } from './IngestionView';
import { MemoryView } from './MemoryView';
import { OptimizationFrameworkView } from './OptimizationFrameworkView';
import { OptimizationView } from './OptimizationView';
import { ProfileMetricsView } from './ProfileMetricsView';
import { ProfilesView } from './ProfilesView';
import { RlmAbView } from './RlmAbView';
import { RoutingView } from './RoutingView';
import { TenantsView } from './TenantsView';
import { WorkflowReviewsView } from './WorkflowReviewsView';

export interface OpsView {
  id: string;
  label: string;
  component: ComponentType;
}

export const OPS_VIEWS: OpsView[] = [
  { id: 'tenants', label: 'Tenants', component: TenantsView },
  { id: 'profiles', label: 'Backend profiles', component: ProfilesView },
  { id: 'config', label: 'Configuration', component: ConfigView },
  { id: 'ingestion', label: 'Ingestion', component: IngestionView },
  { id: 'optimization', label: 'Optimization runs', component: OptimizationView },
  { id: 'optimization-framework', label: 'Optimization framework', component: OptimizationFrameworkView },
  { id: 'memory', label: 'Memory', component: MemoryView },
  { id: 'approvals', label: 'Approvals', component: ApprovalsView },
  { id: 'annotations', label: 'Annotation queue', component: AnnotationsView },
  { id: 'workflows', label: 'Workflow reviews', component: WorkflowReviewsView },
  { id: 'profile-metrics', label: 'Profile metrics', component: ProfileMetricsView },
  { id: 'rlm-ab', label: 'RLM A/B', component: RlmAbView },
  { id: 'analytics', label: 'Analytics', component: AnalyticsView },
  { id: 'evaluation', label: 'Evaluation', component: EvaluationView },
  { id: 'atlas', label: 'Embedding atlas', component: EmbeddingAtlasView },
  { id: 'routing', label: 'Routing evaluation', component: RoutingView },
];
