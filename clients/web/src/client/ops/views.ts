import type { ComponentType } from 'react';
import { AnalyticsView } from './AnalyticsView';
import { AnnotationsView } from './AnnotationsView';
import { ApprovalsView } from './ApprovalsView';
import { IngestionView } from './IngestionView';
import { MemoryView } from './MemoryView';
import { OptimizationView } from './OptimizationView';
import { ProfileMetricsView } from './ProfileMetricsView';
import { ProfilesView } from './ProfilesView';
import { RlmAbView } from './RlmAbView';
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
  { id: 'ingestion', label: 'Ingestion', component: IngestionView },
  { id: 'optimization', label: 'Optimization runs', component: OptimizationView },
  { id: 'memory', label: 'Memory', component: MemoryView },
  { id: 'approvals', label: 'Approvals', component: ApprovalsView },
  { id: 'annotations', label: 'Annotation queue', component: AnnotationsView },
  { id: 'workflows', label: 'Workflow reviews', component: WorkflowReviewsView },
  { id: 'profile-metrics', label: 'Profile metrics', component: ProfileMetricsView },
  { id: 'rlm-ab', label: 'RLM A/B', component: RlmAbView },
  { id: 'analytics', label: 'Analytics', component: AnalyticsView },
];
