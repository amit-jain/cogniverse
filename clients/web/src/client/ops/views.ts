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
  /** One sentence shown under the view's heading. */
  description?: string;
  component: ComponentType;
}

export const OPS_VIEWS: OpsView[] = [
  {
    id: 'tenants',
    label: 'Tenants',
    description: "Create and delete organizations and their tenants, and set each tenant's router tier.",
    component: TenantsView,
  },
  {
    id: 'profiles',
    label: 'Backend profiles',
    description: "Create, edit and delete a tenant's backend profiles and deploy their schemas.",
    component: ProfilesView,
  },
  {
    id: 'config',
    label: 'Configuration',
    description: "Edit the system's and each tenant's configuration, browse its history, and export or import it.",
    component: ConfigView,
  },
  {
    id: 'ingestion',
    label: 'Ingestion',
    description: 'Interactive testing and configuration of ingestion pipelines with different processing profiles.',
    component: IngestionView,
  },
  {
    id: 'optimization',
    label: 'Optimization runs',
    description: 'Upload training examples, run optimizations of routing, search and agent modules, and read their reports.',
    component: OptimizationView,
  },
  {
    id: 'optimization-framework',
    label: 'Optimization framework',
    description: 'Optimization of routing, search quality and agent performance: annotate searches, build datasets and train the profile recommender.',
    component: OptimizationFrameworkView,
  },
  {
    id: 'memory',
    label: 'Memory',
    description: 'Search, add, pin and delete the memories each agent keeps for a tenant.',
    component: MemoryView,
  },
  {
    id: 'approvals',
    label: 'Approvals',
    description: 'Review and approve generated outputs before they are used; approved and rejected items stay for reference.',
    component: ApprovalsView,
  },
  {
    id: 'annotations',
    label: 'Annotation queue',
    description: 'Assign and label the routing decisions queued for a reviewer.',
    component: AnnotationsView,
  },
  {
    id: 'workflows',
    label: 'Workflow reviews',
    description: 'Annotate orchestration workflows; the annotations become ground truth for optimizing routing and orchestration.',
    component: WorkflowReviewsView,
  },
  {
    id: 'profile-metrics',
    label: 'Profile metrics',
    description: 'Profile selections per modality: how many, how fast and how often they succeed.',
    component: ProfileMetricsView,
  },
  {
    id: 'rlm-ab',
    label: 'RLM A/B',
    description: 'RLM-on and RLM-off arms of each compared query, side by side, from cogniverse-optim --mode ab-compare.',
    component: RlmAbView,
  },
  {
    id: 'analytics',
    label: 'Analytics',
    description: "Charts, outliers and root causes of a tenant's traces, and an explorer to find one.",
    component: AnalyticsView,
  },
  {
    id: 'evaluation',
    label: 'Evaluation',
    description: "Score a tenant's searches against its golden set or a Phoenix dataset.",
    component: EvaluationView,
  },
  {
    id: 'atlas',
    label: 'Embedding atlas',
    description: "Map a tenant's documents by their embeddings and see where a query lands among them.",
    component: EmbeddingAtlasView,
  },
  {
    id: 'routing',
    label: 'Routing evaluation',
    description: 'Routing decisions with their outcomes, per-agent accuracy and calibration, and the labels reviewers give them.',
    component: RoutingView,
  },
];
