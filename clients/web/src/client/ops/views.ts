import type { ComponentType } from 'react';
import { AnnotationsView } from './AnnotationsView';
import { ApprovalsView } from './ApprovalsView';
import { IngestionView } from './IngestionView';
import { MemoryView } from './MemoryView';
import { OptimizationView } from './OptimizationView';
import { ProfilesView } from './ProfilesView';
import { TenantsView } from './TenantsView';

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
];
