import type { ComponentType } from 'react';
import { TenantsView } from './TenantsView';

export interface OpsView {
  id: string;
  label: string;
  component: ComponentType;
}

export const OPS_VIEWS: OpsView[] = [{ id: 'tenants', label: 'Tenants', component: TenantsView }];
