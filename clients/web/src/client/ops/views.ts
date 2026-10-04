import type { ComponentType } from 'react';
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
];
