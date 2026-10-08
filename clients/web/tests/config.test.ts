import { describe, expect, it } from 'vitest';
import { exportPreview } from '../src/client/ops/ConfigView';

describe('exportPreview', () => {
  it('lists each exported config with the tenant it came from', () => {
    const text = JSON.stringify({
      tenant_id: 'acme:production',
      include_history: false,
      configs: [
        { scope: 'backend', service: 'backend', config_key: 'backend_config', version: 4, config_value: {} },
        { scope: 'routing', service: 'gateway_agent', config_key: 'routing_config', version: 2, config_value: {} },
      ],
    });
    expect(exportPreview('export.json', text)).toEqual({
      from: 'acme:production',
      configs: [
        { scope: 'backend', service: 'backend', config_key: 'backend_config', version: 4 },
        { scope: 'routing', service: 'gateway_agent', config_key: 'routing_config', version: 2 },
      ],
    });
  });

  it('names a file that is not JSON or not an export', () => {
    expect(() => exportPreview('bad.json', '{"configs": [')).toThrow('bad.json is not valid JSON.');
    expect(() => exportPreview('list.json', '[1, 2]')).toThrow(
      'list.json is not a configuration export: it has no configs list.',
    );
  });
});
