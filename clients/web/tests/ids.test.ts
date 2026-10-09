import { afterEach, describe, expect, it, vi } from 'vitest';
import { newId } from '../src/client/ids';

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

describe('newId', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('builds a version 4 id without crypto.randomUUID, as outside a secure context', () => {
    const real = globalThis.crypto;
    vi.stubGlobal('crypto', { getRandomValues: (bytes: Uint8Array) => real.getRandomValues(bytes) });
    expect(crypto.randomUUID).toBeUndefined();
    const ids = Array.from({ length: 1000 }, () => newId());
    for (const id of ids) expect(id).toMatch(UUID_V4);
    expect(new Set(ids).size).toBe(1000);
  });

  it('sets the version and variant bits of the random bytes', () => {
    expect(newId((bytes) => bytes.fill(0xff))).toBe('ffffffff-ffff-4fff-bfff-ffffffffffff');
    expect(newId((bytes) => bytes.fill(0))).toBe('00000000-0000-4000-8000-000000000000');
  });
});
