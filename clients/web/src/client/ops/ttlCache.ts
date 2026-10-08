import { useCallback, useRef } from 'react';
import { useLoad } from './common';
import { RuntimeRequestError, runtimeJson } from './http';

/** Answers kept for a while, so going back to a view does not read again. */
export class TtlCache<T> {
  private readonly entries = new Map<string, { at: number; value: Promise<T> }>();

  constructor(
    private readonly ttlMs: number,
    private readonly now: () => number = Date.now,
  ) {}

  /** The answer ``load`` gave for ``key`` within the last ``ttlMs``, or a
   * fresh one when there is none or ``fresh`` is set. A failed load is not
   * kept. */
  get(key: string, load: () => Promise<T>, fresh = false): Promise<T> {
    const entry = this.entries.get(key);
    if (!fresh && entry && this.now() - entry.at < this.ttlMs) return entry.value;
    const value = load();
    this.entries.set(key, { at: this.now(), value });
    value.catch(() => {
      if (this.entries.get(key)?.value === value) this.entries.delete(key);
    });
    return value;
  }
}

/** ``useLoad`` over a cached runtime read of ``path``; ``refresh`` reads it
 * again past the cache, and ``errorStatus`` is the HTTP status of a failed
 * read. */
export function useCachedJson<T>(cache: TtlCache<unknown>, path: string) {
  const fresh = useRef(false);
  const status = useRef<number | undefined>(undefined);
  const loaded = useLoad(() => {
    const force = fresh.current;
    fresh.current = false;
    status.current = undefined;
    return (cache.get(path, () => runtimeJson<T>(path), force) as Promise<T>).catch((error: unknown) => {
      status.current = error instanceof RuntimeRequestError ? error.status : undefined;
      throw error;
    });
  }, [path]);
  const reload = loaded.reload;
  const refresh = useCallback(() => {
    fresh.current = true;
    reload();
  }, [reload]);
  return { ...loaded, refresh, errorStatus: loaded.error ? status.current : undefined };
}
