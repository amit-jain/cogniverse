/**
 * A random RFC 4122 version 4 id. ``crypto.randomUUID`` exists only in a
 * secure context (https or localhost), and the client is also served over
 * plain http through the cluster ingress, so the id is built from
 * ``crypto.getRandomValues``, which every context has.
 */
export function newId(random: (bytes: Uint8Array) => Uint8Array = (bytes) => crypto.getRandomValues(bytes)): string {
  const bytes = random(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}
