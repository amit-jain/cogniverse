export interface SseFrame {
  id?: string;
  data: string;
}

/**
 * Splits server-sent-event text into complete frames and the unfinished
 * remainder. Comment lines (heartbeats) are dropped, as are frames with no
 * data; multi-line data is joined with newlines.
 */
export function parseSse(buffer: string): { frames: SseFrame[]; rest: string } {
  const normalised = buffer.replace(/\r\n?/g, '\n');
  const blocks = normalised.split('\n\n');
  const rest = blocks.pop() ?? '';
  const frames: SseFrame[] = [];
  for (const block of blocks) {
    let id: string | undefined;
    const data: string[] = [];
    for (const line of block.split('\n')) {
      if (!line || line.startsWith(':')) continue;
      const colon = line.indexOf(':');
      const field = colon === -1 ? line : line.slice(0, colon);
      const value = colon === -1 ? '' : line.slice(colon + 1).replace(/^ /, '');
      if (field === 'data') data.push(value);
      else if (field === 'id') id = value;
    }
    if (data.length) frames.push({ ...(id === undefined ? {} : { id }), data: data.join('\n') });
  }
  return { frames, rest };
}
