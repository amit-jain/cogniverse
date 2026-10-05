declare module 'plotly.js-dist-min' {
  /** The part of Plotly's API the client uses. */
  export function react(
    root: HTMLElement,
    data: Record<string, unknown>[],
    layout: Record<string, unknown>,
    config?: Record<string, unknown>,
  ): Promise<HTMLElement>;
  export function purge(root: HTMLElement): void;
}
