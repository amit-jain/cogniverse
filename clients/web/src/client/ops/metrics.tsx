/** The windows the telemetry views read, in hours. */
export const LOOKBACKS = [
  { hours: 1, label: 'Last hour' },
  { hours: 6, label: 'Last 6 hours' },
  { hours: 24, label: 'Last day' },
  { hours: 168, label: 'Last week' },
];

export function LookbackSelect({
  value,
  onChange,
  options = LOOKBACKS,
}: {
  value: number;
  onChange: (hours: number) => void;
  options?: { hours: number; label: string }[];
}) {
  return (
    <label>
      Window
      <select value={value} onChange={(e) => onChange(Number(e.target.value))}>
        {options.map((option) => (
          <option key={option.hours} value={option.hours}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

/** 0.4567 -> "45.7%". */
export function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** A signed delta to one decimal, or "—" when it was not measured. */
export function delta(value: number | null, digits = 1): string {
  if (value === null) return '—';
  const text = value.toFixed(digits);
  return value > 0 ? `+${text}` : text;
}

/** One horizontal bar per entry, each as wide as its share of the largest. */
export function Bars({
  title,
  entries,
  format,
}: {
  title: string;
  entries: { label: string; value: number }[];
  format: (value: number) => string;
}) {
  const largest = Math.max(0, ...entries.map((entry) => Math.abs(entry.value)));
  return (
    <figure className="bars" aria-label={title}>
      <figcaption>{title}</figcaption>
      {entries.map((entry) => (
        <div key={entry.label} className="bar-row">
          <span className="bar-label">{entry.label}</span>
          <span className="bar-track">
            <span
              className="bar-fill"
              style={{ width: `${largest ? (Math.abs(entry.value) / largest) * 100 : 0}%` }}
            />
          </span>
          <span className="bar-value">{format(entry.value)}</span>
        </div>
      ))}
    </figure>
  );
}
