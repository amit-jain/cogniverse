import { useState } from 'react';
import { Alert } from './common';

/** ``text`` as a lookback in hours within ``[min, max]``, or an error. */
export function parseHours(text: string, min: number, max: number): { hours: number } | { error: string } {
  const hours = Number(text.trim());
  if (!text.trim() || !Number.isFinite(hours) || hours < min || hours > max)
    return { error: `Lookback must be a number of hours from ${min} to ${max}.` };
  return { hours };
}

/** A lookback in hours, typed and applied, within ``[min, max]``. */
export function HoursInput({
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  value: number;
  onChange: (hours: number) => void;
  min: number;
  max: number;
  step?: number;
}) {
  const [text, setText] = useState(String(value));
  const [error, setError] = useState('');
  return (
    <form
      className="inline-form"
      aria-label="Lookback"
      noValidate
      onSubmit={(e) => {
        e.preventDefault();
        const parsed = parseHours(text, min, max);
        if ('error' in parsed) return setError(parsed.error);
        setError('');
        onChange(parsed.hours);
      }}
    >
      <label>
        Lookback (hours)
        <input type="number" min={min} max={max} step={step} value={text} onChange={(e) => setText(e.target.value)} />
      </label>
      <button type="submit">Apply</button>
      {error && <Alert>{error}</Alert>}
    </form>
  );
}
