import { useId, useState, type ReactNode } from 'react';

/** Tabs that show one panel at a time, the first selected. */
export function Tabs({ label, tabs }: { label: string; tabs: { name: string; panel: () => ReactNode }[] }) {
  const [selected, setSelected] = useState(0);
  const id = useId();
  const shown = Math.min(selected, tabs.length - 1);
  return (
    <div className="tabs">
      <div role="tablist" aria-label={label} className="section-tabs">
        {tabs.map((tab, index) => (
          <button
            key={tab.name}
            role="tab"
            id={`${id}-tab-${index}`}
            aria-selected={index === shown}
            aria-pressed={index === shown}
            aria-controls={`${id}-panel`}
            onClick={() => setSelected(index)}
          >
            {tab.name}
          </button>
        ))}
      </div>
      {tabs.length > 0 && (
        <div role="tabpanel" id={`${id}-panel`} aria-labelledby={`${id}-tab-${shown}`}>
          {tabs[shown].panel()}
        </div>
      )}
    </div>
  );
}
