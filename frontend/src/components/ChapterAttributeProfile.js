import React from "react";
import { chapterAttributeRows, chapterNumericValue } from "../utils/chapterAttributes";

function toneForValue(value) {
  const n = chapterNumericValue(value);
  if (n == null) return "neutral";
  if (n >= 84) return "good";
  if (n >= 72) return "neutral";
  return "warn";
}

function formatValue(value) {
  if (value == null) return "—";
  if (typeof value === "object" && value.band) {
    return `${value.lo}–${value.high ?? value.hi}`;
  }
  const n = chapterNumericValue(value);
  return n != null ? String(n) : "—";
}

export default function ChapterAttributeProfile({ player, compact = false, className = "" }) {
  const rows = chapterAttributeRows(player);
  if (!rows.length) {
    return <p className="chapter-attribute-profile__empty">Chapter ratings are not available for this player.</p>;
  }

  return (
    <div className={`chapter-attribute-profile ${compact ? "is-compact" : ""} ${className}`.trim()}>
      {rows.map(([label, value]) => {
        const numeric = chapterNumericValue(value);
        const pct = numeric != null ? Math.max(0, Math.min(100, numeric)) : 0;
        return (
          <div key={label} className="chapter-attribute-profile__row">
            <div className="chapter-attribute-profile__meta">
              <span>{label}</span>
              <strong>{formatValue(value)}</strong>
            </div>
            <div className="chapter-attribute-profile__track" aria-hidden="true">
              <i className={`is-${toneForValue(value)}`} style={{ width: `${pct}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
