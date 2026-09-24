/**
 * One chemistry scale for every screen (Line Builder, PP/PK, Chemistry report).
 * Thresholds and labels mirror SimEngine `chemistry_label()` so the words a
 * screen shows always match what the backend reports.
 */

export const CHEMISTRY_TIERS = Object.freeze([
  { min: 90, label: "Elite", tone: "high" },
  { min: 75, label: "Strong", tone: "high" },
  { min: 60, label: "Connected", tone: "mid" },
  { min: 45, label: "Neutral", tone: "mid" },
  { min: 30, label: "Awkward", tone: "low" },
  { min: -Infinity, label: "Broken", tone: "low" },
]);

/** Legend bands — three colours, same cut points as the tiers above. */
export const CHEMISTRY_BANDS = Object.freeze([
  { tone: "high", label: "High · 75+ Strong / Elite" },
  { tone: "mid", label: "Medium · 45–74 Neutral / Connected" },
  { tone: "low", label: "Low · below 45 Awkward / Broken" },
]);

export const CHEMISTRY_HIGH = 75;
export const CHEMISTRY_MID = 45;

function tierFor(score) {
  const s = Number(score);
  if (!Number.isFinite(s)) return null;
  return CHEMISTRY_TIERS.find((tier) => s >= tier.min) || null;
}

export function chemistryLabel(score) {
  return tierFor(score)?.label || "—";
}

/** "high" | "mid" | "low" | "" (no score). */
export function chemistryTone(score) {
  return tierFor(score)?.tone || "";
}
