export const GRAPHICS_QUALITY_KEY = "nhl.graphicsQuality";
export const GRAPHICS_QUALITY_CHANGE_EVENT = "nhl-graphics-quality-change";
const LEGACY_LOW_POWER_KEY = "nhlOfficeLowPowerMode";

export const GRAPHICS_QUALITY_PRESETS = [
  { id: "auto", label: "Auto" },
  { id: "high", label: "High" },
  { id: "low", label: "Low" },
];

export function readGraphicsQuality() {
  try {
    const value = window.localStorage.getItem(GRAPHICS_QUALITY_KEY);
    if (value === "auto" || value === "high" || value === "low") return value;
    if (window.localStorage.getItem(LEGACY_LOW_POWER_KEY) === "1") return "low";
  } catch {
    /* storage unavailable */
  }
  return "auto";
}

/** App-wide lite styling (styles/perf-lite.css). In Auto the office hub decides live. */
export function applyGraphicsQualityClass(mode = readGraphicsQuality()) {
  if (typeof document === "undefined") return;
  if (mode === "low") document.documentElement.classList.add("perf-lite");
  else if (mode === "high") document.documentElement.classList.remove("perf-lite");
}

export function writeGraphicsQuality(value) {
  try {
    window.localStorage.setItem(GRAPHICS_QUALITY_KEY, String(value));
    window.localStorage.removeItem(LEGACY_LOW_POWER_KEY);
  } catch {
    /* ignore quota */
  }
  applyGraphicsQualityClass(value);
  window.dispatchEvent(new CustomEvent(GRAPHICS_QUALITY_CHANGE_EVENT, { detail: value }));
}

/** Starting guess for Auto before any frames have been measured. */
export function isLikelyLowEndDevice() {
  if (typeof navigator === "undefined") return false;
  const cores = Number(navigator.hardwareConcurrency || 0);
  const memGb = Number(navigator.deviceMemory || 0);
  return (cores > 0 && cores <= 4) || (memGb > 0 && memGb <= 4);
}
