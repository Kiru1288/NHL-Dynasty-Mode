export const UI_SCALE_STORAGE_KEY = "nhl.uiDensity";
export const UI_SCALE_CHANGE_EVENT = "nhl-ui-scale-change";

export const UI_SCALE_PRESETS = [
  { id: "auto", label: "Match display" },
  { id: "compact", label: "Laptop" },
  { id: "standard", label: "1080p" },
  { id: "wide", label: "1440p / 4K" },
];

const LEGACY_ZOOM_PREFS = new Set(["0.75", "0.85", "0.9", "1", "1.25", "1.5"]);

function viewportSize() {
  return {
    w: Math.max(320, window.innerWidth || 1920),
    h: Math.max(320, window.innerHeight || 1080),
  };
}

/** Map a real window to a display class — not a zoom factor. */
export function resolveDisplayBand(width, height) {
  const w = Number(width) || 0;
  const h = Number(height) || 0;
  if (w >= 3200 || h >= 1800) return "uhd";
  if (w >= 2200 || h >= 1300) return "qhd";
  if (w >= 1600 || h >= 900) return "fhd";
  if (w >= 1280 || h >= 720) return "hd";
  return "compact";
}

export function readUiScalePreference() {
  try {
    const raw = String(window.localStorage.getItem(UI_SCALE_STORAGE_KEY) || window.localStorage.getItem("nhl.uiScale") || "auto");
    if (LEGACY_ZOOM_PREFS.has(raw)) return "auto";
    return raw;
  } catch {
    return "auto";
  }
}

export function writeUiScalePreference(value) {
  try {
    window.localStorage.setItem(UI_SCALE_STORAGE_KEY, String(value));
    window.localStorage.removeItem("nhl.uiScale");
  } catch {
    /* ignore quota */
  }
  window.dispatchEvent(new CustomEvent(UI_SCALE_CHANGE_EVENT, { detail: value }));
}

function bandForPreference(width, height, preference) {
  if (preference === "compact") return width < 1280 ? "compact" : "hd";
  if (preference === "standard") return "fhd";
  if (preference === "wide") return width >= 3200 ? "uhd" : "qhd";
  return resolveDisplayBand(width, height);
}

function clearFakeZoom(root) {
  if (!root) return;
  root.style.removeProperty("width");
  root.style.removeProperty("height");
  root.style.removeProperty("zoom");
  root.style.removeProperty("transform");
  root.style.removeProperty("transform-origin");
}

export function uiPortalTarget() {
  return document.getElementById("root") || document.body;
}

export function applyFluidUiScale() {
  if (typeof document === "undefined") return "fhd";
  const { w, h } = viewportSize();
  const preference = readUiScalePreference();
  const band = bandForPreference(w, h, preference);
  const html = document.documentElement;
  const root = document.getElementById("root");
  html.classList.add("ui-fluid");
  html.dataset.display = band;
  html.dataset.displayPref = preference;
  html.style.setProperty("--ui-vw", `${w}px`);
  html.style.setProperty("--ui-vh", `${h}px`);
  html.style.setProperty("--ui-aspect", String(Math.round((w / h) * 1000) / 1000));
  html.style.removeProperty("--ui-scale");
  html.style.removeProperty("zoom");
  clearFakeZoom(root);
  return band;
}
