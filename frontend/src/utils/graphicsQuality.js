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
let gpuTierCache = null;

/**
 * "integrated" | "software" | "discrete" | "unknown", from the WebGL renderer string.
 * Integrated laptop GPUs (Intel HD/UHD/Iris, AMD Vega/Radeon Graphics APUs, mobile chips)
 * have plenty of CPU cores, so core count alone called them high-end and the 3D hub
 * started on shadows + bloom at 1.5x resolution.
 */
export function detectGpuTier() {
  if (gpuTierCache) return gpuTierCache;
  gpuTierCache = "unknown";
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!gl) return gpuTierCache;
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const name = String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER) || "").toLowerCase();
    if (/swiftshader|llvmpipe|softpipe|software|microsoft basic render/.test(name)) gpuTierCache = "software";
    else if (/intel|iris|uhd|hd graphics|mali|adreno|powervr|apple gpu|radeon\(tm\) graphics|radeon graphics|vega \d+ graphics|mesa/.test(name)) gpuTierCache = "integrated";
    else if (/nvidia|geforce|rtx|gtx|quadro|radeon|amd|apple m\d/.test(name)) gpuTierCache = "discrete";
    const lose = gl.getExtension("WEBGL_lose_context");
    if (lose) lose.loseContext();
  } catch {
    /* keep unknown */
  }
  return gpuTierCache;
}

export function isLikelyLowEndDevice() {
  if (typeof navigator === "undefined") return false;
  const cores = Number(navigator.hardwareConcurrency || 0);
  const memGb = Number(navigator.deviceMemory || 0);
  const gpu = detectGpuTier();
  return gpu === "integrated" || gpu === "software" || (cores > 0 && cores <= 4) || (memGb > 0 && memGb <= 4);
}
