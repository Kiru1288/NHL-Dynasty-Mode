/**
 * Frontend performance profiler for NHL Franchise Mode.
 *
 * Tracks navigation, API calls, and optional React marks.
 * Does not change gameplay. Toggle with localStorage nhl_perf=0 or ?perf=0.
 *
 * window.__NHL_PERF.snapshot() — console dump
 * window.__NHL_PERF.reset()
 */

const STORAGE_KEY = "nhl_perf";
const SLOW_MS = Number(localStorage.getItem("nhl_perf_slow_ms") || 100);

function _enabled() {
  try {
    const q = new URLSearchParams(window.location.search).get("perf");
    if (q === "0" || q === "false") return false;
    if (q === "1" || q === "true") return true;
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "0" || stored === "false") return false;
  } catch {
    // ignore
  }
  return true;
}

let ENABLED = typeof window !== "undefined" ? _enabled() : false;

const buckets = new Map();
const recent = [];
const RECENT_LIMIT = 200;
const startedAt = performance.now();

function ensureBucket(name) {
  let b = buckets.get(name);
  if (!b) {
    b = { count: 0, totalMs: 0, maxMs: 0, minMs: Infinity, lastMs: 0 };
    buckets.set(name, b);
  }
  return b;
}

export function record(name, durationMs, meta) {
  if (!ENABLED) return;
  const ms = Number(durationMs) || 0;
  const b = ensureBucket(name);
  b.count += 1;
  b.totalMs += ms;
  b.lastMs = ms;
  if (ms > b.maxMs) b.maxMs = ms;
  if (ms < b.minMs) b.minMs = ms;
  recent.push({ name, ms: Math.round(ms * 100) / 100, t: Date.now(), meta: meta || undefined });
  if (recent.length > RECENT_LIMIT) recent.splice(0, recent.length - RECENT_LIMIT);
  if (ms >= SLOW_MS) {
    // eslint-disable-next-line no-console
    console.info(`[perf] SLOW ${name} ${ms.toFixed(1)}ms`, meta || "");
  }
}

export function span(name, meta) {
  if (!ENABLED) {
    return { end() {} };
  }
  const t0 = performance.now();
  return {
    end(extra) {
      record(name, performance.now() - t0, { ...(meta || {}), ...(extra || {}) });
    },
  };
}

export async function measureAsync(name, fn, meta) {
  const s = span(name, meta);
  try {
    return await fn();
  } finally {
    s.end();
  }
}

export function markNavigation(fromScreen, toScreen) {
  record("ui.navigate", 0, { from: fromScreen, to: toScreen, at: performance.now() });
  const s = span(`ui.screen.${toScreen || "unknown"}`);
  // Caller should end when screen is interactive; auto-end after paint.
  requestAnimationFrame(() => {
    requestAnimationFrame(() => s.end({ from: fromScreen }));
  });
}

export function markInteraction(action, meta) {
  return span(`ui.action.${action}`, meta);
}

export function snapshot(topN = 40) {
  const rows = [...buckets.entries()].map(([name, b]) => ({
    name,
    count: b.count,
    total_ms: Math.round(b.totalMs * 100) / 100,
    avg_ms: b.count ? Math.round((b.totalMs / b.count) * 100) / 100 : 0,
    max_ms: Math.round(b.maxMs * 100) / 100,
    min_ms: b.count === 0 ? 0 : Math.round(b.minMs * 100) / 100,
    last_ms: Math.round(b.lastMs * 100) / 100,
  }));
  rows.sort((a, b) => b.total_ms - a.total_ms || b.max_ms - a.max_ms);
  return {
    ok: true,
    enabled: ENABLED,
    uptime_ms: Math.round(performance.now() - startedAt),
    slow_threshold_ms: SLOW_MS,
    top_by_total_ms: rows.slice(0, topN),
    slow_by_max_ms: rows
      .filter((r) => r.max_ms >= SLOW_MS)
      .sort((a, b) => b.max_ms - a.max_ms)
      .slice(0, topN),
    recent: recent.slice(-50),
  };
}

export function reset() {
  buckets.clear();
  recent.length = 0;
  return { ok: true };
}

export function setEnabled(on) {
  ENABLED = Boolean(on);
  try {
    localStorage.setItem(STORAGE_KEY, ENABLED ? "1" : "0");
  } catch {
    // ignore
  }
}

if (typeof window !== "undefined") {
  window.__NHL_PERF = {
    snapshot,
    reset,
    record,
    span,
    markNavigation,
    markInteraction,
    setEnabled,
    measureAsync,
  };
  // Initial page load mark
  if (ENABLED && typeof performance !== "undefined" && performance.timing) {
    window.addEventListener("load", () => {
      try {
        const t = performance.timing;
        if (t.loadEventEnd && t.navigationStart) {
          record("ui.page_load", t.loadEventEnd - t.navigationStart);
        }
      } catch {
        // ignore
      }
      try {
        const nav = performance.getEntriesByType?.("navigation")?.[0];
        if (nav) {
          record("ui.navigation_entry", nav.duration, {
            domContentLoaded: nav.domContentLoadedEventEnd,
            responseEnd: nav.responseEnd,
          });
        }
      } catch {
        // ignore
      }
    });
  }
}

export default {
  record,
  span,
  measureAsync,
  markNavigation,
  markInteraction,
  snapshot,
  reset,
  setEnabled,
};
