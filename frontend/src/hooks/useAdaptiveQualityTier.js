import { useEffect, useRef, useState } from "react";

/*
 * Automatic graphics tier ("high" | "low") from what the device is actually doing.
 *
 * Browsers don't expose GPU/CPU percentages, so load is read from the signals they do give:
 *  - frame pacing: requestAnimationFrame cadence vs the display's refresh (GPU + CPU together)
 *  - main-thread long tasks (CPU)
 *  - Compute Pressure API CPU state, where supported (Chromium: nominal/fair/serious/critical)
 *  - battery: discharging below 20% prefers low
 *
 * Dropping to low needs a few seconds of sustained stress; returning to high needs a longer
 * run of headroom. A failed upgrade doubles the wait before the next attempt, so a device on
 * the edge doesn't flip-flop (each switch recompiles shaders).
 */

const WINDOW_MS = 1000;
const SETTLE_MS = 4000;
const DOWN_WINDOWS = 3;
const UP_WINDOWS = 10;
const RELAPSE_MS = 12000;
const BASE_COOLDOWN_MS = 30000;
const MAX_COOLDOWN_MS = 10 * 60 * 1000;
const MIN_FPS_HIGH = 40;

export function useAdaptiveQualityTier({ enabled, initialTier = "high" }) {
  const [state, setState] = useState({ tier: initialTier, reason: "initial" });
  const tierRef = useRef(state.tier);

  useEffect(() => {
    if (!enabled) return undefined;

    let raf = 0;
    let last = 0;
    let deltas = [];
    let windowStart = performance.now();
    let settleUntil = windowStart + SETTLE_MS;
    let stressStreak = 0;
    let headroomStreak = 0;
    let cooldownMs = BASE_COOLDOWN_MS;
    let nextUpgradeAt = 0;
    let lastUpgradeAt = -Infinity;
    let longTaskMs = 0;
    let cpuPressure = "unknown";
    let battery = null;
    let vsyncMs = 1000 / 60;

    let longTaskObserver = null;
    try {
      longTaskObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) longTaskMs += entry.duration;
      });
      longTaskObserver.observe({ type: "longtask", buffered: false });
    } catch {
      longTaskObserver = null;
    }

    let pressureObserver = null;
    if (typeof window.PressureObserver === "function") {
      try {
        pressureObserver = new window.PressureObserver((records) => {
          const latest = records[records.length - 1];
          if (latest?.state) cpuPressure = latest.state;
        });
        Promise.resolve(pressureObserver.observe("cpu", { sampleInterval: 1000 })).catch(() => {
          cpuPressure = "unknown";
        });
      } catch {
        pressureObserver = null;
      }
    }

    if (typeof navigator.getBattery === "function") {
      navigator.getBattery().then((b) => { battery = b; }).catch(() => {});
    }

    const switchTo = (tier, reason) => {
      if (tierRef.current === tier) return;
      const now = performance.now();
      tierRef.current = tier;
      setState({ tier, reason });
      settleUntil = now + SETTLE_MS;
      stressStreak = 0;
      headroomStreak = 0;
      deltas = [];
      if (tier === "high") {
        lastUpgradeAt = now;
      } else {
        if (now - lastUpgradeAt < RELAPSE_MS) {
          cooldownMs = Math.min(cooldownMs * 2, MAX_COOLDOWN_MS);
        }
        nextUpgradeAt = now + cooldownMs;
      }
    };

    const evaluate = (now) => {
      const elapsed = Math.max(1, now - windowStart);
      const longTaskShare = longTaskMs / elapsed;
      const sample = deltas;
      longTaskMs = 0;
      deltas = [];
      windowStart = now;
      if (sample.length < 5) return;

      const sorted = [...sample].sort((a, b) => a - b);
      vsyncMs = Math.min(vsyncMs, Math.max(4, sorted[Math.floor(sorted.length * 0.1)]));
      const meanMs = sample.reduce((sum, d) => sum + d, 0) / sample.length;
      const fps = 1000 / meanMs;
      const jank = sample.filter((d) => d > vsyncMs * 1.8).length / sample.length;
      if (now < settleUntil) return;

      const cpuHot = cpuPressure === "serious" || cpuPressure === "critical";
      const lowBattery = Boolean(battery && !battery.charging && battery.level <= 0.2);

      if (tierRef.current === "high") {
        if (lowBattery) {
          switchTo("low", "battery");
          return;
        }
        const stressed = fps < MIN_FPS_HIGH || jank > 0.25 || cpuHot || longTaskShare > 0.3;
        stressStreak = stressed ? stressStreak + 1 : 0;
        if (stressStreak >= DOWN_WINDOWS) {
          switchTo("low", cpuHot || longTaskShare > 0.3 ? "cpu-load" : "frame-rate");
        }
        return;
      }

      // Low tier renders at a capped 30 fps, so headroom shows as the page still keeping
      // the display's full refresh cadence with no dropped frames or CPU pressure.
      const refreshFps = 1000 / vsyncMs;
      const roomy =
        fps >= refreshFps * 0.9 && jank < 0.02 && !cpuHot && longTaskShare < 0.1 && !lowBattery;
      headroomStreak = roomy ? headroomStreak + 1 : 0;
      if (headroomStreak >= UP_WINDOWS && now >= nextUpgradeAt) switchTo("high", "headroom");
    };

    const loop = (t) => {
      raf = requestAnimationFrame(loop);
      if (document.hidden) {
        last = 0;
        return;
      }
      if (last) {
        const delta = t - last;
        if (delta < 250) deltas.push(delta);
      }
      last = t;
      if (t - windowStart >= WINDOW_MS) evaluate(t);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      longTaskObserver?.disconnect();
      try {
        pressureObserver?.disconnect();
      } catch {
        /* already closed */
      }
    };
  }, [enabled]);

  return state;
}
