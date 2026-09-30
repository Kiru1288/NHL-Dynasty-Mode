import { act, renderHook } from "@testing-library/react";
import { useAdaptiveQualityTier } from "./useAdaptiveQualityTier";

let now = 0;
let queued = [];

beforeEach(() => {
  now = 0;
  queued = [];
  jest.spyOn(performance, "now").mockImplementation(() => now);
  window.requestAnimationFrame = (cb) => {
    queued.push(cb);
    return queued.length;
  };
  window.cancelAnimationFrame = () => {};
  window.PerformanceObserver = class {
    observe() {}
    disconnect() {}
  };
  delete window.PressureObserver;
  Object.defineProperty(navigator, "getBattery", { value: undefined, configurable: true });
});

afterEach(() => {
  jest.restoreAllMocks();
});

/** Advance the fake display, delivering one animation frame every `frameMs`. */
function runFrames(durationMs, frameMs) {
  const end = now + durationMs;
  while (now < end) {
    now += frameMs;
    const callbacks = queued;
    queued = [];
    act(() => {
      callbacks.forEach((cb) => cb(now));
    });
  }
}

test("drops to low after sustained slow frames", () => {
  const { result } = renderHook(() => useAdaptiveQualityTier({ enabled: true, initialTier: "high" }));
  runFrames(2000, 1000 / 60);
  expect(result.current.tier).toBe("high");
  runFrames(8000, 50);
  expect(result.current.tier).toBe("low");
  expect(result.current.reason).toBe("frame-rate");
});

test("ignores a brief hitch", () => {
  const { result } = renderHook(() => useAdaptiveQualityTier({ enabled: true, initialTier: "high" }));
  runFrames(5000, 1000 / 60);
  runFrames(1200, 60);
  runFrames(5000, 1000 / 60);
  expect(result.current.tier).toBe("high");
});

test("returns to high after sustained headroom and a cooldown", () => {
  const { result } = renderHook(() => useAdaptiveQualityTier({ enabled: true, initialTier: "high" }));
  runFrames(10000, 50);
  expect(result.current.tier).toBe("low");
  runFrames(20000, 1000 / 60);
  expect(result.current.tier).toBe("low"); // still inside the 30s cooldown
  runFrames(20000, 1000 / 60);
  expect(result.current.tier).toBe("high");
  expect(result.current.reason).toBe("headroom");
});

test("backs off after a failed upgrade", () => {
  const { result } = renderHook(() => useAdaptiveQualityTier({ enabled: true, initialTier: "high" }));
  runFrames(10000, 50); // -> low, next upgrade allowed after 30s
  let waited = 0;
  while (result.current.tier !== "high" && waited < 60000) {
    runFrames(500, 1000 / 60);
    waited += 500;
  }
  expect(result.current.tier).toBe("high");
  runFrames(9000, 50); // relapse within 12s of upgrading -> low, cooldown doubles to 60s
  expect(result.current.tier).toBe("low");
  runFrames(45000, 1000 / 60); // a plain 30s cooldown would have upgraded by now
  expect(result.current.tier).toBe("low");
  runFrames(30000, 1000 / 60);
  expect(result.current.tier).toBe("high");
});

test("drops to low under CPU pressure even at a smooth frame rate", () => {
  let report = null;
  window.PressureObserver = class {
    constructor(cb) {
      report = cb;
    }
    observe() {
      return Promise.resolve();
    }
    disconnect() {}
  };
  const { result } = renderHook(() => useAdaptiveQualityTier({ enabled: true, initialTier: "high" }));
  runFrames(5000, 1000 / 60);
  report([{ source: "cpu", state: "critical" }]);
  runFrames(5000, 1000 / 60);
  expect(result.current.tier).toBe("low");
  expect(result.current.reason).toBe("cpu-load");
});

test("drops to low on a low, unplugged battery", async () => {
  Object.defineProperty(navigator, "getBattery", {
    value: () => Promise.resolve({ charging: false, level: 0.12 }),
    configurable: true,
  });
  const { result } = renderHook(() => useAdaptiveQualityTier({ enabled: true, initialTier: "high" }));
  await act(async () => {});
  runFrames(6000, 1000 / 60);
  expect(result.current.tier).toBe("low");
  expect(result.current.reason).toBe("battery");
});

test("does nothing when disabled", () => {
  const { result } = renderHook(() => useAdaptiveQualityTier({ enabled: false, initialTier: "high" }));
  runFrames(10000, 50);
  expect(result.current.tier).toBe("high");
});
