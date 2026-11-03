import { getFranchiseAdvanceProgress } from "../services/franchiseService";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** True when the backend reports no sim running and no post-sim finalize. */
function isIdle(p) {
  return !p || (!p.active && !p.finalizing);
}

/**
 * After a failed / aborted advance request, wait (bounded) until the backend
 * sim is idle so the resync reads the finished state, not a half-simmed day.
 */
export async function waitForSimIdle({ maxMs = 120000, intervalMs = 1500 } = {}) {
  const until = Date.now() + maxMs;
  while (Date.now() < until) {
    const p = await getFranchiseAdvanceProgress();
    if (isIdle(p)) return true;
    await sleep(intervalMs);
  }
  return false;
}

/**
 * While an advance request is in flight, watch the progress endpoint. When the
 * backend has been idle for `graceMs` but the request still has not answered
 * (dropped socket, proxy hang, response lost), abort it so the UI can resync
 * instead of freezing behind the overlay until a manual refresh.
 * Returns a stop() function.
 */
export function startSimStallWatchdog(controller, { graceMs = 120000, intervalMs = 3000 } = {}) {
  let idleSince = 0;
  let stopped = false;
  const id = setInterval(async () => {
    if (stopped) return;
    const p = await getFranchiseAdvanceProgress();
    if (stopped) return;
    if (isIdle(p)) {
      if (!idleSince) idleSince = Date.now();
      else if (Date.now() - idleSince >= graceMs) {
        stopped = true;
        clearInterval(id);
        try {
          controller.abort();
        } catch {
          /* ignore */
        }
      }
    } else {
      idleSince = 0;
    }
  }, intervalMs);
  return () => {
    stopped = true;
    clearInterval(id);
  };
}

export function isAbortError(e) {
  return e?.code === "ERR_CANCELED" || e?.name === "CanceledError" || e?.name === "AbortError" || e?.cause?.code === "ERR_CANCELED";
}

export function isSimBusyError(e) {
  return Number(e?.response?.status) === 409 || Number(e?.cause?.response?.status) === 409;
}
