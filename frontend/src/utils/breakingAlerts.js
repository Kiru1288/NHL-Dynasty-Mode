const STORAGE_PREFIX = "nhl_breaking_dismissed_";

export function breakingAlertKey(alert) {
  const sid = alert?.storyline_id ?? alert?.id;
  if (sid != null && String(sid).trim()) return String(sid).trim();
  const headline = String(alert?.headline || "").trim();
  const date = String(alert?.calendar_iso || "").trim();
  if (headline) return `${headline}::${date || "nodate"}`;
  return "";
}

export function readDismissedBreakingKeys(sessionId) {
  const sid = String(sessionId || "anon").trim() || "anon";
  try {
    const raw = sessionStorage.getItem(`${STORAGE_PREFIX}${sid}`);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw);
    return new Set(Array.isArray(parsed) ? parsed.filter(Boolean) : []);
  } catch {
    return new Set();
  }
}

export function writeDismissedBreakingKeys(sessionId, keys) {
  const sid = String(sessionId || "anon").trim() || "anon";
  try {
    sessionStorage.setItem(`${STORAGE_PREFIX}${sid}`, JSON.stringify([...keys]));
  } catch {
    /* storage optional */
  }
}

export function activeBreakingAlerts(alerts, dismissedKeys) {
  return (Array.isArray(alerts) ? alerts : []).filter((alert) => {
    const key = breakingAlertKey(alert);
    return key && !dismissedKeys.has(key);
  });
}
