import { useCallback, useEffect, useMemo, useState } from "react";
import { SCREENS } from "../../game/constants";
import { useGameUI } from "../../game/GameUIContext";

function formatCountdown(totalSeconds) {
  const sec = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function stageLabel(stage) {
  const n = Number(stage) || 1;
  if (n >= 4) return "Full Breakdown";
  if (n >= 3) return "Severe Leverage Loss";
  if (n >= 2) return "Demand Going Public";
  return "Formal Trade Demand";
}

export function TradeDemandCrisisOverlay() {
  const { franchiseState, setScreen, setFranchiseState } = useGameUI();
  const crisis = franchiseState?.trade_demand_crisis || null;
  const [localRemaining, setLocalRemaining] = useState(null);

  useEffect(() => {
    if (!crisis) return undefined;
    setLocalRemaining(Number(crisis.remaining_seconds) || 0);
    const tick = setInterval(() => {
      setLocalRemaining((prev) => Math.max(0, (prev ?? 0) - 1));
    }, 1000);
    return () => clearInterval(tick);
  }, [crisis?.demand_id, crisis?.remaining_seconds]);

  const openTradeHub = useCallback(() => {
    setScreen?.(SCREENS.TRADE);
  }, [setScreen]);

  const dismissCrisis = useCallback(() => {
    setFranchiseState?.((prev) => (prev ? { ...prev, trade_demand_crisis: null } : prev));
  }, [setFranchiseState]);

  const displayRemaining = useMemo(() => {
    if (localRemaining == null && crisis) return Number(crisis.remaining_seconds) || 0;
    return localRemaining ?? 0;
  }, [localRemaining, crisis]);

  if (!crisis) return null;

  const agent = crisis.agent || {};
  const dests = Array.isArray(crisis.preferred_destinations) ? crisis.preferred_destinations : [];
  const destLabel =
    dests.length >= 30
      ? "All 32 clubs"
      : dests.length
        ? dests.slice(0, 10).join(", ") + (dests.length > 10 ? "…" : "")
        : "Open market";

  return (
    <div className="trade-crisis-overlay" role="dialog" aria-modal="true" aria-labelledby="trade-crisis-title">
      <div className="trade-crisis-overlay__backdrop" />
      <div className="trade-crisis-overlay__panel">
        <header className="trade-crisis-overlay__head">
          <p className="trade-crisis-overlay__eyebrow">Trade Demand Crisis</p>
          <h2 id="trade-crisis-title">{crisis.player_name || "Player"}</h2>
          <p className="trade-crisis-overlay__stage">{stageLabel(crisis.crisis_stage)}</p>
        </header>

        <div className="trade-crisis-overlay__timer" data-urgent={displayRemaining <= 120}>
          <span className="trade-crisis-overlay__timer-label">Time remaining</span>
          <strong className="trade-crisis-overlay__timer-value">{formatCountdown(displayRemaining)}</strong>
        </div>

        <div className="trade-crisis-overlay__grid">
          <div>
            <span>Agent</span>
            <strong>{agent.name || "—"}</strong>
            <em>{agent.agency || ""}</em>
          </div>
          <div>
            <span>Primary complaint</span>
            <strong>{crisis.primary_complaint || "—"}</strong>
          </div>
          <div>
            <span>Trade value</span>
            <strong>
              {crisis.value_before ?? "—"} → {crisis.value_after ?? "—"}
            </strong>
          </div>
          <div>
            <span>Destinations</span>
            <strong>{destLabel}</strong>
          </div>
        </div>

        {crisis.body ? <p className="trade-crisis-overlay__body">{crisis.body}</p> : null}

        <div className="trade-crisis-overlay__actions">
          <button type="button" className="trade-crisis-overlay__dismiss" onClick={dismissCrisis}>
            Continue
          </button>
          <button type="button" className="trade-crisis-overlay__cta" onClick={openTradeHub}>
            Open Trade Hub
          </button>
        </div>
      </div>

      {null}
    </div>
  );
}
