import { useCallback, useEffect, useMemo, useState } from "react";
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
  const { franchiseState, refreshFranchise, setScreen, setFranchiseState } = useGameUI();
  const crisis = franchiseState?.trade_demand_crisis || null;
  const [localRemaining, setLocalRemaining] = useState(null);

  useEffect(() => {
    if (!crisis) {
      setLocalRemaining(null);
      return undefined;
    }
    setLocalRemaining(Number(crisis.remaining_seconds) || 0);
    const sync = setInterval(() => {
      refreshFranchise?.();
    }, 2000);
    return () => clearInterval(sync);
  }, [crisis?.demand_id, crisis?.remaining_seconds, refreshFranchise]);

  useEffect(() => {
    if (!crisis) return undefined;
    setLocalRemaining(Number(crisis.remaining_seconds) || 0);
    const tick = setInterval(() => {
      setLocalRemaining((prev) => Math.max(0, (prev ?? 0) - 1));
    }, 1000);
    return () => clearInterval(tick);
  }, [crisis?.demand_id, crisis?.remaining_seconds]);

  const openTradeHub = useCallback(() => {
    setScreen?.("tradehub");
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

      <style>{`
        .trade-crisis-overlay {
          position: fixed;
          inset: 0;
          z-index: 12000;
          display: flex;
          align-items: center;
          justify-content: center;
          pointer-events: auto;
        }
        .trade-crisis-overlay__backdrop {
          position: absolute;
          inset: 0;
          background: rgba(4, 8, 16, 0.82);
          backdrop-filter: blur(3px);
        }
        .trade-crisis-overlay__panel {
          position: relative;
          width: min(560px, calc(100vw - 32px));
          border: 1px solid rgba(255, 90, 90, 0.45);
          border-radius: 14px;
          background: linear-gradient(165deg, rgba(22, 12, 18, 0.97), rgba(10, 14, 24, 0.98));
          box-shadow: 0 24px 80px rgba(0, 0, 0, 0.55), inset 0 0 0 1px rgba(255, 255, 255, 0.04);
          padding: 24px 26px 22px;
          color: #f3f6fb;
        }
        .trade-crisis-overlay__eyebrow {
          margin: 0;
          font-size: 11px;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          color: #ff8a8a;
        }
        .trade-crisis-overlay__head h2 {
          margin: 6px 0 4px;
          font-size: 28px;
          line-height: 1.1;
        }
        .trade-crisis-overlay__stage {
          margin: 0;
          color: #c7d2e3;
          font-size: 14px;
        }
        .trade-crisis-overlay__timer {
          margin: 18px 0 16px;
          padding: 14px 16px;
          border-radius: 10px;
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid rgba(255, 255, 255, 0.08);
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 12px;
        }
        .trade-crisis-overlay__timer[data-urgent="true"] {
          border-color: rgba(255, 70, 70, 0.55);
          background: rgba(120, 20, 20, 0.25);
        }
        .trade-crisis-overlay__timer-label {
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: #aebcd0;
        }
        .trade-crisis-overlay__timer-value {
          font-size: 36px;
          font-variant-numeric: tabular-nums;
          color: #fff;
        }
        .trade-crisis-overlay__grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
        }
        .trade-crisis-overlay__grid div {
          padding: 10px 12px;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.06);
          display: flex;
          flex-direction: column;
          gap: 4px;
        }
        .trade-crisis-overlay__grid span {
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: #95a3ba;
        }
        .trade-crisis-overlay__grid strong {
          font-size: 14px;
          line-height: 1.35;
        }
        .trade-crisis-overlay__grid em {
          font-size: 12px;
          color: #b8c4d8;
          font-style: normal;
        }
        .trade-crisis-overlay__body {
          margin: 14px 0 0;
          color: #d6deea;
          line-height: 1.45;
          font-size: 14px;
        }
        .trade-crisis-overlay__actions {
          margin-top: 18px;
          display: flex;
          justify-content: flex-end;
          gap: 10px;
        }
        .trade-crisis-overlay__dismiss {
          border: 1px solid rgba(255, 255, 255, 0.18);
          border-radius: 999px;
          padding: 11px 16px;
          font-weight: 700;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          font-size: 12px;
          cursor: pointer;
          color: #e8edf5;
          background: transparent;
        }
        .trade-crisis-overlay__cta {
          border: none;
          border-radius: 999px;
          padding: 11px 20px;
          font-weight: 700;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          font-size: 12px;
          cursor: pointer;
          color: #12080a;
          background: linear-gradient(90deg, #ffb4b4, #ff6868);
        }
      `}</style>
    </div>
  );
}
