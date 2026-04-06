import React, { useEffect, useRef } from "react";
import { useGameUI } from "../game/GameUIContext";
import { HUB_MENU } from "../game/constants";
import { GameFooter } from "../components/game/GameFooter";
import { GameHeader } from "../components/game/GameHeader";

export function HubScreen() {
  const {
    franchiseState,
    hubMenuIndex,
    setHubMenuIndex,
    openHubMenu,
    error,
    onAdvanceDay,
    advancing,
    onResolveDecision,
    refreshFranchise,
  } = useGameUI();

  const team = franchiseState?.team;
  const rec = team?.record;
  const pending = franchiseState?.pending_decisions || [];
  const hubMenuIndexRef = useRef(hubMenuIndex);
  hubMenuIndexRef.current = hubMenuIndex;

  useEffect(() => {
    function onKey(e) {
      if (e.target.matches("input, textarea, select, button")) return;
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setHubMenuIndex((i) => Math.max(0, i - 1));
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setHubMenuIndex((i) => Math.min(HUB_MENU.length - 1, i + 1));
      } else if (e.key === "Enter") {
        e.preventDefault();
        openHubMenu(hubMenuIndexRef.current);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [openHubMenu, setHubMenuIndex]);

  return (
    <div className="game-screen hub-screen">
      <GameHeader
        teamName={team?.name || "—"}
        sectionTitle="FRANCHISE OPS"
      />
      <div className="hub-split">
        <nav className="hub-rail game-panel-bevel" aria-label="Main menu">
          {HUB_MENU.map((item, idx) => (
            <div
              key={item.id}
              className={`hub-rail__item ${idx === hubMenuIndex ? "is-selected" : ""}`}
              onClick={() => setHubMenuIndex(idx)}
              onDoubleClick={() => openHubMenu(idx)}
              role="button"
              tabIndex={-1}
            >
              {item.label}
            </div>
          ))}
        </nav>
        <div className="hub-main game-panel-bevel">
          <div className="hub-meta">
            <span>
              REC {rec ? `${rec.w}-${rec.l}-${rec.otl}` : "0-0-0"} · {rec?.pts ?? 0} PTS
            </span>
            <span>CAP {team?.cap_pressure}</span>
            <span>{franchiseState?.calendar_summary}</span>
          </div>
          {error && <div className="game-toast game-toast--err">{error}</div>}
          <div className="hub-feed">
            {(franchiseState?.timeline || []).slice(-24).join("\n") || "— Awaiting orders —"}
          </div>
          <div className="hub-actions">
            <button
              type="button"
              className="game-btn game-btn--primary"
              disabled={!franchiseState?.flags?.can_advance || advancing}
              onClick={onAdvanceDay}
            >
              {advancing ? "ADVANCING…" : "ADVANCE DAY"}
            </button>
            <button type="button" className="game-btn" onClick={refreshFranchise}>
              REFRESH
            </button>
          </div>
          {pending.length > 0 && (
            <div className="hub-decisions">
              <div className="hub-decisions__title">PENDING DECISIONS</div>
              {pending.map((d) => (
                <div key={d.id} className="decision-block">
                  <div className="decision-block__t">{d.title}</div>
                  <div className="decision-block__d">{d.description}</div>
                  <div className="decision-block__opts">
                    {(d.options || []).map((o) => (
                      <button
                        key={o.id}
                        type="button"
                        className="game-btn game-btn--sm"
                        onClick={() => onResolveDecision(d.id, o.id)}
                      >
                        {o.label}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
          {franchiseState?.phase === "complete" && (
            <div className="game-toast">Season complete — NEW FRANCHISE from menu.</div>
          )}
        </div>
      </div>
      <GameFooter />
    </div>
  );
}
