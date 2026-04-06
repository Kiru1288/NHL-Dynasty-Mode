import React, { useEffect, useMemo } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { enrichRosterPlayer } from "../game/rosterColumns";
import { GameFooter } from "../components/game/GameFooter";
import { GameHeader } from "../components/game/GameHeader";

export function RosterScreen() {
  const { franchiseState, rosterRowIndex, setRosterRowIndex, setScreen } = useGameUI();

  const players = useMemo(() => {
    const raw = franchiseState?.roster || [];
    return raw.map((p, i) => enrichRosterPlayer(p, i));
  }, [franchiseState?.roster]);

  const selected = players[rosterRowIndex] || null;

  useEffect(() => {
    if (rosterRowIndex >= players.length) {
      setRosterRowIndex(Math.max(0, players.length - 1));
    }
  }, [players.length, rosterRowIndex, setRosterRowIndex]);

  useEffect(() => {
    function onKey(e) {
      if (e.target.matches("input, textarea, select")) return;
      if (e.key === "Escape") {
        e.preventDefault();
        setScreen(SCREENS.HUB);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setRosterRowIndex((i) => Math.max(0, i - 1));
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        if (players.length === 0) return;
        setRosterRowIndex((i) => Math.min(players.length - 1, i + 1));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [players.length, setRosterRowIndex, setScreen]);

  return (
    <div className="game-screen roster-screen">
      <GameHeader
        teamName={franchiseState?.team?.name || "—"}
        sectionTitle="ROSTERS"
      />
      <div className="roster-layout">
        <aside className="roster-spotlight game-panel-bevel">
          <div className="roster-portrait" aria-hidden>
            <span className="roster-portrait__silhouette" />
          </div>
          <div className="roster-spotlight__name">{selected?.name || "—"}</div>
          <div className="roster-meta-grid">
            <span>HGT</span>
            <span>{selected?.hgt || "—"}</span>
            <span>WGT</span>
            <span>{selected?.wgt || "—"}</span>
            <span>AGE</span>
            <span>{selected?.age ?? "—"}</span>
            <span>PSH</span>
            <span>{selected?.shot || "—"}</span>
            <span>NAT</span>
            <span>{selected?.nat || "—"}</span>
            <span>STS</span>
            <span>{selected?.status || "—"}</span>
          </div>
        </aside>
        <div className="roster-table-wrap game-panel-bevel">
          <div className="roster-table-header">
            <span className="col-name">NAME</span>
            <span className="col-num">OVR</span>
            <span className="col-j">#</span>
            <span className="col-age">AGE</span>
            <span className="col-pos">POS</span>
            <span className="col-cpt">CPT</span>
            <span className="col-st">STAT</span>
            <span className="col-off">OFF</span>
          </div>
          <div className="roster-table-body">
            {players.map((p, idx) => (
              <div
                key={`${p.name}-${idx}`}
                className={`roster-row ${idx === rosterRowIndex ? "is-selected" : ""}`}
                onClick={() => setRosterRowIndex(idx)}
                role="row"
              >
                <span className="col-name">{p.name}</span>
                <span className="col-num">{p.ovr}</span>
                <span className="col-j">{p.num}</span>
                <span className="col-age">{p.age}</span>
                <span className="col-pos">{p.position}</span>
                <span className="col-cpt">{p.cpt}</span>
                <span className="col-st">{p.status}</span>
                <span className="col-off">{p.off}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <GameFooter hints="↑↓ ROSTER  ·  ESC FRANCHISE OPS" />
    </div>
  );
}
