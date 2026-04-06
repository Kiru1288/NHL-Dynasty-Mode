import React, { useEffect, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { GameFooter } from "../components/game/GameFooter";
import { GameHeader } from "../components/game/GameHeader";

export function SetupScreen() {
  const {
    teams,
    setupTeamIndex,
    setSetupTeamIndex,
    setupArchetypeIndex,
    setSetupArchetypeIndex,
    archetypes,
    coachName,
    setCoachName,
    beginFranchise,
    loading,
    error,
  } = useGameUI();

  const [focusSlot, setFocusSlot] = useState(0);

  useEffect(() => {
    function onKey(e) {
      if (e.target.tagName === "INPUT") {
        if (e.key === "Enter") {
          e.preventDefault();
          beginFranchise();
        }
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setFocusSlot((s) => Math.max(0, s - 1));
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setFocusSlot((s) => Math.min(3, s + 1));
      } else if (e.key === "ArrowLeft" && focusSlot === 0 && teams.length) {
        e.preventDefault();
        setSetupTeamIndex((i) => Math.max(0, i - 1));
      } else if (e.key === "ArrowRight" && focusSlot === 0 && teams.length) {
        e.preventDefault();
        setSetupTeamIndex((i) => Math.min(teams.length - 1, i + 1));
      } else if (e.key === "ArrowLeft" && focusSlot === 1) {
        e.preventDefault();
        setSetupArchetypeIndex((i) => Math.max(0, i - 1));
      } else if (e.key === "ArrowRight" && focusSlot === 1) {
        e.preventDefault();
        setSetupArchetypeIndex((i) => Math.min(archetypes.length - 1, i + 1));
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (focusSlot === 3) beginFranchise();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    archetypes.length,
    beginFranchise,
    focusSlot,
    setSetupArchetypeIndex,
    setSetupTeamIndex,
    teams.length,
  ]);

  return (
    <div className="game-screen setup-screen">
      <GameHeader teamName="FRANCHISE MODE" sectionTitle="START UP" />
      <div className="setup-body">
        <div className="setup-panel game-panel-bevel">
          <div
            className={`setup-row ${focusSlot === 0 ? "is-focus-row" : ""}`}
            onClick={() => setFocusSlot(0)}
            role="presentation"
          >
            <span className="setup-label">CLUB</span>
            <span className="setup-value">
              {teams[setupTeamIndex]?.name || "—"} ({teams[setupTeamIndex]?.team_id})
            </span>
          </div>
          <div
            className={`setup-row ${focusSlot === 1 ? "is-focus-row" : ""}`}
            onClick={() => setFocusSlot(1)}
            role="presentation"
          >
            <span className="setup-label">COACH PHILOSOPHY</span>
            <span className="setup-value">{archetypes[setupArchetypeIndex]?.replace(/_/g, " ")}</span>
          </div>
          <div className={`setup-row setup-row--input ${focusSlot === 2 ? "is-focus-row" : ""}`}>
            <span className="setup-label">HEAD COACH NAME</span>
            <input
              className="game-input"
              value={coachName}
              onChange={(e) => setCoachName(e.target.value)}
              onFocus={() => setFocusSlot(2)}
            />
          </div>
          <button
            type="button"
            className={`game-cta ${focusSlot === 3 ? "is-focus-row" : ""}`}
            disabled={loading || !teams.length}
            onClick={() => beginFranchise()}
            onFocus={() => setFocusSlot(3)}
          >
            {loading ? "LOADING…" : "BEGIN FRANCHISE"}
          </button>
        </div>
        {error && <div className="game-toast game-toast--err">{error}</div>}
        <p className="setup-hint">
          Keyboard: ↑↓ fields · ←→ club / philosophy when highlighted · Enter to start
        </p>
      </div>
      <GameFooter />
    </div>
  );
}
