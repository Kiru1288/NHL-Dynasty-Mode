import React, { useEffect } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SETTINGS_ROWS, SCREENS } from "../game/constants";
import { GameFooter } from "../components/game/GameFooter";
import { GameHeader } from "../components/game/GameHeader";

export function SettingsScreen() {
  const {
    franchiseState,
    settingsRowIndex,
    setSettingsRowIndex,
    ruleSliders,
    adjustSlider,
    setScreen,
  } = useGameUI();

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
        setSettingsRowIndex((i) => Math.max(0, i - 1));
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setSettingsRowIndex((i) => Math.min(SETTINGS_ROWS.length - 1, i + 1));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        const row = SETTINGS_ROWS[settingsRowIndex];
        if (row) adjustSlider(row.key, -1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        const row = SETTINGS_ROWS[settingsRowIndex];
        if (row) adjustSlider(row.key, 1);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [adjustSlider, setScreen, setSettingsRowIndex, settingsRowIndex]);

  return (
    <div className="game-screen settings-screen">
      <GameHeader teamName={franchiseState?.team?.name || "—"} sectionTitle="ADVANCED" />
      <div className="settings-list game-panel-bevel">
        {SETTINGS_ROWS.map((row, idx) => {
          const v = ruleSliders[row.key] ?? 50;
          const sel = idx === settingsRowIndex;
          return (
            <div
              key={row.key}
              className={`settings-row ${sel ? "is-selected" : ""}`}
              onClick={() => setSettingsRowIndex(idx)}
              role="presentation"
            >
              <span className="settings-row__label">{row.label}</span>
              <div className="settings-slider">
                <div className="settings-slider__track">
                  <div className="settings-slider__fill" style={{ width: `${v}%` }} />
                </div>
                <span className="settings-slider__val">{v}</span>
              </div>
            </div>
          );
        })}
        <p className="settings-note">
          Sliders are local display rules (discrete steps). Sim tuning hooks can bind later.
        </p>
      </div>
      <GameFooter hints="↑↓ ROW  ·  ←→ VALUE  ·  ESC BACK" />
    </div>
  );
}
