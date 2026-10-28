import React, { useEffect, useState } from "react";
import {
  readUiScalePreference,
  writeUiScalePreference,
  UI_SCALE_PRESETS,
} from "../utils/fluidUiScale";
import {
  GRAPHICS_QUALITY_PRESETS,
  readGraphicsQuality,
  writeGraphicsQuality,
} from "../utils/graphicsQuality";
import { useGameUI } from "../game/GameUIContext";
import { SETTINGS_ROWS, SCREENS } from "../game/constants";
import { GameFooter } from "../components/game/GameFooter";
import { GameHeader } from "../components/game/GameHeader";

const AUTO_REASON_TEXT = {
  initial: "starting from your hardware profile",
  "frame-rate": "frame rate dropped",
  "cpu-load": "CPU was under heavy load",
  battery: "battery is low and unplugged",
  headroom: "your device has spare capacity",
};

function readLiveQuality() {
  try {
    return JSON.parse(window.sessionStorage.getItem("nhl.graphicsQuality.live") || "null");
  } catch {
    return null;
  }
}

const SETTINGS_CATEGORIES = [
  {
    id: "penalties",
    label: "Penalty Frequency",
    note: "Local display rules — sim binding not wired",
    keys: SETTINGS_ROWS.map((row) => row.key),
  },
];

export function SettingsScreen() {
  const {
    franchiseState,
    settingsRowIndex,
    setSettingsRowIndex,
    ruleSliders,
    adjustSlider,
    setScreen,
  } = useGameUI();
  const [uiScalePref, setUiScalePref] = useState(() => readUiScalePreference());
  const [qualityPref, setQualityPref] = useState(() => readGraphicsQuality());
  const liveQuality = readLiveQuality();

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
    <div className="game-screen settings-screen register-shell" data-register="shell">
      <GameHeader teamName={franchiseState?.team?.name || "—"} sectionTitle="SETTINGS" />
      <div className="settings-shell">
        <header className="settings-shell__head">
          <span className="settings-shell__kicker">Settings</span>
          <h2 className="settings-shell__title">Display &amp; Rules</h2>
          <p className="settings-shell__copy">
            Display scale applies immediately. Penalty sliders are saved locally and do not change simulation results yet.
          </p>
        </header>

        <section className="settings-category">
          <div className="settings-category__head">
            <span className="settings-category__label">Display</span>
            <span className="settings-category__status fcn-stamp">Live</span>
          </div>
          <div className="settings-scale-row">
            {UI_SCALE_PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className={`settings-scale-btn ${uiScalePref === preset.id ? "is-active" : ""}`}
                onClick={() => {
                  setUiScalePref(preset.id);
                  writeUiScalePreference(preset.id);
                }}
              >
                {preset.label}
              </button>
            ))}
          </div>
          <p className="settings-note">
            Match display picks spacing and type from the real window (laptop / 1080p / 1440p / 4K). It does not zoom the page.
          </p>
        </section>

        <section className="settings-category">
          <div className="settings-category__head">
            <span className="settings-category__label">Graphics Quality</span>
            <span className="settings-category__status fcn-stamp">Live</span>
          </div>
          <div className="settings-scale-row">
            {GRAPHICS_QUALITY_PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                className={`settings-scale-btn ${qualityPref === preset.id ? "is-active" : ""}`}
                onClick={() => {
                  setQualityPref(preset.id);
                  writeGraphicsQuality(preset.id);
                }}
              >
                {preset.label}
              </button>
            ))}
          </div>
          <p className="settings-note">
            Auto watches frame rate, CPU load and battery while you play and switches the office
            between High and Low on its own. Low caps the office at 30 fps, drops shadows and
            accent lights, and turns off background blur and decorative animation across the game.
            {qualityPref === "auto" && liveQuality?.mode === "auto"
              ? ` Currently ${liveQuality.tier === "low" ? "Low" : "High"}${
                  AUTO_REASON_TEXT[liveQuality.reason] ? ` — ${AUTO_REASON_TEXT[liveQuality.reason]}` : ""
                }.`
              : ""}
          </p>
        </section>

        {SETTINGS_CATEGORIES.map((category) => (
          <section key={category.id} className="settings-category">
            <div className="settings-category__head">
              <span className="settings-category__label">{category.label}</span>
              <span className="settings-category__status fcn-stamp fcn-stamp--locked">
                Not bound to sim
              </span>
            </div>

            <div className="settings-list">
              {SETTINGS_ROWS.filter((row) => category.keys.includes(row.key)).map((row) => {
                const idx = SETTINGS_ROWS.findIndex((r) => r.key === row.key);
                const v = ruleSliders[row.key] ?? 50;
                const sel = idx === settingsRowIndex;
                return (
                  <div
                    key={row.key}
                    className={`settings-row ui-interactive ${sel ? "is-selected" : ""}`}
                    data-tooltip="Local presentation rule — sim binding can wire later"
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
            </div>

            <p className="settings-note">{category.note}</p>
          </section>
        ))}
      </div>
      <GameFooter hints="↑↓ ROW  ·  ←→ VALUE  ·  ESC BACK" />
      {null}
    </div>
  );
}


