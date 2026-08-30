import React, { useEffect, useState } from "react";
import {
  readUiScalePreference,
  writeUiScalePreference,
  UI_SCALE_PRESETS,
} from "../utils/fluidUiScale";
import { useGameUI } from "../game/GameUIContext";
import { SETTINGS_ROWS, SCREENS } from "../game/constants";
import { GameFooter } from "../components/game/GameFooter";
import { GameHeader } from "../components/game/GameHeader";

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
          <span className="settings-shell__kicker">Shell · System Registry</span>
          <h2 className="settings-shell__title">Rule Presentation</h2>
          <p className="settings-shell__copy">
            Adjust local slider values for penalty emphasis. These controls are not yet bound to the simulation engine.
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
      <style>{SETTINGS_SCREEN_CSS}</style>
    </div>
  );
}

const SETTINGS_SCREEN_CSS = `
.settings-screen.register-shell {
  background: var(--shell-bg, #03050c);
  color: var(--shell-text, #eef0f5);
}

.settings-shell {
  flex: 1;
  min-height: 0;
  margin: 0 var(--space-4, 16px) var(--space-3, 12px);
  padding: var(--space-4, 16px);
  overflow-y: auto;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: var(--radius-card, 8px);
  background: var(--shell-structure, rgba(26, 36, 61, 0.55));
  box-shadow: var(--depth-lifted, 0 12px 32px rgba(0, 0, 0, 0.36));
}

.settings-shell__head {
  margin-bottom: var(--space-4, 16px);
  padding-bottom: var(--space-3, 12px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.settings-shell__kicker {
  display: block;
  font-family: var(--font-ops-ui, Inter, sans-serif);
  font-size: var(--type-phase-label-size, 0.68rem);
  font-weight: 900;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--shell-text-muted, #8b93a8);
}

.settings-shell__title {
  margin: 6px 0 4px;
  font-family: var(--font-shell-headline, "Chakra Petch", sans-serif);
  font-size: clamp(1rem, 2vw, 1.2rem);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--shell-text, #eef0f5);
}

.settings-shell__copy {
  margin: 0;
  max-width: 56ch;
  font-family: var(--font-shell-body, "IBM Plex Sans", sans-serif);
  font-size: var(--type-compact-size, 0.8125rem);
  line-height: 1.45;
  color: var(--shell-text-muted, #8b93a8);
}

.settings-category {
  margin-bottom: var(--space-4, 16px);
}

.settings-category__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-2, 8px);
  margin-bottom: var(--space-2, 8px);
}

.settings-category__label {
  font-family: var(--font-ops-ui, Inter, sans-serif);
  font-size: var(--type-ops-heading-size, 0.95rem);
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--shell-text, #eef0f5);
}

/* Honest status mark: a registry stamp, not an alarm colour. */
.settings-screen .settings-category__status {
  color: var(--shell-text-muted, #8b93a8);
  background: transparent;
}

.settings-screen .settings-list {
  flex: unset;
  min-height: unset;
  width: 100%;
  margin: 0;
  padding: 0;
  overflow: visible;
  background: transparent;
  border: none;
  box-shadow: none;
}

/* Registry lines, not settings cards: hairline rows on a single ledger. */
.settings-screen .settings-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3, 12px);
  padding: 7px 10px 7px 12px;
  margin-bottom: 0;
  border: 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 0;
  background: transparent;
}

.settings-screen .settings-row.is-selected {
  background: rgba(56, 189, 248, 0.07);
  outline: none;
  box-shadow: inset 3px 0 0 var(--shell-neon, #38bdf8);
}

.settings-screen .settings-row__label {
  font-family: var(--font-shell-body, "IBM Plex Sans", sans-serif);
  font-size: var(--type-compact-size, 0.8125rem);
  font-weight: 600;
  letter-spacing: 0.04em;
  flex: 0 0 140px;
  color: var(--shell-text, #eef0f5);
}

/* Value reads on a ruled scale with tick marks every 25 units. */
.settings-screen .settings-slider__track {
  position: relative;
  flex: 1;
  height: 6px;
  background: rgba(3, 5, 12, 0.72);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 0;
  background-image: repeating-linear-gradient(
    90deg,
    rgba(255, 255, 255, 0.14) 0 1px,
    transparent 1px 25%
  );
}

.settings-screen .settings-slider__fill {
  height: 100%;
  background: var(--shell-neon, #38bdf8);
  border-radius: 0;
  opacity: 0.85;
}

.settings-screen .settings-slider__val {
  width: 32px;
  text-align: right;
  font-family: var(--font-mono-data, "IBM Plex Mono", monospace);
  font-weight: 700;
  color: var(--shell-text, #eef0f5);
  font-variant-numeric: tabular-nums;
  font-size: var(--type-compact-size, 0.8125rem);
}

.settings-scale-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
}
.settings-scale-btn {
  min-height: 34px;
  padding: 0 14px;
  border-radius: 6px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(3,5,12,0.55);
  color: var(--shell-text, #eef0f5);
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 11px;
  cursor: pointer;
}
.settings-scale-btn.is-active {
  border-color: var(--shell-neon, #38bdf8);
  color: #041018;
  background: var(--shell-neon, #38bdf8);
}
.settings-screen .settings-note {
  margin: var(--space-2, 8px) 0 0;
  font-family: var(--font-shell-body, "IBM Plex Sans", sans-serif);
  font-size: var(--type-table-meta-size, 0.72rem);
  line-height: 1.4;
  color: var(--shell-text-muted, #8b93a8);
}

@media (prefers-reduced-motion: reduce) {
  .settings-screen .settings-row {
    transition: none !important;
  }
}
`;
