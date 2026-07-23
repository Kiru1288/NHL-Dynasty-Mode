/**
 * TEMP — delete this file (+ App.js hook) when menu QA is done.
 *
 * Usage: open the app with ?replayEvents=1
 *   http://localhost:3000/?replayEvents=1
 *
 * Each full page refresh advances to the next cinematic event menu.
 * Arrow keys / on-screen buttons also step without refreshing.
 * Press R or click "Reset" to go back to menu 1.
 */
import React, { useCallback, useEffect, useMemo, useState } from "react";
import EventRouter from "../events/EventRouter";
import "../events/EventRegistry";
import { getEventRegistration } from "../events/EventRegistry";

const STORAGE_KEY = "nhl_event_menu_replay_idx";

/** Order matches the franchise offseason chain + playoffs kickoff. */
const REPLAY_KEYS = [
  "playoffs_start",
  "awards",
  "retirements",
  "salary_cap",
  "development_report",
  "draft_lottery",
  "draft",
  "re_sign",
  "free_agency",
  "roster_cleanup",
  "next_season_reveal",
];

/** Empty shell — menus must show fallback copy, not invented NHL data. */
const EMPTY_FRANCHISE_STATE = {
  season_year: 2025,
  user_team_id: "",
  phase: "offseason",
  season_phase: "offseason",
  team: {},
  flags: { can_generate_next_season: true },
};

function readStartIndex() {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    const n = Number(raw);
    return Number.isFinite(n) ? Math.max(0, Math.min(n, REPLAY_KEYS.length - 1)) : 0;
  } catch {
    return 0;
  }
}

function writeIndex(idx) {
  try {
    sessionStorage.setItem(STORAGE_KEY, String(idx));
  } catch {
    /* ignore */
  }
}

function advanceIndexForNextRefresh(current) {
  writeIndex((current + 1) % REPLAY_KEYS.length);
}

const HUD_CSS = `
.emr-hud {
  position: fixed;
  top: 12px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 99999;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  border-radius: 999px;
  background: rgba(2, 6, 23, 0.92);
  border: 1px solid rgba(148, 163, 184, 0.35);
  color: #e2e8f0;
  font: 600 12px/1 Inter, system-ui, sans-serif;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
  pointer-events: auto;
}
.emr-hud kbd {
  padding: 2px 6px;
  border-radius: 4px;
  background: rgba(30, 41, 59, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.25);
  font-size: 11px;
}
.emr-hud button {
  cursor: pointer;
  border: 1px solid rgba(148, 163, 184, 0.3);
  background: rgba(30, 41, 59, 0.85);
  color: #f8fafc;
  border-radius: 999px;
  padding: 4px 10px;
  font-weight: 700;
  font-size: 11px;
}
.emr-hud button:hover { border-color: rgba(147, 197, 253, 0.5); }
.emr-hud .emr-title { color: #93c5fd; font-weight: 800; letter-spacing: 0.06em; }
.emr-hud .emr-count { color: #94a3b8; }
`;

export default function EventMenuReplay() {
  const [index, setIndex] = useState(readStartIndex);

  useEffect(() => {
    advanceIndexForNextRefresh(index);
  }, [index]);

  const typeKey = REPLAY_KEYS[index] || REPLAY_KEYS[0];
  const registration = getEventRegistration(typeKey);
  const title = registration?.title || typeKey;

  const go = useCallback((delta) => {
    setIndex((i) => {
      const next = (i + delta + REPLAY_KEYS.length) % REPLAY_KEYS.length;
      writeIndex(next);
      return next;
    });
  }, []);

  const reset = useCallback(() => {
    writeIndex(0);
    setIndex(0);
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        go(1);
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        go(-1);
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        reset();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, reset]);

  const noop = useCallback(() => {
    go(1);
  }, [go]);

  const eventData = useMemo(() => {
    if (registration?.getEventData) {
      return registration.getEventData(EMPTY_FRANCHISE_STATE);
    }
    return {};
  }, [registration, typeKey]);

  return (
    <>
      <style>{HUD_CSS}</style>
      <div className="emr-hud" role="toolbar" aria-label="Event menu replay controls">
        <span className="emr-title">REPLAY QA</span>
        <span className="emr-count">
          {index + 1}/{REPLAY_KEYS.length} — {title}
        </span>
        <button type="button" onClick={() => go(-1)} aria-label="Previous menu">
          ← Prev
        </button>
        <button type="button" onClick={() => go(1)} aria-label="Next menu">
          Next →
        </button>
        <button type="button" onClick={reset}>
          Reset
        </button>
        <span style={{ color: "#64748b", fontSize: 11 }}>
          Refresh = next · <kbd>←</kbd>/<kbd>→</kbd>
        </span>
      </div>

      <EventRouter
        typeKey={typeKey}
        franchiseState={EMPTY_FRANCHISE_STATE}
        eventData={eventData}
        onContinue={noop}
        onBack={noop}
        onClose={noop}
        onEnterPlayoffs={noop}
        playoffData={eventData}
      />
    </>
  );
}
