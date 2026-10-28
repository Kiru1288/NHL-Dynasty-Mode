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
  "board_of_governors",
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
      {null}
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
