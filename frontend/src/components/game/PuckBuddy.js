import React, { useEffect, useMemo, useState } from "react";
import { useGameUI } from "../../game/GameUIContext";
import { SCREENS } from "../../game/constants";

/** The pixel puck. One element; the art lives in game.css (.pucky). */
export function Pucky({ size = "", talking = false, bobbing = false, className = "", title }) {
  const cls = ["pucky", size ? `is-${size}` : "", talking ? "is-talking" : "", bobbing ? "is-bobbing" : "", className]
    .filter(Boolean)
    .join(" ");
  return (
    <span className={cls} role={title ? "img" : undefined} aria-label={title} aria-hidden={title ? undefined : true}>
      <i />
    </span>
  );
}

// What Pucky says on each screen: where you are and the one thing worth doing here.
const TIPS = {
  [SCREENS.HUB]: "Welcome back, boss! Advance the day from the Calendar, or poke around the office.",
  [SCREENS.OFFICE]: "Your office. Every desk is a door — click one to jump in.",
  [SCREENS.CALENDAR]: "Big logos are your games. Click a day to preview it, then hit Advance Day.",
  [SCREENS.ROSTER]: "Your whole organization lives here. Click a player for his full file.",
  [SCREENS.EDIT_LINES]: "Drag players onto the ice. Ice time has to add up to 100% before you can leave!",
  [SCREENS.POWER_PLAY]: "Power play units: shooters on the flanks, a big body at the net.",
  [SCREENS.PENALTY_KILL]: "PK units: your best defensive forwards and shot-blockers.",
  [SCREENS.STATS]: "Stats Central. Sort any column; the Expected tab shows who's really driving play.",
  [SCREENS.TRADE]: "Trade floor. Build a package, check the read, then propose.",
  [SCREENS.DRAFT_CLASS]: "Draft board. Scout more to sharpen the fog on a prospect's ratings.",
  [SCREENS.DRAFT_LOTTERY]: "Lottery night! Fingers crossed for a jump.",
  [SCREENS.TEAM_NEEDS]: "Team needs: the holes in your depth chart, worst first.",
  [SCREENS.SCOUTING]: "Send scouts where your draft board is foggiest.",
  [SCREENS.CHEMISTRY]: "Chemistry: who plays well together. Pairs that click go green.",
  [SCREENS.STORYLINES]: "Locker room drama and league gossip. Some choices have consequences!",
  [SCREENS.CAP_LEDGER]: "Cap ledger. Watch next summer's room before you hand out raises.",
  [SCREENS.FREE_AGENCY]: "Free agency! Fit matters: city, role and a contender all sway players.",
  [SCREENS.LEAGUE_OPERATIONS]: "League ops: rules, waivers and what the other GMs are up to.",
  [SCREENS.GM_WORLD]: "League ops: rules, waivers and what the other GMs are up to.",
  [SCREENS.FRANCHISE_IDENTITY]: "Your franchise's identity: colors, history and what you stand for.",
  [SCREENS.LEGACY_HALL]: "The Legacy Hall. Banners, legends and your best seasons.",
  [SCREENS.SETTINGS]: "Settings. Tweak the sim and the screen to your taste.",
  [SCREENS.AHL_CENTER]: "Your farm team. Call-ups and send-downs start here.",
  [SCREENS.PROSPECT_SIGNING]: "Sign your draft picks before their rights run out!",
};

const QUICK = [
  { id: SCREENS.HUB, label: "Hub" },
  { id: SCREENS.CALENDAR, label: "Calendar" },
  { id: SCREENS.EDIT_LINES, label: "Lines" },
  { id: SCREENS.ROSTER, label: "Roster" },
  { id: SCREENS.STATS, label: "Stats" },
  { id: SCREENS.TRADE, label: "Trades" },
];

const HIDE_ON = new Set([SCREENS.SETUP, SCREENS.PUCKGPT]);
const STORE_KEY = "puckbuddy:collapsed";

/** Pucky, the on-screen guide: a tip for the screen you're on and quick jumps. */
export default function PuckBuddy() {
  const { screen, setScreen } = useGameUI();
  const [open, setOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(STORE_KEY) === "1";
    } catch {
      return false;
    }
  });
  const tip = useMemo(() => TIPS[screen] || "Need a hand? Ask me anything about this save.", [screen]);

  // A new screen: he pipes up with that screen's tip for a few seconds.
  const [chatty, setChatty] = useState(false);
  useEffect(() => {
    if (collapsed || HIDE_ON.has(screen)) return undefined;
    setChatty(true);
    const t = setTimeout(() => setChatty(false), 2600);
    return () => clearTimeout(t);
  }, [screen, collapsed]);

  useEffect(() => {
    try {
      localStorage.setItem(STORE_KEY, collapsed ? "1" : "0");
    } catch {
      /* storage unavailable */
    }
  }, [collapsed]);

  if (HIDE_ON.has(screen)) return null;

  const go = (id) => {
    setOpen(false);
    if (typeof setScreen === "function") setScreen(id);
  };

  if (collapsed) {
    return (
      <button type="button" className="puckbuddy puckbuddy--mini" onClick={() => setCollapsed(false)} title="Bring Pucky back">
        <Pucky size="sm" />
      </button>
    );
  }

  return (
    <div className={`puckbuddy${open ? " is-open" : ""}`}>
      {open || chatty ? (
        <div className="puckbuddy__bubble" role="status">
          <p>{tip}</p>
          {open ? (
            <>
              <div className="puckbuddy__nav">
                {QUICK.filter((q) => q.id !== screen).map((q) => (
                  <button key={q.id} type="button" className="puckbuddy__go" onClick={() => go(q.id)}>
                    {q.label}
                  </button>
                ))}
              </div>
              <div className="puckbuddy__row">
                <button type="button" className="puckbuddy__ask" onClick={() => go(SCREENS.PUCKGPT)}>
                  Ask PuckGPT
                </button>
                <button type="button" className="puckbuddy__hide" onClick={() => { setOpen(false); setCollapsed(true); }}>
                  Hide me
                </button>
              </div>
            </>
          ) : null}
        </div>
      ) : null}
      <button
        type="button"
        className="puckbuddy__me"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        title="Pucky — tips and quick navigation"
      >
        <Pucky size="md" talking={chatty && !open} bobbing />
      </button>
    </div>
  );
}
