import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import PlayerHeadshot from "../../components/PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";
import { submitDecision } from "../../services/franchiseService";

/*
 * A sit-down with a veteran who is thinking about retiring.
 * He says why; you make a pitch; he answers. Two failed pitches and his mind is made up.
 * The server owns the outcome (services/retirement_talk.py); this just shows the conversation.
 */

const TYPE_LABEL = {
  injury_retirement: "Health",
  age_retirement: "Age",
  decline_retirement: "Slowing down",
  goalie_longevity_retirement: "Age",
  family_retirement: "Family",
  role_frustration_retirement: "His role",
  morale_retirement: "Lost the fire",
  cup_win_walkaway: "Going out on top",
  legend_farewell: "Farewell tour",
  no_contract_retirement: "No contract",
};

export function isRetirementDecision(d) {
  return d && String(d.kind || d.type || "") === "retirement_decision";
}

function findRosterPlayer(franchiseState, pid) {
  if (!pid) return null;
  const pools = [franchiseState?.roster, franchiseState?.ahl_roster, franchiseState?.team?.roster];
  for (const pool of pools) {
    if (!Array.isArray(pool)) continue;
    const hit = pool.find((p) => String(p?.player_id || p?.id || "") === String(pid));
    if (hit) return hit;
  }
  return null;
}

function riskWord(r) {
  if (r >= 80) return "Mind nearly made up";
  if (r >= 60) return "Leaning toward retiring";
  if (r >= 40) return "Torn";
  return "Open to staying";
}

export default function RetirementTalk({ decision, talks = {}, franchiseState, mergeFranchiseState, onClose }) {
  const meta = decision?.meta || {};
  const pid = String(meta.player_id || decision?.player_id || "");
  const name = meta.player_name || decision?.player_name || "Your veteran";
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  // Once the server resolves the talk the decision disappears; keep showing the finished conversation.
  const [done, setDone] = useState(null);
  const [live, setLive] = useState(decision);
  const logRef = useRef(null);

  useEffect(() => {
    if (decision && !done) setLive(decision);
  }, [decision, done]);

  const lines = done ? done.talk || [] : (live?.meta?.talk || []);
  const options = done ? [] : (live?.options || []);
  const declines = Number(live?.meta?.declines || 0);
  const maxDeclines = Number(live?.meta?.max_declines || 2);
  const risk = Number(meta.retirement_risk || 0);
  const roster = findRosterPlayer(franchiseState, pid);
  const face = useMemo(
    () => ensurePlayerHeadshotFields({ ...(roster || {}), id: pid, player_id: pid, full_name: name, name, age: meta.age }),
    [roster, pid, name, meta.age]
  );

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines.length]);

  const pitch = async (choiceId) => {
    if (busy || !live) return;
    setBusy(true);
    setError("");
    try {
      const res = await submitDecision(live.id, choiceId);
      const state = res?.state || {};
      const again = (state.pending_decisions || []).find((d) => d && d.id === live.id);
      if (again) {
        setLive(again);
      } else {
        const finished = (state.retirement_talks || talks || {})[pid];
        setDone(finished || { talk: lines, outcome: choiceId === "let_retire" ? "retired" : "returning" });
      }
      mergeFranchiseState?.(state);
    } catch (e) {
      setError(e?.response?.data?.detail || e?.message || "That didn't go through. Try again.");
    } finally {
      setBusy(false);
    }
  };

  const stats = [
    meta.age ? `Age ${meta.age}` : null,
    meta.position || null,
    meta.seasons_played ? `${meta.seasons_played} seasons` : null,
    meta.games_played ? `${meta.games_played} GP` : null,
    meta.position === "G"
      ? (meta.goalie_wins != null ? `${meta.goalie_wins} wins` : null)
      : (meta.points != null ? `${meta.points} pts` : null),
    meta.cups_won ? `${meta.cups_won}× Cup` : null,
  ].filter(Boolean);

  const outcome = done?.outcome;

  const view = (
    <div className="rtk-shade" role="dialog" aria-modal="true" aria-label={`Retirement talk with ${name}`}>
      <div className={`rtk-sheet${outcome ? ` is-${outcome}` : ""}`}>
        <aside className="rtk-who">
          <span className="rtk-sup">Closed-door meeting</span>
          <div className="rtk-face">
            <PlayerHeadshot player={face} size="lg" preferPhoto showFlag={false} style={{ "--size": "200px" }} />
          </div>
          <div className="rtk-name">{name}</div>
          <div className="rtk-line">{stats.join(" · ")}</div>
          {meta.retirement_type ? (
            <div className="rtk-why">
              <span>On his mind</span>
              <strong>{TYPE_LABEL[meta.retirement_type] || "His future"}</strong>
            </div>
          ) : null}
          {risk ? (
            <div className="rtk-risk">
              <span>{riskWord(risk)}</span>
              <div className="rtk-risk-bar"><i style={{ width: `${Math.min(100, risk)}%` }} /></div>
            </div>
          ) : null}
          {!outcome ? (
            <div className="rtk-patience" aria-label={`${maxDeclines - declines} pitches left`}>
              <span>His patience</span>
              <div>
                {Array.from({ length: maxDeclines }).map((_, i) => (
                  <i key={i} className={i < maxDeclines - declines ? "is-left" : "is-used"} />
                ))}
              </div>
            </div>
          ) : null}
        </aside>

        <section className="rtk-room">
          <div className="rtk-head">{outcome === "returning" ? `${name.split(" ").slice(-1)[0]} is coming back` : outcome === "retired" ? `${name.split(" ").slice(-1)[0]} retires` : "His future"}</div>
          <div className="rtk-log" ref={logRef}>
            {lines.map((l, i) => (
              <div key={i} className={`rtk-say is-${l.speaker === "gm" ? "gm" : "player"}`} style={{ animationDelay: `${Math.min(i, 3) * 60}ms` }}>
                <span className="rtk-who-tag">{l.speaker === "gm" ? "You" : name.split(" ").slice(-1)[0]}</span>
                <p>{l.text}</p>
              </div>
            ))}
            {busy ? (
              <div className="rtk-say is-player is-typing"><span className="rtk-who-tag">{name.split(" ").slice(-1)[0]}</span><p><i /><i /><i /></p></div>
            ) : null}
          </div>

          {error ? <p className="rtk-error">{error}</p> : null}

          {outcome ? (
            <div className="rtk-end">
              <span className={`rtk-stamp is-${outcome}`}>{outcome === "returning" ? "Returning" : "Retired"}</span>
              <button type="button" className="rtk-go" onClick={onClose}>Done</button>
            </div>
          ) : (
            <div className="rtk-pitches">
              {options.map((o) => (
                <button
                  key={o.id}
                  type="button"
                  className={`rtk-pitch${o.id === "let_retire" ? " is-let" : ""}`}
                  disabled={busy}
                  onClick={() => pitch(o.id)}
                >
                  <strong>{o.label}</strong>
                  <span>{o.effect_summary || o.description}</span>
                </button>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
  return typeof document !== "undefined" ? createPortal(view, document.body) : view;
}
