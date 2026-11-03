import React, { useEffect, useMemo, useState } from "react";
import PlayerHeadshot from "../PlayerHeadshot";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";

/* Awards race desk: one trophy on stage at a time with a narrative read, a
   top-5 race track (gap to the leader), rank movement since about a week ago
   and last-10 form, plus a wall of every ballot showing how tight it is. */

const arr = (v) => (Array.isArray(v) ? v : []);
const s = (v, f = "") => (v == null || v === "" ? f : String(v));
const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : null);

// Boards where the raw counting stat is the race (gap in that stat).
const COUNT_BOARDS = { art_ross: "PTS", rocket: "G", lady_byng: "PTS" };

function fmtMetric(m) {
  if (!m || m.value == null || m.value === "") return "—";
  const n = Number(m.value);
  if (!Number.isFinite(n)) return "—";
  if (m.kind === "pct") return `${(Math.abs(n) > 1.5 ? n : n * 100).toFixed(1)}%`;
  if (m.kind === "signed") return `${n > 0 ? "+" : ""}${n.toFixed(1)}`;
  return String(Math.round(n));
}
function metricOf(leader, label) {
  return arr(leader?.metrics).find((m) => s(m.label) === label) || null;
}
function lastName(name) {
  const parts = s(name, "Player").trim().split(/\s+/);
  return parts.length > 1 ? parts.slice(1).join(" ") : parts[0];
}

/** Gap info for each leader relative to #1. */
function raceGaps(board) {
  const leaders = arr(board?.leaders);
  if (!leaders.length) return [];
  const countLabel = COUNT_BOARDS[s(board.id)];
  const lead = leaders[0];
  const leadCount = countLabel ? num(metricOf(lead, countLabel)?.value) : null;
  const leadScore = num(lead.race_score);
  return leaders.map((l, i) => {
    if (i === 0) return { pct: 100, text: "Leader", back: 0 };
    if (countLabel && leadCount != null) {
      const v = num(metricOf(l, countLabel)?.value) ?? 0;
      const back = Math.max(0, leadCount - v);
      return { pct: leadCount > 0 ? Math.max(6, (v / leadCount) * 100) : 50, text: back ? `−${back} ${countLabel}` : `Tied on ${countLabel}`, back: leadCount > 0 ? back / leadCount : 0 };
    }
    const sc = num(l.race_score);
    if (leadScore != null && sc != null && leadScore !== 0) {
      const frac = Math.max(0, (leadScore - sc) / Math.abs(leadScore));
      return { pct: Math.max(6, 100 - frac * 100 * 2.5), text: frac < 0.005 ? "Dead even" : `${(frac * 100).toFixed(1)}% back`, back: frac };
    }
    return { pct: Math.max(10, 100 - i * 16), text: `#${i + 1}`, back: null };
  });
}

function suspense(board) {
  const gaps = raceGaps(board);
  if (gaps.length < 2 || gaps[1].back == null) return { key: "open", label: "Open ballot", note: "Too early to call." };
  const g2 = gaps[1].back;
  const g3 = gaps[2]?.back ?? 1;
  if (g2 < 0.02 && g3 < 0.05) return { key: "scrum", label: "Photo finish", note: "Three names inside a whisker." };
  if (g2 < 0.03) return { key: "duel", label: "Two-horse race", note: "One hot week flips it." };
  if (g2 < 0.08) return { key: "chase", label: "Chase is on", note: "The leader has a cushion, not a lock." };
  return { key: "runaway", label: "Runaway", note: "Somebody needs a monster stretch." };
}

function Movement({ leader }) {
  if (leader?.new_entry) return <span className="ard-move is-new" title="New to the top 5 since last week">NEW</span>;
  const d = num(leader?.rank_delta);
  if (d == null) return <span className="ard-move is-flat" title="No earlier snapshot yet">·</span>;
  if (d > 0) return <span className="ard-move is-up" title={`Up ${d} since last week`}>▲{d}</span>;
  if (d < 0) return <span className="ard-move is-down" title={`Down ${-d} since last week`}>▼{-d}</span>;
  return <span className="ard-move is-flat" title="Same spot as last week">=</span>;
}

function FormChip({ form }) {
  if (!form) return null;
  const t = s(form.trend, "steady");
  const text = form.kind === "goalie"
    ? `L${form.games}: ${form.recent_sv != null ? Number(form.recent_sv).toFixed(3).replace(/^0/, "") : "—"} SV%`
    : `L${form.games}: ${form.recent_g}G ${form.recent_a}A`;
  const icon = t === "hot" ? "🔥" : t === "cold" ? "❄" : "—";
  return <span className={`ard-form is-${t}`} title={form.kind === "goalie" ? `Season ${Number(form.season_sv).toFixed(3)}` : `Season ${form.season_ppg} P/GP vs ${form.recent_ppg} lately`}><i aria-hidden>{icon}</i>{text}</span>;
}

function narrative(board) {
  const leaders = arr(board?.leaders);
  if (!leaders.length) return { head: "No qualified names yet", body: "The ballot fills in as games hit the ledger." };
  const [a, b, c] = leaders;
  const gaps = raceGaps(board);
  const sus = suspense(board);
  const why = arr(a.metrics).slice(0, 3).map((m) => `${fmtMetric(m)} ${s(m.label)}`).join(", ");
  let head = `${s(a.name, "Player")} leads the ${s(board.name)}`;
  if (b && gaps[1]?.text && gaps[1].text !== "#2") head += ` — ${b.name ? lastName(b.name) : "the chaser"} ${gaps[1].text.replace("back", "behind")}`;
  const bits = [];
  if (why) bits.push(`${lastName(a.name)} is out front on ${why}.`);
  const hot = leaders.find((l, i) => i > 0 && l.form?.trend === "hot");
  const climber = leaders.find((l, i) => i > 0 && num(l.rank_delta) > 0);
  if (a.form?.trend === "cold") bits.push(`The leader has cooled off (${a.form.kind === "goalie" ? "save rate down" : `${a.form.recent_p} points in ${a.form.games}`}) — the door is open.`);
  else if (a.form?.trend === "hot") bits.push(`And he is getting hotter: ${a.form.kind === "goalie" ? "a wall lately" : `${a.form.recent_p} points in his last ${a.form.games}`}.`);
  if (climber) bits.push(`${lastName(climber.name)} is the mover, up ${climber.rank_delta} spot${climber.rank_delta > 1 ? "s" : ""} this week.`);
  else if (hot) bits.push(`Watch ${lastName(hot.name)}: on a heater with ${hot.form.kind === "goalie" ? "big saves" : `${hot.form.recent_p} points in ${hot.form.games}`}.`);
  if (!climber && !hot && c) bits.push(`${lastName(c.name)} rounds out the podium.`);
  bits.push(sus.note);
  return { head, body: bits.join(" ") };
}

export default function AwardsRaceDesk({ boards, status, sealed, sealedMessage, windowDays }) {
  const list = arr(boards).filter((b) => b && (b.leaders || b.sealed));
  const [active, setActive] = useState("");
  useEffect(() => {
    if (!list.length) return;
    if (!list.some((b) => s(b.id) === active)) setActive(s(list[0].id));
  }, [list, active]);
  const board = useMemo(() => list.find((b) => s(b.id) === active) || list[0], [list, active]);

  if (sealed) {
    return (
      <section className="ard">
        <header className="ccx-head ard-head">
          <div><span className="ccx-kicker">League desk</span><h2 className="ccx-title">Awards race</h2></div>
          <span className="ccx-stamp is-bad">Sealed</span>
        </header>
        <div className="ccx-panel ard-sealed">
          <h3>The envelopes are sealed</h3>
          <p>{sealedMessage || "The race is dark for the final 15 days. Whoever is leading when the season ends keeps the trophy."}</p>
        </div>
      </section>
    );
  }

  const gaps = board ? raceGaps(board) : [];
  const story = board ? narrative(board) : null;
  const sus = board ? suspense(board) : null;

  return (
    <section className="ard">
      <header className="ccx-head ard-head">
        <div>
          <span className="ccx-kicker">League desk</span>
          <h2 className="ccx-title">Awards race</h2>
          <p className="ard-lede">Who is leading, by how much, and who is coming. Movement compares to {windowDays ? `${windowDays} day${windowDays === 1 ? "" : "s"} ago` : "your last look"}; form is the last 10 games. The board goes dark for the final 15 days.</p>
        </div>
      </header>

      {status === "loading" && !list.length ? <p className="ard-status">Pulling the race from the awards ledger…</p> : null}
      {status === "error" ? <p className="ard-status">The awards ledger did not load. Advance a day and open this desk again.</p> : null}
      {status !== "loading" && status !== "error" && !list.length ? <p className="ard-status">No qualified players yet. The race fills in as games hit the ledger.</p> : null}

      {board ? (
        <div className="ard-layout">
          <nav className="ard-trophies" aria-label="Trophies">
            {list.map((b) => {
              const lead = arr(b.leaders)[0];
              const su = suspense(b);
              return (
                <button key={s(b.id)} type="button" className={`ard-trophy ${s(b.id) === s(board.id) ? "is-active" : ""} is-${su.key}`} onClick={() => setActive(s(b.id))} aria-pressed={s(b.id) === s(board.id)}>
                  <span className="ard-trophy__name">{s(b.name)}</span>
                  <span className="ard-trophy__lead">{lead ? lastName(lead.name) : "—"}</span>
                  <span className="ard-trophy__tag">{su.label}</span>
                </button>
              );
            })}
          </nav>

          <div className="ard-stage">
            <article className="ccx-panel ard-story">
              <div className="ard-story__top">
                <span className="ccx-kicker">{s(board.name)} · {s(board.blurb)}</span>
                {sus ? <span className={`ccx-stamp ard-sus is-${sus.key}`}>{sus.label}</span> : null}
              </div>
              <h3 className="ard-story__head">{story.head}</h3>
              <p className="ard-story__body">{story.body}</p>
            </article>

            <ol className="ard-track">
              {arr(board.leaders).map((l, i) => {
                const portrait = ensurePlayerHeadshotFields({ ...l, id: l.player_id, player_id: l.player_id });
                const g = gaps[i] || {};
                return (
                  <li key={`${board.id}-${l.player_id || l.name}-${i}`} className={`ard-lane ${i === 0 ? "is-leader" : ""}`}>
                    <span className="ard-rank">{i + 1}</span>
                    <span className="ard-face">
                      {l.position === "HC" ? <span className="ard-coach">HC</span> : <PlayerHeadshot player={portrait} size="xs" showFlag={false} />}
                    </span>
                    <span className="ard-who">
                      <b>{s(l.name, "Player")}</b>
                      <i>{s(l.team_abbr)}{l.position && l.position !== "HC" ? ` · ${l.position}` : ""}{l.gp ? ` · ${l.gp} GP` : ""}</i>
                    </span>
                    <span className="ard-bar" aria-hidden>
                      <span className="ard-bar__fill" style={{ width: `${Math.min(100, Math.max(0, g.pct || 0))}%` }} />
                    </span>
                    <span className="ard-gap">{g.text}</span>
                    <Movement leader={l} />
                    <span className="ard-stats">
                      {arr(l.metrics).slice(0, 4).map((m) => (
                        <span key={m.label}><small>{m.label}</small>{fmtMetric(m)}</span>
                      ))}
                    </span>
                    <FormChip form={l.form} />
                  </li>
                );
              })}
              {!arr(board.leaders).length ? <li className="ard-lane is-empty">No qualified names on this ballot yet.</li> : null}
            </ol>
          </div>
        </div>
      ) : null}
    </section>
  );
}
