/**
 * Arcade box score pieces (Double Dash / pixel-scoreboard look).
 *
 * Used by BoxScoreView: a big scoreboard with team logos, a period-by-period
 * scoreboard strip, chunky per-period shot / xG bars with a momentum line,
 * tug-of-war team stat bars, three-stars cards and goal cards with faces.
 * Pure SVG + CSS; consumes the existing box-score payload unchanged.
 */

import React from "react";
import PlayerHeadshot from "../PlayerHeadshot";
import TeamLogoBadge from "../ui/TeamLogoBadge";
import { resolveFranchiseTeamLogo } from "../../utils/teamLogos";

const TEAM_COLORS = {
  ANA: ["#f47a20", "#b9975b"],
  ARI: ["#8c2633", "#e2d6b5"],
  BOS: ["#ffb81c", "#111111"],
  BUF: ["#003087", "#ffb81c"],
  CAR: ["#cc0000", "#a2aaad"],
  CBJ: ["#002654", "#ce1126"],
  CGY: ["#c8102e", "#f1be48"],
  CHI: ["#cf0a2c", "#ff671b"],
  COL: ["#6f263d", "#236192"],
  DAL: ["#006847", "#8f8f8c"],
  DET: ["#ce1126", "#ffffff"],
  EDM: ["#ff4c00", "#041e42"],
  FLA: ["#c8102e", "#b9975b"],
  LAK: ["#a2aaad", "#111111"],
  MIN: ["#154734", "#a6192e"],
  MTL: ["#af1e2d", "#192168"],
  NJD: ["#ce1126", "#000000"],
  NSH: ["#ffb81c", "#041e42"],
  NYI: ["#00539b", "#f47d30"],
  NYR: ["#0038a8", "#ce1126"],
  OTT: ["#c52032", "#c2912c"],
  PHI: ["#f74902", "#000000"],
  PIT: ["#fcb514", "#000000"],
  SEA: ["#99d9d9", "#e9072b"],
  SJS: ["#006d75", "#ea7200"],
  STL: ["#002f87", "#fcb514"],
  TBL: ["#002868", "#ffffff"],
  TOR: ["#00205b", "#ffffff"],
  UTA: ["#6cace4", "#010101"],
  VAN: ["#00843d", "#00205b"],
  VGK: ["#b4975a", "#333f42"],
  WPG: ["#041e42", "#7b303e"],
  WSH: ["#c8102e", "#041e42"],
};
const FALLBACK = [
  ["#2f6fdf", "#ffd23f"],
  ["#d1335b", "#ffffff"],
  ["#1f9e6e", "#f4f1de"],
  ["#7a3cc8", "#ffcf56"],
  ["#e0761f", "#14213d"],
];

function hash(s) {
  let h = 0;
  const str = String(s || "");
  for (let i = 0; i < str.length; i += 1) h = (h * 31 + str.charCodeAt(i)) >>> 0;
  return h;
}

export function teamPalette(team = {}) {
  const code = String(team.abbr || team.id || "").toUpperCase();
  const hit = TEAM_COLORS[code];
  const [primary, secondary] = hit || FALLBACK[hash(code || team.name) % FALLBACK.length];
  return { primary, secondary, accent: secondary };
}

function luminance(hex) {
  const n = parseInt(String(hex).replace("#", ""), 16);
  if (!Number.isFinite(n)) return 0.3;
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

/** A bar colour that stays visible on the dark sheet. */
function barColor(pal) {
  return luminance(pal.primary) < 0.2 ? pal.secondary : pal.primary;
}

const num = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
};

function teamLogo(team) {
  return resolveFranchiseTeamLogo({
    team_id: team.id,
    id: team.id,
    abbrev: team.abbr,
    abbreviation: team.abbr,
    name: team.name,
  });
}

function portraitFor(box, pid, name, teamId) {
  const extra = (box.portraits || {})[String(pid)] || {};
  return { ...extra, id: pid, player_id: pid, name, team_id: teamId };
}

/* ------------------------------------------------------------------ */

export function ArcadeScoreboard({ box }) {
  const { home, away } = box;
  const result = box.shootout ? "FINAL/SO" : box.overtime ? "FINAL/OT" : "FINAL";
  const side = (t, other, where) => {
    const pal = teamPalette(t);
    const won = num(t.goals) > num(other.goals);
    return (
      <div
        className={`bxa-board__team is-${where}${won ? " is-winner" : ""}`}
        style={{ "--bxa-team": pal.primary, "--bxa-team2": pal.secondary }}
      >
        <span className="bxa-board__logo">
          <TeamLogoBadge teamLogo={teamLogo(t)} teamName={t.name || t.abbr} size={92} variant="badge" />
        </span>
        <span className="bxa-board__names">
          <b>{t.abbr || t.id}</b>
          <small>{t.name}</small>
          <em>{where === "away" ? "AWAY" : "HOME"}</em>
        </span>
        <strong className="bxa-board__score" aria-label={`${t.name} ${t.goals}`}>
          {t.goals}
        </strong>
        {won ? <span className="bxa-board__win">WIN</span> : null}
      </div>
    );
  };
  return (
    <section className="bxa-board" aria-label="Final score">
      {side(away, home, "away")}
      <div className="bxa-board__mid">
        <span className="bxa-board__result">{result}</span>
        <span className="bxa-board__dash">-</span>
        {box.date ? <span className="bxa-board__date">{box.date}</span> : null}
        {box.scope && box.scope !== "regular_season" ? (
          <span className="bxa-board__scope">{String(box.scope).replace(/_/g, " ")}</span>
        ) : null}
      </div>
      {side(home, away, "home")}
    </section>
  );
}

export function PeriodStrip({ box }) {
  const { home, away } = box;
  const labels = box.period_labels || ["1st", "2nd", "3rd"];
  const cols = labels.length + 2;
  return (
    <section className="bxa-strip" style={{ "--bxa-cols": cols }} aria-label="Scoring by period">
      <div className="bxa-strip__grid">
        <span className="bxa-strip__corner">PER</span>
        {labels.map((l) => (
          <span key={l} className="bxa-strip__head">{l}</span>
        ))}
        <span className="bxa-strip__head is-total">T</span>
        {[away, home].map((t) => {
          const pal = teamPalette(t);
          return (
            <React.Fragment key={t.id}>
              <span className="bxa-strip__team" style={{ "--bxa-team": pal.primary }}>
                {t.abbr || t.id}
              </span>
              {labels.map((l, i) => {
                const g = num(t.periods?.[i]);
                return (
                  <span key={l} className={`bxa-strip__cell${g > 0 ? " is-lit" : ""}`}>
                    {g}
                    <small>{num(t.shots?.[i])} SOG</small>
                  </span>
                );
              })}
              <span className="bxa-strip__cell is-total">
                {t.goals}
                <small>{t.sog} SOG</small>
              </span>
            </React.Fragment>
          );
        })}
      </div>
    </section>
  );
}

/** Per-period shot bars + estimated xG bars + cumulative shot momentum line. */
export function ShotBattle({ box }) {
  const { home, away } = box;
  const labels = box.period_labels || ["1st", "2nd", "3rd"];
  const pa = teamPalette(away);
  const ph = teamPalette(home);
  const ca = barColor(pa);
  const chRaw = barColor(ph);
  const ch = chRaw.toLowerCase() === ca.toLowerCase() ? ph.secondary : chRaw;
  const shotsA = labels.map((_, i) => num(away.shots?.[i]));
  const shotsH = labels.map((_, i) => num(home.shots?.[i]));
  const sogA = Math.max(1, num(away.sog));
  const sogH = Math.max(1, num(home.sog));
  // xG is stored per game; split it by each period's shot share (labelled as an estimate).
  const xgA = shotsA.map((s) => (num(away.xg) * s) / sogA);
  const xgH = shotsH.map((s) => (num(home.xg) * s) / sogH);
  const maxShots = Math.max(4, ...shotsA, ...shotsH);
  const maxXg = Math.max(0.5, ...xgA, ...xgH);

  // Momentum: running shot differential (away - home) at the end of each period.
  const diffs = [0];
  labels.forEach((_, i) => diffs.push(diffs[diffs.length - 1] + shotsA[i] - shotsH[i]));
  const maxAbs = Math.max(3, ...diffs.map((d) => Math.abs(d)));
  const W = 300;
  const H = 96;
  const step = W / Math.max(1, labels.length);
  const y = (d) => H / 2 - (d / maxAbs) * (H / 2 - 8);
  let path = `M0 ${y(0)}`;
  diffs.slice(1).forEach((d, i) => {
    path += ` H${(i + 1) * step - step / 2} V${y(d)}`;
  });
  path += ` H${W}`;
  const goalsByPeriod = labels.map((_, i) =>
    (box.goals || []).filter((g) => g && Number(g.period) === i + 1 && g.scorer)
  );

  const Bars = ({ a, h, max, fmt, unit }) => (
    <div className="bxa-bars">
      {labels.map((l, i) => (
        <div key={l} className="bxa-bars__group">
          <div className="bxa-bars__cols">
            <span className="bxa-bars__col" style={{ "--h": `${Math.max(4, (a[i] / max) * 100)}%`, "--c": ca }}>
              <b>{fmt(a[i])}</b>
            </span>
            <span className="bxa-bars__col" style={{ "--h": `${Math.max(4, (h[i] / max) * 100)}%`, "--c": ch }}>
              <b>{fmt(h[i])}</b>
            </span>
          </div>
          <span className="bxa-bars__label">{l}</span>
        </div>
      ))}
      <span className="bxa-bars__unit">{unit}</span>
    </div>
  );

  return (
    <section className="bxa-card bxa-battle">
      <header className="bxa-card__head">
        <h4>Shot battle</h4>
        <span className="bxa-legend">
          <i style={{ background: ca }} /> {away.abbr || away.name}
          <i style={{ background: ch }} /> {home.abbr || home.name}
        </span>
      </header>
      <div className="bxa-battle__grid">
        <div>
          <p className="bxa-mini">Shots on goal</p>
          <Bars a={shotsA} h={shotsH} max={maxShots} fmt={(v) => v} unit="SOG" />
        </div>
        <div>
          <p className="bxa-mini">xG (by period, est.)</p>
          <Bars a={xgA} h={xgH} max={maxXg} fmt={(v) => v.toFixed(1)} unit="xG" />
        </div>
      </div>
      <div className="bxa-momentum">
        <p className="bxa-mini">Momentum (running shot edge)</p>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="bxa-momentum__svg" aria-hidden="true">
          <rect x="0" y="0" width={W} height={H / 2} fill={ca} opacity="0.14" />
          <rect x="0" y={H / 2} width={W} height={H / 2} fill={ch} opacity="0.14" />
          {labels.map((l, i) => (
            <line key={l} x1={(i + 1) * step} x2={(i + 1) * step} y1="0" y2={H} stroke="#ffffff" strokeOpacity="0.12" strokeWidth="2" strokeDasharray="4 4" />
          ))}
          <line x1="0" x2={W} y1={H / 2} y2={H / 2} stroke="#ffffff" strokeOpacity="0.35" strokeWidth="2" />
          <path d={path} fill="none" stroke="#071233" strokeWidth="9" strokeLinejoin="miter" />
          <path d={path} fill="none" stroke="#ffe23a" strokeWidth="4.5" strokeLinejoin="miter" />
          {goalsByPeriod.map((gs, i) =>
            gs.map((g, k) => {
              const isAway = String(g.team_id) === String(away.id);
              const cx = i * step + ((k + 1) * step) / (gs.length + 1);
              return (
                <rect
                  key={`${i}-${k}`}
                  x={cx - 6}
                  y={isAway ? 4 : H - 16}
                  width="12"
                  height="12"
                  fill={isAway ? ca : ch}
                  stroke="#071233"
                  strokeWidth="3"
                />
              );
            })
          )}
        </svg>
        <div className="bxa-momentum__labels">
          <span>{away.abbr || away.name} pressing</span>
          <span>{home.abbr || home.name} pressing</span>
        </div>
      </div>
    </section>
  );
}

/** Head-to-head stat bars (tug of war). rows: [label, awayValue, homeValue, awayNum?, homeNum?] */
export function StatTugOfWar({ box, rows }) {
  const { home, away } = box;
  const ca = barColor(teamPalette(away));
  let ch = barColor(teamPalette(home));
  if (ch.toLowerCase() === ca.toLowerCase()) ch = teamPalette(home).secondary;
  return (
    <section className="bxa-card bxa-tug">
      <header className="bxa-card__head">
        <h4>Team stats</h4>
        <span className="bxa-legend">
          <i style={{ background: ca }} /> {away.abbr || away.name}
          <i style={{ background: ch }} /> {home.abbr || home.name}
        </span>
      </header>
      {rows.map(([label, a, h, an, hn]) => {
        const x = Number.isFinite(an) ? an : parseFloat(a);
        const z = Number.isFinite(hn) ? hn : parseFloat(h);
        const tot = (Number.isFinite(x) ? x : 0) + (Number.isFinite(z) ? z : 0);
        const share = tot > 0 ? (x / tot) * 100 : 50;
        return (
          <div key={label} className="bxa-tug__row">
            <b className={x > z ? "is-lead" : ""}>{a}</b>
            <div className="bxa-tug__bar" style={{ "--a": ca, "--h": ch, "--s": `${share}%` }}>
              <span className="bxa-tug__label">{label}</span>
            </div>
            <b className={z > x ? "is-lead" : ""}>{h}</b>
          </div>
        );
      })}
    </section>
  );
}

export function ThreeStars({ box, teamName }) {
  if (!box.stars?.length) return null;
  const order = [...box.stars].sort((a, b) => a.rank - b.rank);
  return (
    <section className="bxa-card bxa-stars">
      <header className="bxa-card__head">
        <h4>Three stars</h4>
      </header>
      <div className="bxa-stars__row">
        {order.map((s) => {
          const team = String(s.team_id) === String(box.home.id) ? box.home : box.away;
          const pal = teamPalette(team);
          return (
            <article
              key={s.player_id}
              className={`bxa-star is-rank-${s.rank}`}
              style={{ "--bxa-team": pal.primary, "--bxa-team2": pal.secondary }}
            >
              <span className="bxa-star__rank">
                {"★".repeat(s.rank)}
                <small>{s.rank === 1 ? "1ST STAR" : s.rank === 2 ? "2ND STAR" : "3RD STAR"}</small>
              </span>
              <PlayerHeadshot
                player={portraitFor(box, s.player_id, s.name, s.team_id)}
                size="lg"
                className="bxa-star__face"
                teamColors={pal}
                showFlag={false}
              />
              <strong>{s.name}</strong>
              <span className="bxa-star__team">{teamName(s.team_id)}</span>
              <em>{s.line}</em>
            </article>
          );
        })}
      </div>
    </section>
  );
}

const ODDITY = {
  goalie_goal: "GOALIE GOAL",
  penalty_shot: "PENALTY SHOT",
  fluke_ref_skate: "OFF REF'S SKATE",
  fluke_broken_stick: "BROKEN STICK",
  fluke_boards: "BOARDS BOUNCE",
};

export function GoalCards({ box, byPeriod, total }) {
  return (
    <section className="bxa-card bxa-goals">
      <header className="bxa-card__head">
        <h4>Scoring summary</h4>
        <span className="bxa-count">
          {total} goal{total === 1 ? "" : "s"}
        </span>
      </header>
      {!box.event_detail ? <p className="bxs-note">Simmed in bulk: scorers are recorded, goal times are not.</p> : null}
      {Object.keys(byPeriod).length === 0 ? <p className="bxs-empty">No goals.</p> : null}
      {Object.entries(byPeriod).map(([period, goals]) => (
        <div key={period} className="bxa-goals__period">
          <div className="bxa-goals__ptag">{period}</div>
          <div className="bxa-goals__list">
            {goals.map((g, i) => {
              const team = String(g.team_id) === String(box.home.id) ? box.home : box.away;
              const pal = teamPalette(team);
              const tags = [];
              if (g.strength === "PP") tags.push(["PPG", "is-pp"]);
              if (g.strength === "SH") tags.push(["SHG", "is-sh"]);
              if (g.strength && !["EV", "PP", "SH", "OT"].includes(g.strength)) tags.push([g.strength, ""]);
              if (g.empty_net) tags.push(["EN", ""]);
              if (g.overtime) tags.push(["OT", ""]);
              if (g.gwg) tags.push(["GWG", "is-gold"]);
              if (g.tag) tags.push([ODDITY[g.tag] || "FLUKE", "is-gold"]);
              return (
                <article
                  key={i}
                  className="bxa-goal"
                  style={{ "--bxa-team": pal.primary, "--bxa-team2": pal.secondary }}
                >
                  <PlayerHeadshot
                    player={portraitFor(box, g.scorer?.id, g.scorer?.name, g.team_id)}
                    size="sm"
                    className="bxa-goal__face"
                    teamColors={pal}
                    showFlag={false}
                  />
                  <div className="bxa-goal__body">
                    <div className="bxa-goal__top">
                      <span className="bxa-goal__logo">
                        <TeamLogoBadge teamLogo={teamLogo(team)} teamName={team.name || team.abbr} size={26} variant="badge" />
                      </span>
                      <strong>{g.scorer?.name}</strong>
                    </div>
                    <small>
                      {g.assists?.length ? `Assists: ${g.assists.map((a) => a.name).join(", ")}` : "Unassisted"}
                    </small>
                    {tags.length ? (
                      <span className="bxa-goal__tags">
                        {tags.map(([t, c]) => (
                          <em key={t} className={`bxs-tag ${c}`}>{t}</em>
                        ))}
                      </span>
                    ) : null}
                  </div>
                  <div className="bxa-goal__side">
                    <span className="bxa-goal__clock">{g.clock || "--:--"}</span>
                    {g.score ? <span className="bxa-goal__score">{g.score}</span> : null}
                  </div>
                </article>
              );
            })}
          </div>
        </div>
      ))}
    </section>
  );
}
