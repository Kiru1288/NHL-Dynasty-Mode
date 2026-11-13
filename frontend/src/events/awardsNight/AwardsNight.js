import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import PlayerHeadshot from "../../components/PlayerHeadshot";
import { pickHeadshotIdentityFields } from "../../utils/playerHeadshots";
// Theme + fonts: paste the same style imports EntryDraft.jsx has at its top here
// (the files that define --ops-*, --font-broadcast-display, --font-ops-ui, --font-mono-data).
import {
  buildAwardsCeremonySlides,
  buildAwardsFanTweets,
  buildAwardsNightSummary,
  buildCeremonyRailGroups,
  buildFallbackAwardFans,
  collectPlayerRows,
  countCalculationFallbackAwards,
  listAwardsMissingFinalists,
  mentionsWinner,
  normalizeAwardsPayload,
  SEASON_MILESTONES,
} from "./awardHelpers";

const PHASE = {
  GATE: "gate",
  TITLE: "title",
  FINALISTS: "finalists",
  SEAL: "seal",
  REVEAL: "reveal",
  CITATION: "citation",
  SUMMARY: "summary",
};

const TIMING = {
  [PHASE.TITLE]: 2600,
  [PHASE.FINALISTS]: 900,
  [PHASE.SEAL]: 1400,
  [PHASE.REVEAL]: 2400,
  [PHASE.CITATION]: 6500,
};

const PHASE_LABEL = {
  [PHASE.GATE]: "Pre-show",
  [PHASE.TITLE]: "On stage",
  [PHASE.FINALISTS]: "Finalists",
  [PHASE.SEAL]: "Ballots sealed",
  [PHASE.REVEAL]: "Winner",
  [PHASE.CITATION]: "The case",
  [PHASE.SUMMARY]: "Complete",
};

const METHOD_LABEL = {
  ballot: "Media ballot",
  stat_race: "Stat race",
  selection: "Selection",
  team: "Standings",
  playoff_series: "Playoffs",
};

const PLACEHOLDERS = new Set(["", "?", "—", "–", "-", "N/A", "n/a", "NaN", "null", "undefined"]);
const DEBUG_TEXT = /fallback|documented|missing advanced|calculated with/i;

/* ─────────────── pure helpers ─────────────── */

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function toNum(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function pad2(n) {
  return String(n).padStart(2, "0");
}

function seasonLabel(franchiseState) {
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${y}–${Number(y) + 1}` : "Season complete";
}

function ordinal(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "";
  const suffix = ["th", "st", "nd", "rd"];
  const m = n % 100;
  return `${n}${suffix[(m - 20) % 10] || suffix[m] || suffix[0]}`;
}

function formatPoints(n) {
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

function formatStat(value, fmt) {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") {
    const s = value.trim();
    if (PLACEHOLDERS.has(s)) return null;
    if (!Number.isFinite(Number(s))) return s;
  }
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  switch (fmt) {
    case "sv3":
      return n.toFixed(3).replace(/^0(?=\.)/, "");
    case "pct1":
      return `${(Math.abs(n) <= 1 ? n * 100 : n).toFixed(1)}%`;
    case "dec2":
      return n.toFixed(2);
    case "signed":
      return `${n > 0 ? "+" : ""}${Math.round(n)}`;
    case "toi": {
      let m = Math.floor(n);
      let s = Math.round((n - m) * 60);
      if (s === 60) {
        m += 1;
        s = 0;
      }
      return `${m}:${String(s).padStart(2, "0")}`;
    }
    case "int":
      return String(Math.round(n));
    default:
      return typeof value === "string" ? value.trim() : String(n);
  }
}

function cleanLines(lines) {
  const seen = new Set();
  return safeArray(lines)
    .map((line) => (typeof line === "string" ? line.trim() : ""))
    .filter((line) => line && !DEBUG_TEXT.test(line) && !seen.has(line) && seen.add(line));
}

function isRevealedPhase(phase) {
  return phase === PHASE.REVEAL || phase === PHASE.CITATION;
}

function slideEvidence(slide) {
  return (
    slide?.evidence ||
    slide?.award?.evidence ||
    slide?.award?.result?.evidence ||
    slide?.raw?.evidence ||
    null
  );
}

function isTeamSlide(slide) {
  return slide?.slideKind === "team" || slide?.awardKind === "team";
}

function isRosterSlide(slide) {
  return slide?.slideKind === "roster";
}

function isCupSlide(slide) {
  return /stanley/i.test(`${slide?.awardKey || ""} ${slide?.awardLabel || ""}`);
}

function categoryLabel(slide) {
  if (isRosterSlide(slide)) return "All-Star team";
  if (isTeamSlide(slide)) return "Team award";
  return "Player award";
}

function cardTeam(card) {
  return card?.teamName || card?.team_name || card?.player?.team_name || card?.player?.teamName || "";
}

function cardLogo(card) {
  return card?.teamLogoSrc || card?.logoSrc || card?.team_logo || card?.player?.teamLogoSrc || null;
}

function cardPosition(card) {
  return card?.position || card?.pos || card?.player?.position || card?.player?.pos || "";
}

function winnerLogo(slide) {
  return slide?.winnerTeamLogoSrc || slide?.winnerLogoSrc || null;
}

function isWinnerCard(card, slide) {
  return Boolean(card?.isWinner) || Boolean(card?.label && card.label === slide?.winnerLabel);
}

function lastName(label = "") {
  const parts = String(label).trim().split(/\s+/);
  return parts[parts.length - 1] || "";
}

function slideFinalists(slide) {
  const cards = slide?.finalistCards?.length ? slide.finalistCards : slide?.candidateCards;
  // Top three by finish, then shown alphabetically — finish order would put the winner first.
  return safeArray(cards)
    .slice(0, 3)
    .sort(
      (a, b) =>
        lastName(a?.label).localeCompare(lastName(b?.label)) ||
        String(a?.label || "").localeCompare(String(b?.label || ""))
    );
}

/* evidence → rows (no placeholders ever rendered) */

function statRows(line) {
  return safeArray(line)
    .map((stat) => {
      const value = formatStat(stat?.value, stat?.fmt);
      if (!value || !stat?.label) return null;
      const rank = stat?.rank && stat?.of ? `${ordinal(stat.rank)} of ${stat.of}` : "";
      return { key: stat.key || stat.label, label: stat.label, value, rank };
    })
    .filter(Boolean);
}

function legacyStatRows(cards) {
  return safeArray(cards)
    .map((stat) => {
      const value = formatStat(stat?.value, stat?.fmt);
      if (!value || !stat?.label) return null;
      return { key: stat.label, label: stat.label, value: `${value}${stat.suffix || ""}`, rank: "" };
    })
    .filter(Boolean);
}

function evidenceFor(card, ev) {
  const list = safeArray(ev?.finalists);
  if (!list.length || !card) return null;
  const id = String(card?.player?.id ?? card?.player?.player_id ?? card?.entityId ?? card?.entity_id ?? "");
  return (
    list.find((f) => (id && String(f?.entity_id) === id) || (f?.name && f.name === card.label)) || null
  );
}

function finalistStatRows(card, ev) {
  const fromEvidence = statRows(evidenceFor(card, ev)?.stat_line);
  if (fromEvidence.length) return fromEvidence;
  const own = statRows(card?.statLine || card?.stat_line);
  if (own.length) return own;
  return legacyStatRows(card?.statCards);
}

function winnerStatRows(slide, ev) {
  const fromEvidence = statRows(ev?.winner?.stat_line);
  return fromEvidence.length ? fromEvidence : legacyStatRows(slide?.statCards);
}

function ballotFor(card, ev) {
  const b = evidenceFor(card, ev)?.ballot;
  if (b && toNum(b.points) !== null) return b;
  const pts = toNum(card?.ballotPoints ?? card?.ballot_points);
  if (pts === null) return null;
  return { points: pts, first_place_votes: card?.firstPlaceVotes ?? card?.first_place_votes };
}

function rosterSlotsFor(slide, ev) {
  const fromEvidence = safeArray(ev?.selections).map((s) => ({
    slot: s?.slot,
    label: s?.name || s?.label,
    teamName: s?.team_name || s?.teamName,
    teamLogoSrc: s?.teamLogoSrc || null,
    stat: statRows(s?.stat_line)[0] || null,
  }));
  if (fromEvidence.length) return fromEvidence;
  return safeArray(slide?.rosterSlots).map((s) => ({
    slot: s?.slot,
    label: s?.label,
    teamName: s?.teamName,
    teamLogoSrc: s?.teamLogoSrc || s?.logoSrc || null,
    stat: statRows(s?.statLine || s?.stat_line)[0] || null,
  }));
}

/* headshots: award candidates are stubs — resolve the full roster player */

function buildPlayerIndex(franchiseState) {
  const index = new Map();
  const add = (player, team) => {
    if (!player || typeof player !== "object") return;
    const merged =
      team && player.team_id == null ? { ...player, team_id: team.team_id ?? team.id } : player;
    const id = player.id ?? player.player_id ?? player.playerId;
    if (id !== null && id !== undefined) index.set(String(id), merged);
    const name = player.name || [player.first_name, player.last_name].filter(Boolean).join(" ");
    if (name) index.set(`name:${String(name).trim().toLowerCase()}`, merged);
  };
  const visitTeam = (team) => {
    if (!team || typeof team !== "object") return;
    [team.roster, team.players, team.nhl_roster, team.active_roster].forEach((list) =>
      safeArray(list).forEach((player) => add(player, team))
    );
  };
  const visit = (source, fn) => {
    if (Array.isArray(source)) source.forEach((item) => fn(item));
    else if (source && typeof source === "object") Object.values(source).forEach((item) => fn(item));
  };
  [
    franchiseState?.teams,
    franchiseState?.league?.teams,
    franchiseState?.nhl_teams,
    franchiseState?.all_teams,
  ].forEach((source) => visit(source, visitTeam));
  [franchiseState?.players, franchiseState?.player_index].forEach((source) =>
    visit(source, (player) => add(player, null))
  );
  collectPlayerRows(franchiseState).forEach((player) => add(player, null));
  return index;
}

function lookupPlayer(player, index, label) {
  const id = player?.id ?? player?.player_id ?? player?.playerId;
  if (id !== null && id !== undefined && index.has(String(id))) return index.get(String(id));
  const name = String(label || player?.name || "").trim().toLowerCase();
  return name && index.has(`name:${name}`) ? index.get(`name:${name}`) : null;
}

function resolvePlayer(player, index, label) {
  const full = lookupPlayer(player, index, label);
  if (full && player) {
    // Award rows carry the backend headshot identity (NHL id, NHL headshot URL,
    // portrait seed). Lean roster rows often lack it, so the award row wins.
    return {
      ...player,
      ...full,
      ...pickHeadshotIdentityFields(full),
      ...pickHeadshotIdentityFields(player),
    };
  }
  return full || player || null;
}

/* ─────────────── season lines + finalist case (real stats from the award rows) ─────────────── */

const SKATER_POS = /^(C|LW|RW|W|F|D|LD|RD)$/;

function rawRow(card) {
  const p = card?.player;
  return p && typeof p === "object" ? p : card || {};
}

function rowPos(card) {
  return String(cardPosition(card) || rawRow(card)?.position || "").toUpperCase();
}

function numOf(row, ...keys) {
  for (const k of keys) {
    const v = toNum(row?.[k]);
    if (v !== null) return v;
  }
  return null;
}

function isGoalieCard(card) {
  const pos = rowPos(card);
  if (pos === "G") return true;
  const r = rawRow(card);
  return !pos && numOf(r, "sv_pct") !== null && numOf(r, "w", "wins") !== null;
}

function isSkaterCard(card) {
  return SKATER_POS.test(rowPos(card));
}

const POS_NAME = { C: "Centre", LW: "Left wing", RW: "Right wing", W: "Wing", F: "Forward", D: "Defence", LD: "Defence", RD: "Defence", G: "Goaltender" };

const AWARD_EXTRAS = {
  norris: [["toi_pg", "TOI", "toi"], ["blk", "BLK", "int"]],
  selke: [["takeaways", "TK", "int"], ["blk", "BLK", "int"]],
  rocket: [["sog", "SOG", "int"]],
  lady_byng: [["pim", "PIM", "int"]],
  hart: [["gp", "GP", "int"]],
  art_ross: [["gp", "GP", "int"]],
  calder: [["gp", "GP", "int"]],
  ted_lindsay: [["gp", "GP", "int"]],
  conn_smythe: [["gp", "GP", "int"]],
};

/** Season line straight from the award candidate row (goalies: record / SV% / GAA / SO). */
function seasonLine(card, awardKey) {
  const r = rawRow(card);
  const out = [];
  const add = (key, label, value) => {
    if (value !== null && value !== undefined && value !== "") out.push({ key, label, value: String(value) });
  };
  if (isGoalieCard(card)) {
    const w = numOf(r, "w", "wins");
    if (w !== null) add("rec", "W-L-OT", [w, numOf(r, "l", "losses") ?? 0, numOf(r, "otl") ?? 0].join("-"));
    add("sv", "SV%", formatStat(numOf(r, "sv_pct", "save_pct"), "sv3"));
    add("gaa", "GAA", formatStat(numOf(r, "gaa"), "dec2"));
    add("so", "SO", formatStat(numOf(r, "so", "shutouts"), "int"));
    add("gp", "GP", formatStat(numOf(r, "gp", "games_played"), "int"));
    return out;
  }
  const pts = numOf(r, "pts", "points");
  if (isSkaterCard(card) || (pts !== null && pts > 0)) {
    add("g", "G", formatStat(numOf(r, "g", "goals"), "int"));
    add("a", "A", formatStat(numOf(r, "a", "assists"), "int"));
    add("pts", "PTS", formatStat(pts, "int"));
    add("pm", "+/-", formatStat(numOf(r, "plus_minus", "pm"), "signed"));
    (AWARD_EXTRAS[awardKey] || []).forEach(([k, label, fmt]) => add(k, label, formatStat(numOf(r, k), fmt)));
    return out;
  }
  const dv = formatStat(card?.displayValue ?? r?.display_value);
  if (dv) add("dv", r?.display_metric || card?.displayMetric || "Score", dv);
  return out;
}

function lineFor(card, slide, ev) {
  const own = seasonLine(card, slide?.awardKey);
  if (own.length) return own;
  return finalistStatRows(card, ev).map((row) => ({ key: row.key, label: row.label, value: row.value }));
}

const CASE_DEFS = {
  pts: ["points", "int", 1],
  g: ["goals", "int", 1],
  a: ["assists", "int", 1],
  plus_minus: ["plus-minus", "signed", 1],
  toi_pg: ["ice time per game", "toi", 1],
  blk: ["blocked shots", "int", 1],
  takeaways: ["takeaways", "int", 1],
  sog: ["shots", "int", 1],
  pim: ["penalty minutes", "int", -1],
  sv_pct: ["save percentage", "sv3", 1],
  gaa: ["goals-against average", "dec2", -1],
  w: ["wins", "int", 1],
  so: ["shutouts", "int", 1],
};

const CASE_ORDER = {
  norris: ["pts", "toi_pg", "blk", "plus_minus"],
  selke: ["takeaways", "plus_minus", "blk", "pts"],
  rocket: ["g", "sog", "pts"],
  lady_byng: ["pim", "pts", "a"],
  skater: ["pts", "g", "a", "plus_minus"],
  goalie: ["sv_pct", "gaa", "w", "so"],
};

/** One short "why he's here" line per finalist, measured against the other finalists only. */
function finalistCases(cards, awardKey) {
  const out = new Map();
  const groups = { goalie: [], skater: [] };
  cards.forEach((card) => {
    if (isGoalieCard(card)) groups.goalie.push(card);
    else if (isSkaterCard(card) || numOf(rawRow(card), "pts", "points") !== null) groups.skater.push(card);
  });
  Object.entries(groups).forEach(([role, list]) => {
    if (!list.length) return;
    const keys = role === "goalie" ? CASE_ORDER.goalie : CASE_ORDER[awardKey] || CASE_ORDER.skater;
    list.forEach((card) => {
      const r = rawRow(card);
      if (list.length === 1) {
        const line = seasonLine(card, awardKey).slice(0, 3).map((s) => `${s.value} ${s.label}`).join(" · ");
        if (cards.length > 1 && line) out.set(card, `Only ${role === "goalie" ? "goaltender" : "skater"} among the finalists: ${line}`);
        return;
      }
      const leads = [];
      let fallback = null;
      keys.forEach((key) => {
        const [noun, fmt, dir] = CASE_DEFS[key];
        const mine = numOf(r, key);
        if (mine === null) return;
        const vals = list.map((c) => numOf(rawRow(c), key)).filter((v) => v !== null);
        if (vals.length < 2) return;
        const better = vals.filter((v) => (dir > 0 ? v > mine : v < mine)).length;
        const ties = vals.filter((v) => v === mine).length;
        const shown = formatStat(mine, fmt);
        if (better === 0 && ties === 1) leads.push(dir > 0 ? `Leads the finalists in ${noun} (${shown})` : `Lowest ${noun} of the finalists (${shown})`);
        else if (better === 0) leads.push(`Tied for the finalist lead in ${noun} (${shown})`);
        else if (!fallback) fallback = `${ordinal(better + 1)} among the finalists in ${noun} (${shown})`;
      });
      const text = leads.length ? leads.slice(0, 2).join(". ") : fallback;
      if (text) out.set(card, text);
    });
  });
  return out;
}

function fitStyle(text, max) {
  return { "--aw2-len": Math.max(8, String(text || "").length), ...(max ? { "--aw2-max": `${max}px` } : {}) };
}

function isUserCard(card, userTeam) {
  if (!card || !userTeam) return false;
  const name = cardTeam(card);
  if (userTeam.name && name && name === userTeam.name) return true;
  const tid = rawRow(card)?.team_id;
  return Boolean(userTeam.id && tid !== undefined && tid !== null && String(tid) === String(userTeam.id));
}

function posLine(card, player) {
  const pos = rowPos(card) || String(player?.position || "").toUpperCase();
  const age = numOf(rawRow(card), "age") ?? numOf(player, "age");
  return [POS_NAME[pos] || pos, age ? `age ${age}` : ""].filter(Boolean).join(" · ");
}

/* ─────────────── small components ─────────────── */

function TeamLogo({ src, size = 32, className = "" }) {
  if (!src) return null;
  return (
    <span className={`aw2-logo ${className}`} style={{ "--aw2-logo": `${size}px` }} aria-hidden="true">
      <img
        src={src}
        alt=""
        onError={(event) => {
          event.currentTarget.style.visibility = "hidden";
        }}
      />
    </span>
  );
}

function Headshot({ player, size = 120 }) {
  if (!player) return null;
  return (
    <div className="aw2-face" style={{ "--aw2-face": `${size}px` }}>
      <PlayerHeadshot player={player} size="lg" mood="neutral" animate="in" preferPhoto showFlag={false} />
    </div>
  );
}

function StatLine({ items, size = "md" }) {
  if (!items.length) return null;
  return (
    <dl className={`aw2-line is-${size}`}>
      {items.map((s) => (
        <div key={s.key || s.label}>
          <dd>{s.value}</dd>
          <dt>{s.label}</dt>
        </div>
      ))}
    </dl>
  );
}

function BallotRail({ points, leader, fpv }) {
  if (points === null || !leader) return null;
  const width = Math.max(2, Math.min(100, (points / leader) * 100));
  return (
    <div className="aw2-ballot">
      <div className="aw2-rail">
        <i style={{ width: `${width}%` }} />
      </div>
      <span>
        {formatPoints(points)} ballot points{fpv !== null ? ` · ${fpv} first-place votes` : ""}
      </span>
    </div>
  );
}

function Bars({ items }) {
  return (
    <div className="aw2-bars">
      {items.map((c) => (
        <div key={c.key || c.label}>
          <div className="aw2-bars-head">
            <span>{c.label}</span>
            <b>{c.text}</b>
          </div>
          <div className="aw2-rail">
            <i style={{ width: `${Math.max(0, Math.min(100, c.pct))}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

/* ─────────────── left rail: ceremony order ─────────────── */

function CeremonyLog({ groups, slidesCount, activeIndex, revealedIds, onSelect }) {
  const presented = groups.reduce(
    (n, g) => n + safeArray(g?.items).filter(({ slide }) => slide && revealedIds.has(slide.id)).length,
    0
  );
  return (
    <aside className="aw2-log" aria-label="Ceremony order">
      <div className="aw2-railhead">
        <div className="aw2-railhead-name" role="heading" aria-level={2}>
          Ceremony order
        </div>
        <p>
          {presented} of {slidesCount} presented
        </p>
      </div>
      <div className="aw2-log-scroll">
        {groups.map((group, g) => (
          <div key={group?.id || group?.label || g} className="aw2-log-group">
            {group?.label ? <div className="aw2-log-grouplabel">{group.label}</div> : null}
            {safeArray(group?.items).map(({ slide, index, offRail }) => {
              if (!slide) return null;
              const clickable = !offRail && Number.isInteger(index);
              const revealed = Boolean(offRail) || revealedIds.has(slide.id);
              const selected = clickable && index === activeIndex;
              const sub = revealed
                ? isRosterSlide(slide)
                  ? "Lineup named"
                  : slide.winnerLabel
                : selected
                  ? "On stage now"
                  : "";
              const classes = [
                "aw2-log-row",
                selected ? "is-on" : "",
                revealed ? "is-done" : "",
                isMajorSlide(slide) ? "is-major" : "",
              ]
                .filter(Boolean)
                .join(" ");
              return (
                <button key={slide.id} type="button" className={classes} disabled={!clickable} onClick={() => onSelect(index)}>
                  <span className="aw2-log-num">{clickable ? pad2(index + 1) : ""}</span>
                  <span className="aw2-log-body">
                    <strong>{slide.awardLabel}</strong>
                    {sub ? <span>{sub}</span> : null}
                  </span>
                  <span className="aw2-log-mark">
                    {revealed && !isRosterSlide(slide) ? <TeamLogo src={winnerLogo(slide)} size={30} /> : null}
                  </span>
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </aside>
  );
}

function hashOf(text) {
  let h = 2166136261;
  const str = String(text || "");
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function joinNames(list) {
  const names = list.filter(Boolean);
  if (names.length <= 1) return names[0] || "";
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

/** "Dallas Stars" -> "Stars fan" */
function fanOf(teamName) {
  const full = String(teamName || "").trim();
  if (!full) return "Hockey fan";
  const two = ["Maple Leafs", "Red Wings", "Blue Jackets", "Golden Knights"].find((n) => full.endsWith(n));
  return `${two || full.split(" ").slice(-1)[0]} fan`;
}

function winnersOf(slide) {
  return String(slide?.winnerLabel || "")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);
}

/**
 * Fan Pulse items for one moment of the night. Every line is written from the actual result
 * (winner, club, numbers, finalists), so nothing reads as a filled-in blank and nothing repeats.
 * stage: "gate" | "pre" | "post" | "summary". Nothing in "gate" or "pre" names a winner.
 */
function buildPulse({ stage, slide, slides, ev }) {
  const out = [];
  const push = (kind, text, extra = {}) => {
    if (!text) return;
    out.push({ kind, text, likes: 40 + (hashOf(text) % 2400), ...extra });
  };
  const pickOne = (seed, options) => options[hashOf(seed) % options.length];

  if (stage === "gate") {
    const cup = slides.find(isCupSlide);
    push("news", `Awards Night is here. ${slides.length} honours will be handed out tonight.`);
    if (cup?.winnerLabel) push("news", `The ${cup.winnerLabel} arrive as Stanley Cup champions.`, { logo: winnerLogo(cup) });
    const offset = hashOf(`gate-${slides.length}-${cup?.winnerLabel || ""}`);
    MAJOR_KEYS.forEach((key, k) => {
      const sl = slides.find((x) => x.awardKey === key);
      if (!sl) return;
      const cards = slideFinalists(sl);
      if (cards.length < 2) return;
      const fav = cards[hashOf(`${key}-fav`) % cards.length];
      const other = cards.find((c) => c !== fav) || cards[0];
      const favLine = seasonLine(fav, key).slice(0, 3).map((s) => `${s.value} ${s.label}`).join(", ");
      const options = [
        `${sl.awardLabel} finalists are ${joinNames(cards.map((c) => c.label))}. I'd give it to ${lastName(fav.label)}.`,
        `${lastName(fav.label)} for the ${sl.awardLabel}.${favLine ? ` ${favLine}. Next question.` : ""}`,
        `${joinNames(cards.map((c) => lastName(c.label)))} for the ${sl.awardLabel}. Any of the three would be a fair winner.`,
        `If ${lastName(fav.label)} doesn't win the ${sl.awardLabel} tonight I'm going to be annoyed.`,
        `People are sleeping on ${lastName(fav.label)} in the ${sl.awardLabel} race.${favLine ? ` ${favLine}.` : ""}`,
        `${lastName(fav.label)} or ${lastName(other.label)} for the ${sl.awardLabel}? Too close to call for me.`,
      ];
      push("fan", options[(offset + k) % options.length], { who: fanOf(cardTeam(fav)), logo: cardLogo(fav) });
    });
    return out;
  }

  if (stage === "summary") {
    const cup = slides.find(isCupSlide);
    const hart = slides.find((x) => x.awardKey === "hart");
    if (cup?.winnerLabel) push("news", `The ${cup.winnerLabel} are Stanley Cup champions.`, { logo: winnerLogo(cup) });
    if (hart?.winnerLabel) push("news", `${hart.winnerLabel} is the league's most valuable player.`, { logo: winnerLogo(hart) });
    const byPlayer = new Map();
    const byTeam = new Map();
    slides.forEach((sl) => {
      if (isCupSlide(sl) || isRosterSlide(sl)) return;
      if (!isTeamSlide(sl)) {
        winnersOf(sl).forEach((name) => byPlayer.set(name, [...(byPlayer.get(name) || []), sl]));
      }
      const team = isTeamSlide(sl) ? sl.winnerLabel : sl.winnerTeamName;
      if (team) byTeam.set(team, [...(byTeam.get(team) || []), sl]);
    });
    [...byPlayer.entries()]
      .filter(([, list]) => list.length >= 2)
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, 2)
      .forEach(([name, list]) => {
        push("stat", `${name} leaves with ${list.length} trophies: ${joinNames(list.map((x) => x.awardLabel))}.`, {
          logo: winnerLogo(list[0]),
        });
        push("fan", `${list.length} trophies in one night. ${lastName(name)} just had the season of his life.`, {
          who: fanOf(list[0].winnerTeamName),
          logo: winnerLogo(list[0]),
        });
      });
    const topTeam = [...byTeam.entries()].sort((a, b) => b[1].length - a[1].length)[0];
    if (topTeam && topTeam[1].length >= 3) {
      push("stat", `${topTeam[0]} collected ${topTeam[1].length} honours tonight, more than any other club.`, { logo: winnerLogo(topTeam[1][0]) });
    }
    const calder = slides.find((x) => x.awardKey === "calder");
    if (calder?.winnerLabel) {
      push("fan", `${lastName(calder.winnerLabel)} winning rookie of the year means we get to watch him for the next fifteen.`, {
        who: fanOf(calder.winnerTeamName),
        logo: winnerLogo(calder),
      });
    }
    const vez = slides.find((x) => x.awardKey === "vezina");
    if (vez?.winnerLabel) {
      push("fan", `Best goalie in the league and it wasn't close. ${lastName(vez.winnerLabel)} stole us a dozen games.`, {
        who: fanOf(vez.winnerTeamName),
        logo: winnerLogo(vez),
      });
    }
    push("news", "That wraps the season. Retirements, the draft and free agency are next.");
    return out;
  }

  if (!slide) return out;
  const award = slide.awardLabel;
  const team = isTeamSlide(slide);
  const roster = isRosterSlide(slide);
  const cards = roster ? [] : slideFinalists(slide);

  if (stage === "pre") {
    push("news", `Up now: the ${award}.${slide.stageLine ? ` ${slide.stageLine}` : ""}`);
    if (roster) {
      push("fan", `Lineup time. Six names, and somebody's favourite always gets left off the ${award}.`, { who: "Hockey fan" });
      return out;
    }
    if (cards.length < 2) {
      push("stat", `The ${award} is settled by the numbers. There is no ballot for this one.`);
      return out;
    }
    const start = hashOf(award);
    cards.forEach((card, ci) => {
      const lineRows = lineFor(card, slide, ev);
      const row = lineRows.find((r) => r.key === "pts" || r.key === "sv") || lineRows[0];
      const num = row ? ` ${row.value} ${row.label} speaks for itself.` : "";
      push(
        "fan",
        [
          `${card.label} should win the ${award}.${num}`,
          `Put me down for ${lastName(card.label)}.${num}`,
          `It has to be ${lastName(card.label)} tonight.${num}`,
        ][(start + ci) % 3],
        { who: fanOf(cardTeam(card)), logo: cardLogo(card) }
      );
    });
    push("debate", `${lastName(cards[0].label)} or ${lastName(cards[1].label)}? This ${award} vote could go either way.`);
    return out;
  }

  // stage === "post": the result is known
  const names = winnersOf(slide);
  const logo = winnerLogo(slide);
  const why = cleanLines(safeArray(ev?.why).length ? ev.why : slide?.whyTheyWon);
  const stats = roster ? [] : winnerStatRows(slide, ev).slice(0, 3);
  if (roster) {
    push("news", `The ${award} is set.`, { logo });
    push("fan", `That's a strong ${award}. Hard to argue with any of the six.`, { who: "Hockey fan" });
    push("debate", `Every year one deserving name misses the ${award}. Who was it this time?`);
    return out;
  }
  if (team) {
    push("news", `The ${slide.winnerLabel} take the ${award}.`, { logo });
    if (why[0]) push("stat", why[0], { logo });
    push("fan", `${award} is ours. Now do it again next year.`, { who: fanOf(slide.winnerLabel), logo });
    return out;
  }
  push("news", names.length > 1 ? `${joinNames(names)} share the ${award}.` : `${slide.winnerLabel} wins the ${award}.`, { logo });
  if (why[0]) push("stat", why[0], { logo });
  else if (stats.length) push("stat", `${slide.winnerLabel}: ${stats.map((r) => `${r.value} ${r.label}`).join(", ")}.`, { logo });
  const last = lastName(names[0] || slide.winnerLabel);
  push(
    "fan",
    pickOne(`${award}-win`, [
      `${last} earned every bit of that ${award}. We watched it all season.`,
      `Nobody in this building is surprised. ${last} was the best at it all year.`,
      `Get ${last}'s name on that trophy. Fully deserved.`,
    ]),
    { who: fanOf(slide.winnerTeamName), logo }
  );
  const runner = [...cards]
    .filter((card) => !isWinnerCard(card, slide))
    .sort((a, b) => (toNum(a?.rank) ?? 99) - (toNum(b?.rank) ?? 99))[0];
  if (runner) {
    push(
      "debate",
      pickOne(`${award}-runner`, [
        `${runner.label} had a real case for the ${award}, but it's hard to argue with ${last}'s year.`,
        `Tough one for ${lastName(runner.label)}. Most seasons that wins the ${award}.`,
        `I had ${lastName(runner.label)} on my ballot. No complaints about ${last} though.`,
      ]),
      { who: fanOf(cardTeam(runner)), logo: cardLogo(runner) }
    );
  }
  return out;
}

const PULSE_TAG = { news: "News", stat: "By the numbers", debate: "Debate" };

function FanPulse({ mode, items, emptyText, feedKey }) {
  return (
    <aside className="aw2-pulse" aria-label="Fan Pulse">
      <div className="aw2-railhead">
        <div className="aw2-railhead-name" role="heading" aria-level={2}>
          Fan Pulse
        </div>
        <p>{mode === "reactions" ? "Reactions around the league" : "Before the envelopes open"}</p>
      </div>
      <div className="aw2-pulse-list" key={feedKey}>
        {items.length ? (
          items.map((item, i) => (
            <article key={`${item.kind}-${i}`} className={`aw2-post is-${item.kind}`} style={{ animationDelay: `${i * 90}ms` }}>
              {item.kind === "fan" || item.kind === "debate" ? (
                <>
                  <header>
                    {item.logo ? <TeamLogo src={item.logo} size={26} /> : <span className="aw2-post-dot" aria-hidden="true" />}
                    <b>{item.who || "Around the league"}</b>
                    {item.kind === "debate" ? <em>{PULSE_TAG.debate}</em> : null}
                  </header>
                  <p>{item.text}</p>
                  <footer>{item.likes.toLocaleString()} likes</footer>
                </>
              ) : (
                <>
                  <header>
                    {item.logo ? <TeamLogo src={item.logo} size={26} /> : null}
                    <em>{PULSE_TAG[item.kind]}</em>
                  </header>
                  <p className="aw2-post-headline">{item.text}</p>
                </>
              )}
            </article>
          ))
        ) : (
          <p className="aw2-muted">{emptyText}</p>
        )}
      </div>
    </aside>
  );
}

/* ─────────────── stage bodies ─────────────── */

const MAJOR_KEYS = ["hart", "norris", "vezina", "calder", "selke", "conn_smythe"];
const isMajorSlide = (slide) => MAJOR_KEYS.includes(String(slide?.awardKey || ""));
// Decided on the ice or in the standings: nobody votes on these, so they are shown as results, not sealed envelopes.
const isDecidedSlide = (slide) => isCupSlide(slide) || /president|conference/i.test(String(slide?.awardKey || ""));

/** One nominee column: headshot, full name, logo + full team name, season line, case. */
function Nominee({ card, slide, ev, playerIndex, userTeam, caseText, faceSize = 112, team = false }) {
  const player = team ? null : resolvePlayer(card?.player, playerIndex, card?.label);
  const mine = isUserCard(card, userTeam);
  const line = team ? [] : lineFor(card, slide, ev);
  const meta = team ? "" : posLine(card, player);
  return (
    <div className={`aw2-nominee${mine ? " is-mine" : ""}`}>
      {team ? <TeamLogo src={cardLogo(card)} size={faceSize} /> : <Headshot player={player} size={faceSize} />}
      <div className="aw2-nominee-name">{card?.label}</div>
      {!team && cardTeam(card) ? (
        <div className="aw2-nominee-team">
          <TeamLogo src={cardLogo(card)} size={26} />
          <span>{cardTeam(card)}</span>
        </div>
      ) : null}
      {meta ? <div className="aw2-nominee-meta">{meta}{mine ? " · your club" : ""}</div> : mine ? <div className="aw2-nominee-meta">Your club</div> : null}
      <StatLine items={line.slice(0, 5)} size="sm" />
      {caseText ? <p className="aw2-nominee-case">{caseText}</p> : null}
    </div>
  );
}

function GateBody({ slides, playerIndex, userTeam, onPick, evidenceByKey }) {
  const indexOf = (slide) => slides.findIndex((x) => x.id === slide.id);
  const majors = MAJOR_KEYS.map((k) => slides.find((x) => x.awardKey === k)).filter(Boolean);
  const decided = slides.filter(isDecidedSlide);
  const cup = decided.find(isCupSlide) || null;
  const others = decided.filter((s) => !isCupSlide(s));
  const rest = slides.filter((x) => !isMajorSlide(x) && !isDecidedSlide(x));
  const mine = [];
  slides.forEach((slide) => {
    if (isRosterSlide(slide) || isTeamSlide(slide)) return;
    slideFinalists(slide).forEach((card) => {
      if (isUserCard(card, userTeam)) mine.push({ slide, card });
    });
  });
  const pickKeys = (slide) => ({
    role: "button",
    tabIndex: 0,
    onClick: () => onPick?.(indexOf(slide)),
    onKeyDown: (e) => {
      if (e.key === "Enter") onPick?.(indexOf(slide));
    },
  });

  return (
    <div className="aw2-gate">
      {cup || others.length || mine.length ? (
        <section className="aw2-marquee">
          {cup ? (
            <div className="aw2-champ">
              <TeamLogo src={winnerLogo(cup)} size={150} />
              <div className="aw2-champ-text">
                <span>Stanley Cup champions</span>
                <div className="aw2-display aw2-fit" style={fitStyle(cup.winnerLabel, 58)}>
                  {cup.winnerLabel}
                </div>
                {cup.stageLine ? <p>{cup.stageLine}</p> : null}
              </div>
            </div>
          ) : null}
          <div className="aw2-marquee-side">
            {others.map((slide) => (
              <div key={slide.id} className="aw2-decided">
                <TeamLogo src={winnerLogo(slide)} size={52} />
                <div>
                  <span>{slide.awardLabel}</span>
                  <b>{slide.winnerLabel}</b>
                </div>
              </div>
            ))}
            {mine.length ? (
              <div className="aw2-mine">
                <span className="aw2-mine-label">Your club tonight</span>
                {mine.map(({ slide, card }) => (
                  <button key={`${slide.id}-${card.label}`} type="button" onClick={() => onPick?.(indexOf(slide))}>
                    <Headshot player={resolvePlayer(card?.player, playerIndex, card?.label)} size={56} />
                    <span>
                      <b>{card.label}</b>
                      <i>{slide.awardLabel} finalist</i>
                    </span>
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        </section>
      ) : null}

      {majors.length ? (
        <div className="aw2-sectionhead">
          <div role="heading" aria-level={3}>The major awards</div>
          <p>Three finalists each, ballots sealed. Pick an award to open it on stage.</p>
        </div>
      ) : null}
      <div className="aw2-majors">
        {majors.map((slide) => {
          const team = isTeamSlide(slide);
          const ev = slideEvidence(slide) || evidenceByKey?.[slide.awardKey] || null;
          const finalists = slideFinalists(slide);
          const cases = team ? new Map() : finalistCases(finalists, slide.awardKey);
          const mineHere = finalists.some((c) => isUserCard(c, userTeam));
          return (
            <section key={slide.id} className={`aw2-award${mineHere ? " is-mine" : ""}`} {...pickKeys(slide)}>
              <div className="aw2-award-head">
                <div className="aw2-display aw2-fit" style={fitStyle(slide.awardLabel, 34)}>
                  {slide.awardLabel}
                </div>
                {slide.stageLine ? <p>{slide.stageLine}</p> : null}
              </div>
              <div className="aw2-nominees" style={{ "--aw2-cols": Math.max(1, finalists.length) }}>
                {finalists.map((card) => (
                  <Nominee
                    key={card.label}
                    card={card}
                    slide={slide}
                    ev={ev}
                    playerIndex={playerIndex}
                    userTeam={userTeam}
                    caseText={cases.get(card)}
                    team={team}
                    faceSize={104}
                  />
                ))}
              </div>
            </section>
          );
        })}
      </div>

      {rest.length ? (
        <div className="aw2-sectionhead">
          <div role="heading" aria-level={3}>Also tonight</div>
          <p>{rest.length} more honours, from the scoring races to the All-Star teams.</p>
        </div>
      ) : null}
      <div className="aw2-rest">
        {rest.map((slide) => {
          const cards = isRosterSlide(slide) ? [] : slideFinalists(slide);
          const mineHere = cards.some((c) => isUserCard(c, userTeam));
          const names = cards.map((c) => lastName(c.label));
          return (
            <button key={slide.id} type="button" className={`aw2-rest-row${mineHere ? " is-mine" : ""}`} onClick={() => onPick?.(indexOf(slide))}>
              <span className="aw2-rest-num">{pad2(indexOf(slide) + 1)}</span>
              <span>
                <b>{slide.awardLabel}</b>
                <i>
                  {isRosterSlide(slide)
                    ? "Six-man lineup"
                    : names.length > 1
                      ? `Finalists: ${names.join(", ")}`
                      : names.length === 1
                        ? "Decided by the numbers"
                        : slide.stageLine || categoryLabel(slide)}
                </i>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function TitleBody({ slide, ev, finalists, rule = "", playerIndex, userTeam }) {
  const criteria = safeArray(ev?.criteria).filter((c) => c?.label && toNum(c.weight) !== null);
  const excluded = safeArray(ev?.excluded_criteria).filter((c) => c?.label);
  const eligibility = cleanLines([ev?.eligibility, slide?.eligibilitySummary, slide?.eligibility_summary])[0] || "";
  const method = METHOD_LABEL[ev?.method] || (isRosterSlide(slide) ? METHOD_LABEL.selection : "");
  const voters = toNum(ev?.winner?.ballot?.voter_count);
  const poolSize = toNum(ev?.pool?.size);
  const facts = [
    method ? { key: "f", label: "Format", value: method } : null,
    voters !== null ? { key: "v", label: "Voters", value: String(voters) } : null,
    poolSize !== null ? { key: "p", label: ev?.pool?.noun ? `Eligible ${ev.pool.noun}` : "Eligible", value: String(poolSize) } : null,
    finalists.length ? { key: "n", label: "Finalists", value: String(finalists.length) } : null,
  ].filter(Boolean);
  const maxWeight = Math.max(0, ...criteria.map((c) => Number(c.weight))) || 1;
  const team = isTeamSlide(slide);

  return (
    <div className="aw2-title">
      <section className="aw2-title-main">
        <span className="aw2-label">{categoryLabel(slide)}</span>
        {rule ? <p className="aw2-rule">{rule}</p> : slide?.stageLine ? <p className="aw2-rule">{slide.stageLine}</p> : null}
        {eligibility ? <p className="aw2-copy">This season: {eligibility}</p> : null}
        {!rule && !eligibility && !criteria.length ? (
          <p className="aw2-muted">Voting criteria weren't published with this season's results.</p>
        ) : null}
        {facts.length ? <StatLine items={facts} size="lg" /> : null}
      </section>
      <section className="aw2-title-side">
        {criteria.length ? (
          <>
            <span className="aw2-label">How the ballot weighs it</span>
            <Bars
              items={criteria.map((c) => ({
                key: c.key || c.label,
                label: c.label,
                text: `${Math.round(Number(c.weight) * 100)}%`,
                pct: (Number(c.weight) / maxWeight) * 100,
              }))}
            />
          </>
        ) : null}
        {excluded.map((c) => (
          <p key={c.key || c.label} className="aw2-muted">
            {c.label}: not tracked this season
          </p>
        ))}
        {finalists.length && !isRosterSlide(slide) ? (
          <div className="aw2-title-who">
            <span className="aw2-label">In the envelope</span>
            <div className="aw2-title-faces">
              {finalists.map((card) => (
                <div key={card.label} className={isUserCard(card, userTeam) ? "is-mine" : ""}>
                  {team ? (
                    <TeamLogo src={cardLogo(card)} size={64} />
                  ) : (
                    <Headshot player={resolvePlayer(card?.player, playerIndex, card?.label)} size={64} />
                  )}
                  <b>{card.label}</b>
                  {!team ? <i>{cardTeam(card)}</i> : null}
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </section>
    </div>
  );
}

function FinalistsBody({ slide, ev, finalists, phase, shown, playerIndex, userTeam }) {
  const revealed = isRevealedPhase(phase);
  const team = isTeamSlide(slide);
  const ballots = finalists.map((card) => ballotFor(card, ev));
  const leader = Math.max(0, ...ballots.map((b) => toNum(b?.points) ?? 0));
  const cases = team ? new Map() : finalistCases(finalists, slide?.awardKey);

  return (
    <div className="aw2-finalists" style={{ "--aw2-cols": Math.max(1, finalists.length) }}>
      {finalists.map((card, i) => {
        const winner = revealed && isWinnerCard(card, slide);
        const mine = isUserCard(card, userTeam);
        const classes = [
          "aw2-finalist",
          phase === PHASE.FINALISTS && i >= shown ? "is-pending" : "is-in",
          phase === PHASE.SEAL ? "is-sealed" : "",
          winner ? "is-winner" : "",
          revealed && !winner ? "is-dim" : "",
          mine ? "is-mine" : "",
        ]
          .filter(Boolean)
          .join(" ");
        const player = team ? null : resolvePlayer(card?.player, playerIndex, card?.label);
        const line = team ? [] : lineFor(card, slide, ev);
        const ballot = ballots[i];
        const teamName = cardTeam(card);
        const meta = team ? "" : posLine(card, player);
        const extra = line.length ? null : formatStat(card?.stat) || formatStat(card?.subline);

        return (
          <article key={`${card?.label}-${i}`} className={classes}>
            <span className="aw2-finalist-state">{winner ? "Winner" : mine ? "Finalist · your club" : "Finalist"}</span>
            {team ? <TeamLogo src={cardLogo(card)} size={168} /> : <Headshot player={player} size={176} />}
            <div className="aw2-finalist-name">{card?.label}</div>
            {!team && teamName ? (
              <div className="aw2-nominee-team is-lg">
                <TeamLogo src={cardLogo(card)} size={36} />
                <span>{teamName}</span>
              </div>
            ) : null}
            {meta ? <div className="aw2-nominee-meta">{meta}</div> : null}
            {line.length ? <StatLine items={line.slice(0, 6)} size="md" /> : extra ? <p className="aw2-copy">{extra}</p> : null}
            {cases.get(card) ? <p className="aw2-nominee-case">{cases.get(card)}</p> : null}
            {revealed && ballot && leader > 0 ? (
              <BallotRail points={toNum(ballot.points)} leader={leader} fpv={toNum(ballot.first_place_votes)} />
            ) : null}
          </article>
        );
      })}
    </div>
  );
}

function RosterBody({ slide, ev, phase, playerIndex }) {
  const revealed = isRevealedPhase(phase);
  const slots = rosterSlotsFor(slide, ev);
  if (!slots.length) {
    return <p className="aw2-muted aw2-pad">Selections weren't recorded for this season.</p>;
  }
  return (
    <div className="aw2-roster">
      {slots.map((slot, i) => {
        const who = revealed ? resolvePlayer({ name: slot.label }, playerIndex, slot.label) : null;
        return (
          <div
            key={`${slot.slot}-${i}`}
            className={`aw2-roster-row${revealed ? " is-in" : ""}`}
            style={phase === PHASE.REVEAL ? { animationDelay: `${i * 220}ms` } : undefined}
          >
            <span className="aw2-roster-pos">{slot.slot}</span>
            <span className="aw2-roster-face">{who ? <Headshot player={who} size={64} /> : null}</span>
            <span className="aw2-roster-name">{revealed ? slot.label : "Envelope sealed"}</span>
            <span className="aw2-nominee-team">
              {revealed ? (
                <>
                  <TeamLogo src={slot.teamLogoSrc} size={30} />
                  <span>{slot.teamName || ""}</span>
                </>
              ) : null}
            </span>
            <span className="aw2-roster-stat">{revealed && slot.stat ? `${slot.stat.value} ${slot.stat.label}` : ""}</span>
          </div>
        );
      })}
    </div>
  );
}

/* ballot breakdown: top-5 vote getters with 1st-place votes + ballot points */
function VoteTable({ table, format, playerIndex, team }) {
  const rows = safeArray(table).filter((r) => r && r.name);
  if (!rows.length) return null;
  const ballot = rows.some((r) => toNum(r?.points) !== null);
  // A "1st" column of zeros and a ballot formula with no ballots read as broken: show them only when votes exist.
  const hasFirst = rows.some((r) => (toNum(r?.first_place_votes) ?? 0) > 0);
  const curve = safeArray(format?.points).map((p) => formatPoints(Number(p))).join("-");
  return (
    <div className="aw2-votes">
      <span className="aw2-label">{ballot ? (hasFirst ? "Voting results" : "How the race finished") : "Final order"}</span>
      {ballot && hasFirst && format ? (
        <p className="aw2-muted">
          {[format.voters ? `${format.voters} ballots` : "", format.body || "", curve ? `${curve} points` : ""].filter(Boolean).join(" · ")}
        </p>
      ) : null}
      <div className="aw2-votes-row is-head">
        <span>#</span>
        <span>Player</span>
        <span>{ballot && hasFirst ? "1st" : ""}</span>
        <span>{ballot ? (hasFirst ? "Pts" : "Score") : "Total"}</span>
      </div>
      {rows.map((r, i) => {
        const who = team ? null : resolvePlayer({ ...r, player_id: r.player_id ?? r.entity_id }, playerIndex, r.name);
        const sub = [r.team_name || r.team_abbr, r.position, r.summary].filter(Boolean).join(" · ");
        return (
          <div key={`${r.entity_id || r.name}-${i}`} className={`aw2-votes-row${r.is_winner ? " is-winner" : ""}`}>
            <span className="aw2-num">{toNum(r.rank) ?? i + 1}</span>
            <span className="aw2-votes-who">
              {who ? <Headshot player={who} size={40} /> : null}
              <span>
                <strong>{r.name}</strong>
                {sub ? <small>{sub}</small> : null}
              </span>
            </span>
            <em className="aw2-num">{ballot && hasFirst ? toNum(r.first_place_votes) ?? 0 : ""}</em>
            <em className="aw2-num">{ballot ? formatPoints(toNum(r.points) ?? 0) : formatStat(r.value) || ""}</em>
          </div>
        );
      })}
    </div>
  );
}

function CitationBody({ slide, ev, finalists, playerIndex, rule = "", userTeam }) {
  const team = isTeamSlide(slide);
  const winnerCard = finalists.find((card) => isWinnerCard(card, slide)) || null;
  const player = team ? null : resolvePlayer(slide?.winnerPlayer || winnerCard?.player, playerIndex, slide?.winnerLabel);
  const logo = winnerLogo(slide) || cardLogo(winnerCard);
  const teamName = slide?.winnerTeamName || cardTeam(winnerCard);
  const meta = team ? "" : posLine(winnerCard, player);
  const why = cleanLines(safeArray(ev?.why).length ? ev.why : slide?.whyTheyWon);
  const own = winnerCard && !team ? seasonLine(winnerCard, slide?.awardKey) : [];
  const stats = own.length ? own : winnerStatRows(slide, ev).map((r) => ({ key: r.key, label: r.label, value: r.value }));
  const components = safeArray(ev?.winner?.components).filter((c) => c?.label && toNum(c.pct) !== null);
  const ballot = ev?.winner?.ballot || null;
  const race = ev?.race || null;
  const order = [...finalists].sort((a, b) => (toNum(a?.rank) ?? 99) - (toNum(b?.rank) ?? 99));
  const hasBallot = Boolean(ballot) && toNum(ballot.first_place_votes) !== null;
  const hasRace = Boolean(race) && toNum(race.leader_value) !== null;
  const voteTable = safeArray(ev?.voting_table).filter((r) => r && r.name);
  const hasResult = hasBallot || hasRace || order.length > 1 || voteTable.length > 0;
  const hasCase = why.length || stats.length || components.length;
  const mine = (userTeam?.name && teamName === userTeam.name) || isUserCard(winnerCard, userTeam);
  const peak = hasBallot
    ? [
        { key: "fpv", label: toNum(ballot.voter_count) !== null ? `First-place votes of ${toNum(ballot.voter_count)}` : "First-place votes", value: String(toNum(ballot.first_place_votes)) },
        toNum(ballot.points) !== null ? { key: "pts", label: "Ballot points", value: formatPoints(toNum(ballot.points)) } : null,
        toNum(ballot.margin) !== null ? { key: "m", label: ballot.runner_up_name ? `Margin over ${lastName(ballot.runner_up_name)}` : "Margin", value: `+${formatPoints(toNum(ballot.margin))}` } : null,
      ].filter(Boolean)
    : hasRace
      ? [
          { key: "lv", label: race.metric_label || "Leader", value: formatStat(race.leader_value) },
          toNum(race.margin) !== null ? { key: "m", label: "Lead", value: `+${formatStat(race.margin)}` } : null,
          toNum(race.runner_up_value) !== null ? { key: "ru", label: "Runner-up", value: formatStat(race.runner_up_value) } : null,
        ].filter(Boolean)
      : [];

  return (
    <div className={`aw2-cite${hasResult ? "" : " is-two"}`}>
      <section className={`aw2-cite-id${mine ? " is-mine" : ""}`}>
        {team ? <TeamLogo src={logo} size={190} /> : <Headshot player={player} size={210} />}
        <span className="aw2-label is-gold">{slide?.awardLabel}</span>
        <div className="aw2-display aw2-fit" style={fitStyle(slide?.winnerLabel, 46)}>
          {slide?.winnerLabel}
        </div>
        {!team && teamName ? (
          <div className="aw2-nominee-team is-lg">
            <TeamLogo src={logo} size={36} />
            <span>{teamName}</span>
          </div>
        ) : null}
        {meta ? <div className="aw2-nominee-meta">{meta}{mine ? " · your club" : ""}</div> : null}
      </section>

      <section className="aw2-cite-case">
        {stats.length ? (
          <div>
            <span className="aw2-label">{slide?.awardKey === "conn_smythe" ? "Playoff line" : "Season line"}</span>
            <StatLine items={stats.slice(0, 7)} size="lg" />
          </div>
        ) : null}
        {why.length ? (
          <div className="aw2-why">
            <span className="aw2-label">Why they won</span>
            {why.slice(0, 5).map((line) => (
              <p key={line}>{line}</p>
            ))}
          </div>
        ) : null}
        {components.length ? (
          <div>
            <span className="aw2-label">Against the field</span>
            <Bars
              items={components.map((c) => ({
                key: c.key || c.label,
                label: c.label,
                text: [formatStat(c.value, c.fmt), c.rank && c.of ? `${ordinal(c.rank)} of ${c.of}` : ""].filter(Boolean).join(" · "),
                pct: Math.max(0, Math.min(1, Number(c.pct))) * 100,
              }))}
            />
          </div>
        ) : null}
        {!hasCase ? <p className="aw2-muted">Detailed results weren't recorded for this season.</p> : null}
        {rule ? (
          <p className="aw2-copy">
            <b>How you win it. </b>
            {rule}
          </p>
        ) : null}
      </section>

      {hasResult ? (
        <section className="aw2-cite-result">
          {peak.length ? <StatLine items={peak} size="md" /> : null}
          {hasBallot && ev?.closest_of_night ? <p className="aw2-copy is-gold">Closest vote of the night.</p> : null}
          {voteTable.length ? (
            <VoteTable table={voteTable} format={ev?.ballot_format} playerIndex={playerIndex} team={team} />
          ) : order.length > 1 ? (
            <div className="aw2-votes">
              <span className="aw2-label">Final order</span>
              {order.map((card, i) => {
                const b = ballotFor(card, ev);
                const tail = b ? `${formatPoints(toNum(b.points))} pts` : formatStat(card?.displayValue ?? card?.display_value) || "";
                const who = team ? null : resolvePlayer(card?.player, playerIndex, card?.label);
                return (
                  <div key={`${card?.label}-${i}`} className={`aw2-votes-row is-3${isWinnerCard(card, slide) ? " is-winner" : ""}`}>
                    <span className="aw2-num">{toNum(card?.rank) ?? i + 1}</span>
                    <span className="aw2-votes-who">
                      {who ? <Headshot player={who} size={40} /> : <TeamLogo src={cardLogo(card)} size={34} />}
                      <span>
                        <strong>{card?.label}</strong>
                        {!team && cardTeam(card) ? <small>{cardTeam(card)}</small> : null}
                      </span>
                    </span>
                    <em className="aw2-num">{tail}</em>
                  </div>
                );
              })}
            </div>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}

function SummaryBody({ slides, playerIndex, userTeam }) {
  const cup = slides.find(isCupSlide) || null;
  const honours = slides.filter((x) => !isCupSlide(x));
  const firstPick = honours.find((x) => x.awardKey === "hart") || honours.find((x) => !isTeamSlide(x) && !isRosterSlide(x)) || honours[0] || null;
  const [pickId, setPickId] = useState(firstPick?.id || null);
  const pick = honours.find((x) => x.id === pickId) || firstPick;
  const ev = slideEvidence(pick);
  const team = isTeamSlide(pick);
  const roster = isRosterSlide(pick);
  const finalists = pick && !roster ? slideFinalists(pick) : [];
  const winnerCard = finalists.find((card) => isWinnerCard(card, pick)) || null;
  const player = pick && !team && !roster ? resolvePlayer(pick?.winnerPlayer || winnerCard?.player, playerIndex, pick?.winnerLabel) : null;
  const own = winnerCard && !team && !roster ? seasonLine(winnerCard, pick?.awardKey) : [];
  const stats = pick && !roster ? (own.length ? own : winnerStatRows(pick, ev).map((r) => ({ key: r.key, label: r.label, value: r.value }))).slice(0, 6) : [];
  const why = pick ? cleanLines(safeArray(ev?.why).length ? ev.why : pick?.whyTheyWon).slice(0, 2) : [];
  const runnersUp = [...finalists].filter((card) => !isWinnerCard(card, pick)).sort((a, b) => (toNum(a?.rank) ?? 99) - (toNum(b?.rank) ?? 99));
  const slots = roster ? rosterSlotsFor(pick, ev) : [];
  const voteRows = safeArray(ev?.voting_table).filter((r) => r && r.name).slice(0, 5);
  const groups = [
    ["Major awards", honours.filter(isMajorSlide)],
    ["More honours", honours.filter((x) => !isMajorSlide(x) && !isRosterSlide(x))],
    ["League selections", honours.filter(isRosterSlide)],
  ].filter(([, list]) => list.length);
  const mineWins = honours.filter((x) => !isRosterSlide(x) && userTeam?.name && (x.winnerTeamName === userTeam.name || (isTeamSlide(x) && x.winnerLabel === userTeam.name)));

  return (
    <div className="aw2-summary">
      {cup ? (
        <section className="aw2-champ is-summary">
          <TeamLogo src={winnerLogo(cup)} size={120} />
          <div className="aw2-champ-text">
            <span>Stanley Cup champions</span>
            <div className="aw2-display aw2-fit" style={fitStyle(cup.winnerLabel, 52)}>
              {cup.winnerLabel}
            </div>
            {mineWins.length ? (
              <p className="is-gold">
                Your club took home {mineWins.length === 1 ? "one honour" : `${mineWins.length} honours`}: {mineWins.map((x) => x.awardLabel).join(", ")}.
              </p>
            ) : null}
          </div>
        </section>
      ) : null}

      {pick ? (
        <section className={`aw2-spot${roster ? " is-roster" : ""}`} key={pick.id}>
          {roster ? null : (
            <div className="aw2-spot-face">
              {team ? <TeamLogo src={winnerLogo(pick)} size={170} /> : <Headshot player={player} size={180} />}
            </div>
          )}
          <div className="aw2-spot-main">
            <span className="aw2-label is-gold">{roster ? "League selection" : pick.awardLabel}</span>
            <div className="aw2-display aw2-fit" style={fitStyle(roster ? pick.awardLabel : pick.winnerLabel, 42)}>
              {roster ? pick.awardLabel : pick.winnerLabel}
            </div>
            {!team && !roster && pick.winnerTeamName ? (
              <div className="aw2-nominee-team is-lg">
                <TeamLogo src={winnerLogo(pick)} size={32} />
                <span>{pick.winnerTeamName}</span>
              </div>
            ) : null}
            {pick.stageLine ? <p className="aw2-muted">{pick.stageLine}</p> : null}
            <StatLine items={stats} size="md" />
            {why.length ? <p className="aw2-copy">{why.join(" ")}</p> : null}
            {runnersUp.length ? (
              <p className="aw2-copy">
                <b>Also in the running: </b>
                {runnersUp.map((c) => `${c.label}${cardTeam(c) ? ` (${cardTeam(c)})` : ""}`).join(", ")}
              </p>
            ) : null}
            {slots.length ? (
              <div className="aw2-spot-lineup">
                {slots.map((slot, i) => (
                  <div key={`${slot?.slot || i}-${slot?.label || i}`}>
                    <i>{slot?.slot || ""}</i>
                    <b>{slot?.label || ""}</b>
                    {slot?.teamName ? (
                      <span className="aw2-nominee-team">
                        <TeamLogo src={slot.teamLogoSrc} size={22} />
                        <span>{slot.teamName}</span>
                      </span>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : null}
          </div>
          {!roster && voteRows.length > 1 ? (
            <div className="aw2-spot-votes">
              <VoteTable table={voteRows} format={ev?.ballot_format} playerIndex={playerIndex} team={team} />
            </div>
          ) : null}
        </section>
      ) : null}

      {groups.map(([label, list]) => (
        <section key={label} className="aw2-honours">
          <div className="aw2-sectionhead">
            <div role="heading" aria-level={3}>{label}</div>
          </div>
          <div className="aw2-honour-grid">
            {list.map((slide) => {
              const mine = !isRosterSlide(slide) && userTeam?.name && (slide.winnerTeamName === userTeam.name || slide.winnerLabel === userTeam.name);
              return (
                <button
                  key={slide.id}
                  type="button"
                  className={`aw2-honour${pick && slide.id === pick.id ? " is-on" : ""}${mine ? " is-mine" : ""}`}
                  aria-pressed={Boolean(pick && slide.id === pick.id)}
                  onClick={() => setPickId(slide.id)}
                >
                  {isRosterSlide(slide) ? <span className="aw2-honour-blank" /> : <TeamLogo src={winnerLogo(slide)} size={40} />}
                  <span>
                    <i>{slide.awardLabel}</i>
                    <b>{isRosterSlide(slide) ? "See the lineup" : slide.winnerLabel}</b>
                  </span>
                </button>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}

/* ─────────────── screen ─────────────── */

/**
 * Permanent Awards Night franchise ceremony — Entry Draft broadcast floor model.
 */
export default function AwardsNight({ franchiseState = {}, eventData = {}, onContinue, onBack }) {
  const awards = useMemo(
    () => normalizeAwardsPayload(franchiseState, eventData),
    [franchiseState, eventData]
  );
  const slides = useMemo(
    () => safeArray(buildAwardsCeremonySlides(awards, franchiseState)),
    [awards, franchiseState]
  );
  const summary = useMemo(() => buildAwardsNightSummary(awards) || {}, [awards]);
  // Plain-language rule for each trophy (backend AWARD_CRITERIA), keyed like the slides.
  const ruleByKey = useMemo(() => {
    const out = {};
    safeArray(awards).forEach((a) => {
      if (a?.awardKey && typeof a.criteria === "string" && a.criteria) out[a.awardKey] = a.criteria;
    });
    return out;
  }, [awards]);
  const evidenceByKey = useMemo(() => {
    const out = {};
    safeArray(awards).forEach((a) => {
      const e = a?.evidence || a?.result?.evidence;
      if (a?.awardKey && e && typeof e === "object") out[a.awardKey] = e;
    });
    return out;
  }, [awards]);
  const indexGroups = useMemo(() => safeArray(buildCeremonyRailGroups(slides)), [slides]);
  const playerIndex = useMemo(() => buildPlayerIndex(franchiseState), [franchiseState]);
  const userTeamName = String(
    franchiseState?.user_team_name ||
      franchiseState?.user_team?.name ||
      franchiseState?.user_team?.team_name ||
      franchiseState?.team_name ||
      franchiseState?.team?.name ||
      ""
  );
  const userTeamId = franchiseState?.user_team_id ?? franchiseState?.userTeamId ?? franchiseState?.user_team?.team_id ?? null;
  const userTeam = useMemo(() => ({ name: userTeamName, id: userTeamId }), [userTeamName, userTeamId]);
  const fanPool = useMemo(() => buildFallbackAwardFans(24, "awards-night-live"), []);

  const preShowPool = useMemo(
    () =>
      safeArray(
        buildAwardsFanTweets(awards, {
          fans: fanPool,
          maxTweets: 48,
          tweetsPerAward: 3,
          seed: "awards-night-pre",
          spoilerFree: true,
          includeSummaryTweets: false,
        })
      ),
    [awards, fanPool]
  );

  const reactionPool = useMemo(
    () =>
      safeArray(
        buildAwardsFanTweets(awards, {
          fans: fanPool,
          maxTweets: 48,
          tweetsPerAward: 3,
          seed: "awards-night-post",
          spoilerFree: false,
          includeSummaryTweets: false,
        })
      ),
    [awards, fanPool]
  );

  const [activeIndex, setActiveIndex] = useState(0);
  const [phase, setPhase] = useState(PHASE.GATE);
  const [finalistShown, setFinalistShown] = useState(0);
  const [paused, setPaused] = useState(false);
  const [voiceOn, setVoiceOn] = useState(false); // Awards Night is read, not spoken
  const [revealedIds, setRevealedIds] = useState(() => new Set());
  const [reactionFeed, setReactionFeed] = useState([]);
  const [continuing, setContinuing] = useState(false);
  const [continueError, setContinueError] = useState("");

  const timerRef = useRef(null);
  const reactionTimerRef = useRef(null);
  const spokenRef = useRef("");
  const revealedRef = useRef(revealedIds);
  revealedRef.current = revealedIds;

  const reducedMotion = useMemo(
    () =>
      typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    []
  );

  const activeSlide = slides[activeIndex] || null;
  // Ceremony slides don't carry the award's evidence; fall back to the award with that key.
  const ev = slideEvidence(activeSlide) || evidenceByKey[activeSlide?.awardKey] || null;
  const finalists = useMemo(() => slideFinalists(activeSlide), [activeSlide]);
  const roster = isRosterSlide(activeSlide);
  const hasFinalistPhase = finalists.length > 0 && !roster;
  const inCeremony = phase !== PHASE.GATE && phase !== PHASE.SUMMARY;
  const awardRevealed = Boolean(
    activeSlide && (revealedIds.has(activeSlide.id) || isRevealedPhase(phase))
  );

  /* dev diagnostics: fallback awards, missing finalists, missing logos, unresolved headshots */
  useEffect(() => {
    if (process.env.NODE_ENV === "production") return;
    let finalistsWithoutLogo = 0;
    let playersWithoutRosterMatch = 0;
    slides.forEach((slide) => {
      slideFinalists(slide).forEach((card) => {
        if (!cardLogo(card)) finalistsWithoutLogo += 1;
        if (!isTeamSlide(slide) && !lookupPlayer(card?.player, playerIndex, card?.label)) {
          playersWithoutRosterMatch += 1;
        }
      });
    });
    console.debug("[AwardsNight]", {
      calculationFallbackAwards: countCalculationFallbackAwards(awards),
      awardsMissingFinalists: listAwardsMissingFinalists(slides),
      finalistsWithoutLogo,
      playersWithoutRosterMatch,
      playerIndexEntries: playerIndex.size,
    });
  }, [awards, playerIndex, slides]);

  useEffect(() => {
    setActiveIndex(0);
    setPhase(PHASE.GATE);
    setFinalistShown(0);
    setRevealedIds(new Set());
    setReactionFeed([]);
    spokenRef.current = "";
  }, [slides.length]);

  useEffect(() => {
    if (!activeSlide || !isRevealedPhase(phase)) return;
    setRevealedIds((prev) => (prev.has(activeSlide.id) ? prev : new Set(prev).add(activeSlide.id)));
  }, [activeSlide, phase]);

  /* reactions trickle in after the reveal — only this award's, never another sealed winner */
  useEffect(() => {
    if (reactionTimerRef.current) window.clearTimeout(reactionTimerRef.current);
    if (!activeSlide || !awardRevealed) {
      setReactionFeed([]);
      return undefined;
    }
    const otherSealed = slides
      .filter((s) => s.id !== activeSlide.id && !revealedRef.current.has(s.id))
      .map((s) => s.winnerLabel)
      .filter(Boolean);
    const queue = reactionPool
      .filter(
        (t) =>
          t?.awardKey === activeSlide.awardKey &&
          !otherSealed.some((w) => mentionsWinner(String(t?.text || ""), w))
      )
      .slice(0, 8);
    setReactionFeed([]);
    let i = 0;
    const tick = () => {
      if (i >= queue.length) return;
      const tweet = queue[i];
      i += 1;
      setReactionFeed((prev) => [tweet, ...prev].slice(0, 8));
      reactionTimerRef.current = window.setTimeout(tick, reducedMotion ? 0 : 700);
    };
    tick();
    return () => {
      if (reactionTimerRef.current) window.clearTimeout(reactionTimerRef.current);
    };
  }, [activeSlide, awardRevealed, reactionPool, reducedMotion, slides]);

  const preShowTweets = useMemo(() => {
    const sealed = slides
      .filter((s) => !revealedIds.has(s.id))
      .map((s) => s.winnerLabel)
      .filter(Boolean);
    const scoped =
      activeSlide && inCeremony ? preShowPool.filter((t) => t?.awardKey === activeSlide.awardKey) : [];
    const pool = scoped.length ? scoped : preShowPool;
    return pool
      .filter((t) => !sealed.some((w) => mentionsWinner(String(t?.text || ""), w)))
      .slice(0, 6);
  }, [activeSlide, inCeremony, preShowPool, revealedIds, slides]);

  const summaryTweets = useMemo(
    () =>
      safeArray(summary.fanTweets).length
        ? summary.fanTweets.slice(0, 8)
        : reactionPool.slice(0, 8),
    [reactionPool, summary]
  );

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const cancelSpeech = useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) window.speechSynthesis.cancel();
  }, []);

  useEffect(() => () => cancelSpeech(), [cancelSpeech]);

  const speak = useCallback(
    (text) => {
      if (!voiceOn || !text || typeof window === "undefined" || !window.speechSynthesis) return;
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 0.94;
      window.speechSynthesis.speak(utter);
    },
    [voiceOn]
  );

  /* no cleanup here — cancelling on phase change cut the line off mid-sentence */
  useEffect(() => {
    if (!voiceOn || !activeSlide || phase !== PHASE.REVEAL) return;
    const key = `${activeSlide.id}:reveal`;
    if (spokenRef.current === key) return;
    spokenRef.current = key;
    if (roster) speak(`The ${activeSlide.awardLabel}.`);
    else if (activeSlide.winnerLabel) speak(`The ${activeSlide.awardLabel} goes to ${activeSlide.winnerLabel}.`);
  }, [activeSlide, phase, roster, speak, voiceOn]);

  const toggleVoice = useCallback(() => {
    if (voiceOn) cancelSpeech();
    setVoiceOn(!voiceOn);
  }, [cancelSpeech, voiceOn]);

  const goToSummary = useCallback(() => {
    clearTimer();
    cancelSpeech();
    setPhase(PHASE.SUMMARY);
  }, [cancelSpeech, clearTimer]);

  const jumpToAward = useCallback(
    (indexValue) => {
      if (!slides.length) return;
      const next = Math.max(0, Math.min(slides.length - 1, Number(indexValue) || 0));
      clearTimer();
      cancelSpeech();
      spokenRef.current = "";
      setActiveIndex(next);
      setFinalistShown(0);
      setPhase(PHASE.TITLE);
    },
    [cancelSpeech, clearTimer, slides.length]
  );

  const beginCeremony = useCallback(() => {
    if (!slides.length) return;
    clearTimer();
    setFinalistShown(0);
    setPhase(PHASE.TITLE);
  }, [clearTimer, slides.length]);

  const advanceFromCitation = useCallback(() => {
    if (activeIndex >= slides.length - 1) {
      goToSummary();
      return;
    }
    setActiveIndex((i) => i + 1);
    setFinalistShown(0);
    setPhase(PHASE.TITLE);
  }, [activeIndex, goToSummary, slides.length]);

  const advancePhase = useCallback(() => {
    clearTimer();
    if (phase === PHASE.GATE) {
      beginCeremony();
      return;
    }
    if (phase === PHASE.TITLE) {
      if (hasFinalistPhase) {
        setFinalistShown(1);
        setPhase(PHASE.FINALISTS);
      } else {
        setPhase(PHASE.SEAL);
      }
      return;
    }
    if (phase === PHASE.FINALISTS) {
      if (finalistShown < finalists.length) setFinalistShown((n) => n + 1);
      else setPhase(PHASE.SEAL);
      return;
    }
    if (phase === PHASE.SEAL) {
      setPhase(PHASE.REVEAL);
      return;
    }
    if (phase === PHASE.REVEAL) {
      setPhase(PHASE.CITATION);
      return;
    }
    if (phase === PHASE.CITATION) advanceFromCitation();
  }, [
    advanceFromCitation,
    beginCeremony,
    clearTimer,
    finalistShown,
    finalists.length,
    hasFinalistPhase,
    phase,
  ]);

  const revealNow = useCallback(() => {
    clearTimer();
    setFinalistShown(finalists.length);
    setPhase(PHASE.REVEAL);
  }, [clearTimer, finalists.length]);

  const skipAward = useCallback(() => {
    clearTimer();
    cancelSpeech();
    if (activeIndex >= slides.length - 1) {
      goToSummary();
      return;
    }
    setActiveIndex((i) => i + 1);
    setFinalistShown(0);
    setPhase(PHASE.TITLE);
  }, [activeIndex, cancelSpeech, clearTimer, goToSummary, slides.length]);

  useEffect(() => {
    if (paused || !inCeremony) return undefined;
    const delay = TIMING[phase];
    if (!delay) return undefined;
    timerRef.current = window.setTimeout(() => advancePhase(), delay);
    return clearTimer;
  }, [advancePhase, clearTimer, inCeremony, paused, phase]);

  useEffect(() => {
    const onKey = (event) => {
      if (event.target?.closest?.("input, select, textarea")) return;
      if (event.key === "Escape") {
        if (phase !== PHASE.SUMMARY) goToSummary();
        return;
      }
      if (phase === PHASE.SUMMARY) return;
      if (event.key === " " || event.key === "Spacebar") {
        if (event.target?.closest?.("button")) return;
        event.preventDefault();
        advancePhase();
        return;
      }
      if (phase === PHASE.GATE) return;
      if (event.key === "ArrowRight") {
        event.preventDefault();
        skipAward();
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        jumpToAward(activeIndex - 1);
      } else if (event.key === "p" || event.key === "P") {
        setPaused((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeIndex, advancePhase, goToSummary, jumpToAward, phase, skipAward]);

  const handleContinue = useCallback(async () => {
    if (continuing || typeof onContinue !== "function") return;
    setContinuing(true);
    setContinueError("");
    try {
      await onContinue();
    } catch (error) {
      const message =
        error?.response?.data?.detail ||
        error?.response?.data?.message ||
        (error?.message && /network/i.test(String(error.message))
          ? "Network error advancing the offseason. Return to Hub and use Resume Offseason Timeline."
          : null) ||
        error?.message ||
        "Could not continue the offseason.";
      setContinueError(String(message));
    } finally {
      setContinuing(false);
    }
  }, [continuing, onContinue]);

  /* ── stage head ── */
  let headNum = "";
  let headState = "";
  let headLive = false;
  let headTitle = "";
  const headMeta = [];
  if (!slides.length) {
    headNum = "00";
    headState = "No results";
    headTitle = "Awards Night";
  } else if (phase === PHASE.GATE) {
    headNum = pad2(slides.length);
    headState = PHASE_LABEL[PHASE.GATE];
    headTitle = "Tonight's card";
    const majorCount = slides.filter(isMajorSlide).length;
    const userNods = slides
      .filter((sl) => !isRosterSlide(sl) && !isTeamSlide(sl))
      .reduce((n, sl) => n + slideFinalists(sl).filter((c) => isUserCard(c, userTeam)).length, 0);
    headMeta.push({
      text: majorCount ? `${majorCount} major awards on media ballots, ${slides.length - majorCount} more honours` : `${slides.length} honours`,
      tone: "strong",
    });
    headMeta.push({ text: seasonLabel(franchiseState) });
    if (userNods) headMeta.push({ text: `Your club: ${userNods} finalist nod${userNods === 1 ? "" : "s"}`, tone: "gold" });
  } else if (phase === PHASE.SUMMARY) {
    headNum = pad2(slides.length);
    headState = PHASE_LABEL[PHASE.SUMMARY];
    headLive = true;
    headTitle = "The NHL Awards";
    headMeta.push({ text: `${slides.length} honours presented`, tone: "strong" });
    headMeta.push({ text: seasonLabel(franchiseState) });
  } else {
    headNum = pad2(activeIndex + 1);
    headState = `${categoryLabel(activeSlide)} · ${PHASE_LABEL[phase]}`;
    headLive = awardRevealed;
    headTitle = activeSlide?.awardLabel || "";
    if (activeSlide?.stageLine) headMeta.push({ text: activeSlide.stageLine, tone: "strong" });
    if (awardRevealed && !roster && activeSlide?.winnerLabel) {
      headMeta.push({ text: `Winner — ${activeSlide.winnerLabel}`, tone: "gold" });
    }
  }
  const upNextStart = phase === PHASE.GATE ? 0 : activeIndex + 1;
  const upNext =
    phase === PHASE.GATE || inCeremony ? slides.slice(upNextStart, upNextStart + 3) : [];

  /* ── stage body ── */
  let body;
  if (!slides.length) {
    body = (
      <p className="aw2-muted aw2-pad">
        No award results were found. Complete the playoffs to reveal the season's hardware.
      </p>
    );
  } else if (phase === PHASE.GATE) {
    body = (
      <GateBody slides={slides} playerIndex={playerIndex} userTeam={userTeam} onPick={jumpToAward} evidenceByKey={evidenceByKey} />
    );
  } else if (phase === PHASE.SUMMARY) {
    body = <SummaryBody slides={slides} playerIndex={playerIndex} userTeam={userTeam} />;
  } else if (phase === PHASE.TITLE || (phase === PHASE.SEAL && !hasFinalistPhase && !roster)) {
    body = (
      <TitleBody
        key={`title-${activeSlide.id}`}
        slide={activeSlide}
        ev={ev}
        finalists={roster ? [] : finalists}
        rule={ruleByKey[activeSlide?.awardKey] || ""}
        playerIndex={playerIndex}
        userTeam={userTeam}
      />
    );
  } else if (roster) {
    body = <RosterBody key={`roster-${activeSlide.id}`} slide={activeSlide} ev={ev} phase={phase} playerIndex={playerIndex} />;
  } else if (hasFinalistPhase && phase !== PHASE.CITATION) {
    body = (
      <FinalistsBody
        key={`finalists-${activeSlide.id}`}
        slide={activeSlide}
        ev={ev}
        finalists={finalists}
        phase={phase}
        shown={finalistShown}
        playerIndex={playerIndex}
        userTeam={userTeam}
      />
    );
  } else {
    body = (
      <CitationBody
        key={`citation-${activeSlide.id}`}
        slide={activeSlide}
        ev={ev}
        rule={ruleByKey[activeSlide?.awardKey] || ""}
        finalists={finalists}
        playerIndex={playerIndex}
        userTeam={userTeam}
      />
    );
  }

  /* ── fan pulse ── */
  const pulseStage = !slides.length ? "gate" : phase === PHASE.SUMMARY ? "summary" : phase === PHASE.GATE ? "gate" : awardRevealed ? "post" : "pre";
  const pulseMode = pulseStage === "post" || pulseStage === "summary" ? "reactions" : "pre-show";
  const pulseItems = slides.length ? buildPulse({ stage: pulseStage, slide: activeSlide, slides, ev }) : [];
  const pulseEmpty = "Reactions land once the night begins.";

  const displayRevealed =
    phase === PHASE.SUMMARY ? new Set(slides.map((s) => s.id)) : revealedIds;
  const isLast = activeIndex >= slides.length - 1;
  const steps = !inCeremony
    ? []
    : roster
      ? [PHASE.TITLE, PHASE.REVEAL, PHASE.CITATION]
      : hasFinalistPhase
        ? [PHASE.TITLE, PHASE.FINALISTS, PHASE.SEAL, PHASE.REVEAL, PHASE.CITATION]
        : [PHASE.TITLE, PHASE.SEAL, PHASE.REVEAL, PHASE.CITATION];
  const phaseForSteps = roster && phase === PHASE.SEAL ? PHASE.TITLE : phase;
  const stepAt = steps.indexOf(phaseForSteps);
  const presentedCount = slides.filter((s) => displayRevealed.has(s.id)).length;

  return (
    <section className="aw2-root" aria-label="Awards Night">
      <header className="aw2-top">
        <div className="aw2-top-id">
          <div className="aw2-display aw2-top-name" role="heading" aria-level={1}>
            Awards Night
          </div>
          <span>{seasonLabel(franchiseState)}</span>
        </div>
        <nav className="aw2-steps" aria-label="Offseason timeline">
          {safeArray(SEASON_MILESTONES).map((milestone, i, list) => {
            const at = list.findIndex((m) => m.id === "awards");
            const state = milestone.id === "awards" ? "is-now" : i < at ? "is-past" : "";
            return (
              <span key={milestone.id} className={state} aria-current={milestone.id === "awards" ? "step" : undefined}>
                {milestone.label}
              </span>
            );
          })}
        </nav>
        <div className="aw2-top-actions">
          <button type="button" className="aw2-quiet" onClick={() => setPaused((v) => !v)} disabled={!inCeremony}>
            {paused ? "Resume" : "Pause"}
          </button>
          {phase !== PHASE.SUMMARY && slides.length ? (
            <button type="button" className="aw2-quiet" onClick={goToSummary}>
              Skip to summary
            </button>
          ) : null}
          {typeof onBack === "function" ? (
            <button type="button" className="aw2-quiet" onClick={onBack}>
              Leave to Hub
            </button>
          ) : null}
        </div>
      </header>

      <div className="aw2-floor">
        <CeremonyLog
          groups={indexGroups}
          slidesCount={slides.length}
          activeIndex={inCeremony ? activeIndex : -1}
          revealedIds={displayRevealed}
          onSelect={jumpToAward}
        />

        <section className="aw2-stage" aria-live="polite">
          <header className={`aw2-head${headLive ? " is-live" : ""}`}>
            <div className="aw2-head-num">
              <b>{headNum}</b>
              {inCeremony ? <span>of {pad2(slides.length)}</span> : <span>{phase === PHASE.SUMMARY ? "presented" : "awards"}</span>}
            </div>
            <div className="aw2-head-main">
              <span className="aw2-head-state">{headState}</span>
              <div className="aw2-display aw2-fit" style={fitStyle(headTitle, 44)} role="heading" aria-level={2}>
                {headTitle}
              </div>
              {headMeta.length ? (
                <div className="aw2-head-meta">
                  {headMeta.map((item) => (
                    <span key={item.text} className={item.tone === "gold" ? "is-gold" : item.tone === "strong" ? "is-strong" : ""}>
                      {item.text}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>
            {steps.length ? (
              <ol className="aw2-phases" aria-label="Award progress">
                {steps.map((s, i) => (
                  <li key={s} className={i < stepAt ? "is-done" : i === stepAt ? "is-now" : ""}>
                    {PHASE_LABEL[s]}
                  </li>
                ))}
              </ol>
            ) : upNext.length ? (
              <div className="aw2-next">
                <span>Opening the night</span>
                {upNext.map((slide, i) => (
                  <button key={slide.id} type="button" onClick={() => jumpToAward(upNextStart + i)}>
                    <b>{pad2(upNextStart + i + 1)}</b>
                    {slide.awardLabel}
                  </button>
                ))}
              </div>
            ) : null}
          </header>

          <div className="aw2-body">{body}</div>

          <footer className="aw2-dock">
            <div className="aw2-dock-progress">
              <div className="aw2-dock-text">
                {phase === PHASE.SUMMARY || !slides.length ? (
                  <>
                    <b>{slides.length ? `All ${slides.length} awards presented` : "Nothing to present"}</b>
                  </>
                ) : phase === PHASE.GATE ? (
                  <>
                    <b>Ceremony ready</b>
                    <span>{slides.length} awards tonight · space bar starts the show</span>
                  </>
                ) : (
                  <>
                    <b>
                      Award {activeIndex + 1} of {slides.length}: {PHASE_LABEL[phase]}
                    </b>
                    <span>
                      {paused ? "Paused · " : ""}
                      {presentedCount} presented
                      {upNext.length ? ` · Next: ${upNext[0].awardLabel}` : ""}
                    </span>
                  </>
                )}
              </div>
              {slides.length ? (
                <div className="aw2-segs" aria-hidden="true">
                  {slides.map((s, i) => (
                    <i
                      key={s.id}
                      className={[
                        displayRevealed.has(s.id) ? "is-done" : "",
                        inCeremony && i === activeIndex ? "is-now" : "",
                      ]
                        .filter(Boolean)
                        .join(" ")}
                    />
                  ))}
                </div>
              ) : null}
              {continueError ? <p className="aw2-dock-error">{continueError}</p> : null}
            </div>
            <div className="aw2-dock-actions">
              {phase === PHASE.SUMMARY || !slides.length ? (
                <button type="button" className="aw2-go" onClick={handleContinue} disabled={continuing}>
                  {continuing ? "Continuing…" : slides.length ? "Continue to Retirements" : "Continue"}
                </button>
              ) : phase === PHASE.GATE ? (
                <button type="button" className="aw2-go" onClick={beginCeremony}>
                  Begin ceremony
                </button>
              ) : (
                <>
                  <button type="button" className="aw2-quiet" onClick={revealNow} disabled={isRevealedPhase(phase)}>
                    Reveal now
                  </button>
                  <button type="button" className="aw2-quiet" onClick={skipAward}>
                    Skip award
                  </button>
                  <button type="button" className="aw2-go" onClick={advancePhase}>
                    {phase === PHASE.CITATION ? (isLast ? "Finish ceremony" : "Next award") : "Continue"}
                  </button>
                </>
              )}
            </div>
          </footer>
        </section>

        <FanPulse mode={pulseMode} items={pulseItems} emptyText={pulseEmpty} feedKey={`${pulseStage}-${activeSlide?.id || "gate"}`} />
      </div>
    </section>
  );
}
