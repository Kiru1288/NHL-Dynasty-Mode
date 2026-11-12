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

/* ─────────────── small components ─────────────── */

function TeamLogo({ src, size = "sm" }) {
  if (!src) return null;
  return (
    <span className={`an-logo ${size}`} aria-hidden="true">
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

function Headshot({ player, size = "md" }) {
  if (!player) return null;
  return (
    <div className="an-headshot">
      <PlayerHeadshot player={player} size={size} variant="hero" mood="neutral" animate="in" />
    </div>
  );
}

function StatTable({ rows }) {
  if (!rows.length) return null;
  const withRank = rows.some((row) => row.rank);
  return (
    <table className="an-stat-table">
      <thead>
        <tr>
          <th>Stat</th>
          <th>Value</th>
          {withRank ? <th>Rank</th> : null}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.key}>
            <td>{row.label}</td>
            <td className="an-num">{row.value}</td>
            {withRank ? <td className="an-rank">{row.rank}</td> : null}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function BallotRail({ points, leader, fpv }) {
  if (points === null || !leader) return null;
  const width = Math.max(2, Math.min(100, (points / leader) * 100));
  return (
    <div className="an-ballot">
      <div className="an-rail">
        <i className="an-rail__fill" style={{ width: `${width}%` }} />
      </div>
      <div className="an-ballot__cap an-num">
        {formatPoints(points)} pts{fpv !== null ? ` · ${fpv} first-place` : ""}
      </div>
    </div>
  );
}

function ComponentList({ items }) {
  return (
    <div className="an-facts">
      {items.map((c) => {
        const value = formatStat(c.value, c.fmt);
        const rank = c.rank && c.of ? `${ordinal(c.rank)} of ${c.of}` : "";
        const pct = Math.max(0, Math.min(1, Number(c.pct)));
        return (
          <div key={c.key || c.label}>
            <div className="an-fact-head">
              <span>{c.label}</span>
              <strong className="an-num">{[value, rank].filter(Boolean).join(" · ")}</strong>
            </div>
            <div className="an-track">
              <i style={{ width: `${pct * 100}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ─────────────── columns ─────────────── */

function CeremonyLog({ groups, slidesCount, activeIndex, revealedIds, onSelect }) {
  return (
    <aside className="an-log" aria-label="Ceremony order">
      <div className="an-section-head">
        <div>
          <h3 className="an-section-title">Ceremony Order</h3>
          <p className="an-section-meta">{slidesCount} awards</p>
        </div>
      </div>
      <div className="an-log-scroll">
        {groups.map((group, g) => (
          <div key={group?.id || group?.label || g}>
            {group?.label ? <div className="an-log-round">{group.label}</div> : null}
            {safeArray(group?.items).map(({ slide, index, offRail }) => {
              if (!slide) return null;
              const clickable = !offRail && Number.isInteger(index);
              const revealed = Boolean(offRail) || revealedIds.has(slide.id);
              const selected = clickable && index === activeIndex;
              const sub = revealed ? (isRosterSlide(slide) ? "Lineup named" : slide.winnerLabel) : selected ? "On stage" : "Sealed";
              const classes = [
                "an-log-row",
                selected ? "is-selected" : "",
                revealed ? "is-revealed" : "",
                selected && revealed ? "is-live" : "",
                isCupSlide(slide) ? "is-cup" : "",
                isMajorSlide(slide) ? "is-major" : "",
              ]
                .filter(Boolean)
                .join(" ");
              return (
                <button
                  key={slide.id}
                  type="button"
                  className={classes}
                  disabled={!clickable}
                  onClick={() => onSelect(index)}
                >
                  <span className="an-log-num">{clickable ? pad2(index + 1) : ""}</span>
                  <span className="an-log-logo">
                    {revealed && !isRosterSlide(slide) ? <TeamLogo src={winnerLogo(slide)} size="sm" /> : null}
                  </span>
                  <span className="an-log-body">
                    <strong>{slide.awardLabel}</strong>
                    <span>{sub}</span>
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
    MAJOR_KEYS.forEach((key) => {
      const sl = slides.find((x) => x.awardKey === key);
      if (!sl) return;
      const cards = slideFinalists(sl);
      if (cards.length < 2) return;
      const fav = cards[hashOf(`${key}-fav`) % cards.length];
      push(
        "fan",
        pickOne(key, [
          `${sl.awardLabel} finalists are ${joinNames(cards.map((c) => c.label))}. I'd give it to ${lastName(fav.label)}.`,
          `If ${lastName(fav.label)} doesn't win the ${sl.awardLabel} tonight I'm going to be annoyed.`,
          `${joinNames(cards.map((c) => lastName(c.label)))} for the ${sl.awardLabel}. Any of the three would be a fair winner.`,
        ]),
        { who: fanOf(cardTeam(fav)), logo: cardLogo(fav) }
      );
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
    cards.forEach((card) => {
      const row = finalistStatRows(card, ev)[0];
      const num = row ? ` ${row.value} ${row.label} speaks for itself.` : "";
      push(
        "fan",
        pickOne(`${award}-${card.label}`, [
          `${card.label} should win the ${award}.${num}`,
          `Put me down for ${lastName(card.label)}.${num}`,
          `It has to be ${lastName(card.label)} tonight.${num}`,
        ]),
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
    <aside className="an-board" aria-label="Fan Pulse">
      <div className="an-section-head">
        <div>
          <h3 className="an-section-title">Fan Pulse</h3>
          <p className="an-section-meta">{mode === "reactions" ? "Reactions" : "Before the envelope"}</p>
        </div>
      </div>
      <div className="an-pulse-lane an-pulse-list" key={feedKey}>
        {items.length ? (
          items.map((item, i) => (
            <article key={`${item.kind}-${i}`} className={`an-pulse is-${item.kind}`} style={{ animationDelay: `${i * 90}ms` }}>
              {item.kind === "fan" || item.kind === "debate" ? (
                <>
                  <header className="an-pulse-head">
                    {item.logo ? <TeamLogo src={item.logo} size="sm" /> : <span className="an-pulse-dot" aria-hidden="true" />}
                    <b>{item.who || "Around the league"}</b>
                    {item.kind === "debate" ? <em>{PULSE_TAG.debate}</em> : null}
                  </header>
                  <p>{item.text}</p>
                  <footer className="an-pulse-foot">{item.likes.toLocaleString()} likes</footer>
                </>
              ) : (
                <>
                  <header className="an-pulse-head">
                    <em>{PULSE_TAG[item.kind]}</em>
                    {item.logo ? <TeamLogo src={item.logo} size="sm" /> : null}
                  </header>
                  <p>{item.text}</p>
                </>
              )}
            </article>
          ))
        ) : (
          <p className="an-muted">{emptyText}</p>
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

function GateBody({ slides, playerIndex, userTeamName, onPick }) {
  const indexOf = (slide) => slides.findIndex((x) => x.id === slide.id);
  const majors = MAJOR_KEYS.map((k) => slides.find((x) => x.awardKey === k)).filter(Boolean);
  const decided = slides.filter(isDecidedSlide);
  const rest = slides.filter((x) => !isMajorSlide(x) && !isDecidedSlide(x));
  const mine = [];
  if (userTeamName) {
    slides.forEach((slide) => {
      if (isRosterSlide(slide) || isTeamSlide(slide)) return;
      slideFinalists(slide).forEach((card) => {
        if (cardTeam(card) && cardTeam(card) === userTeamName) mine.push({ slide, card });
      });
    });
  }
  return (
    <div className="an-card an-gate">
      {decided.length ? (
        <div className="an-gate-decided">
          {decided.map((slide) => (
            <div key={slide.id} className={`an-decided${isCupSlide(slide) ? " is-cup" : ""}`}>
              <TeamLogo src={winnerLogo(slide)} size={isCupSlide(slide) ? "hero" : "md"} />
              <div>
                <small>{isCupSlide(slide) ? "Stanley Cup champions" : slide.awardLabel}</small>
                <strong>{slide.winnerLabel}</strong>
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {mine.length ? (
        <div className="an-gate-mine">
          <span className="an-gate-kicker">Your club tonight</span>
          {mine.map(({ slide, card }) => (
            <button key={`${slide.id}-${card.label}`} type="button" className="an-mine-chip" onClick={() => onPick?.(indexOf(slide))}>
              <b>{card.label}</b>
              <i>{slide.awardLabel} finalist</i>
            </button>
          ))}
        </div>
      ) : null}

      {majors.length ? <h4 className="an-gate-head">The major awards</h4> : null}
      <div className="an-gate-majors">
        {majors.map((slide) => {
          const team = isTeamSlide(slide);
          const finalists = slideFinalists(slide);
          return (
            <button key={slide.id} type="button" className="an-major" onClick={() => onPick?.(indexOf(slide))}>
              <span className="an-major-name">{slide.awardLabel}</span>
              {slide.stageLine ? <span className="an-major-line">{slide.stageLine}</span> : null}
              <span className="an-major-faces">
                {finalists.map((card) => (
                  <span key={card.label} className="an-major-face">
                    {team ? (
                      <TeamLogo src={cardLogo(card)} size="md" />
                    ) : (
                      <Headshot player={resolvePlayer(card?.player, playerIndex, card?.label)} size="sm" />
                    )}
                    <b>{lastName(card.label)}</b>
                    <i>{cardTeam(card)}</i>
                  </span>
                ))}
              </span>
              <span className="an-major-seal">{finalists.length > 1 ? `${finalists.length} finalists · sealed` : "Sealed"}</span>
            </button>
          );
        })}
      </div>

      {rest.length ? <h4 className="an-gate-head">Also tonight</h4> : null}
      <div className="an-gate-rest">
        {rest.map((slide) => {
          const names = isRosterSlide(slide) ? [] : slideFinalists(slide).map((c) => lastName(c.label));
          return (
            <button key={slide.id} type="button" className="an-rest-row" onClick={() => onPick?.(indexOf(slide))}>
              <b>{slide.awardLabel}</b>
              <i>{isRosterSlide(slide) ? "Six-man lineup" : names.length > 1 ? names.join(" · ") : names.length === 1 ? "Decided by the numbers" : categoryLabel(slide)}</i>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function TitleBody({ slide, ev, finalistsCount, rule = "" }) {
  const criteria = safeArray(ev?.criteria).filter((c) => c?.label && toNum(c.weight) !== null);
  const excluded = safeArray(ev?.excluded_criteria).filter((c) => c?.label);
  const eligibility =
    cleanLines([ev?.eligibility, slide?.eligibilitySummary, slide?.eligibility_summary])[0] || "";
  const method = METHOD_LABEL[ev?.method] || (isRosterSlide(slide) ? METHOD_LABEL.selection : "");
  const voters = toNum(ev?.winner?.ballot?.voter_count);
  const poolSize = toNum(ev?.pool?.size);
  const facts = [
    method ? ["Format", method] : null,
    voters !== null ? ["Voters", String(voters)] : null,
    poolSize !== null ? ["Eligible", `${poolSize}${ev?.pool?.noun ? ` ${ev.pool.noun}` : ""}`] : null,
    finalistsCount ? ["Finalists", String(finalistsCount)] : null,
  ].filter(Boolean);
  const maxWeight = Math.max(0, ...criteria.map((c) => Number(c.weight))) || 1;

  return (
    <div className="an-card">
      <div className={`an-card-grid${facts.length ? "" : " is-two"}`}>
        <section className="an-z-identity">
          <p className="an-pos">{categoryLabel(slide)}</p>
          <h2 className="an-name">{slide?.awardLabel}</h2>
          {slide?.stageLine ? <p className="an-club">{slide.stageLine}</p> : null}
        </section>

        <section className="an-z-eval">
          <div>
            <h4>How it's decided</h4>
            {criteria.length ? (
              <div className="an-facts">
                {criteria.map((c) => (
                  <div key={c.key || c.label}>
                    <div className="an-fact-head">
                      <span>{c.label}</span>
                      <strong className="an-num">{Math.round(Number(c.weight) * 100)}%</strong>
                    </div>
                    <div className="an-track">
                      <i style={{ width: `${(Number(c.weight) / maxWeight) * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            ) : eligibility && !rule ? (
              <p className="an-club">{eligibility}</p>
            ) : !rule ? (
              <p className="an-muted">Voting criteria weren't published with this season's results.</p>
            ) : null}
            {rule ? (
              <p className="an-criteria">
                <b>How you win it</b>
                {rule}
              </p>
            ) : null}
            {rule && eligibility ? <p className="an-club">This season: {eligibility}</p> : null}
            {excluded.map((c) => (
              <p key={c.key || c.label} className="an-excluded">
                {c.label} — not tracked this season
              </p>
            ))}
          </div>
        </section>

        {facts.length ? (
          <section className="an-z-result">
            <p className="an-result-title">Format</p>
            <div className="an-result-meta">
              {facts.map(([label, value]) => (
                <div key={label}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          </section>
        ) : null}
      </div>
    </div>
  );
}

function FinalistsBody({ slide, ev, finalists, phase, shown, playerIndex }) {
  const revealed = isRevealedPhase(phase);
  const team = isTeamSlide(slide);
  const ballots = finalists.map((card) => ballotFor(card, ev));
  const leader = Math.max(0, ...ballots.map((b) => toNum(b?.points) ?? 0));

  return (
    <div className="an-card">
      <div
        className="an-finalists"
        style={{ gridTemplateColumns: `repeat(${Math.max(1, finalists.length)}, minmax(0, 1fr))` }}
      >
        {finalists.map((card, i) => {
          const winner = revealed && isWinnerCard(card, slide);
          const classes = [
            "an-finalist",
            phase === PHASE.FINALISTS && i >= shown ? "is-pending" : "is-in",
            phase === PHASE.SEAL ? "is-sealed" : "",
            winner ? "is-winner" : "",
            revealed && !winner ? "is-dim" : "",
          ]
            .filter(Boolean)
            .join(" ");
          const player = team ? null : resolvePlayer(card?.player, playerIndex, card?.label);
          const rows = finalistStatRows(card, ev).slice(0, 3);
          const ballot = ballots[i];
          const teamName = cardTeam(card);
          const position = cardPosition(card) || player?.position || "";
          const extra = rows.length ? null : formatStat(card?.stat) || formatStat(card?.subline);

          return (
            <article key={`${card?.label}-${i}`} className={classes}>
              <div className="an-finalist__label">{winner ? "Winner" : "Finalist"}</div>
              {team ? <TeamLogo src={cardLogo(card)} size="xl" /> : <Headshot player={player} size="md" />}
              <h3 className="an-name is-sm">{card?.label}</h3>
              {!team && teamName ? (
                <p className="an-vitals">
                  <TeamLogo src={cardLogo(card)} size="sm" />
                  <span>{teamName}</span>
                </p>
              ) : null}
              {position ? <p className="an-pos">{position}</p> : null}
              {rows.length ? <StatTable rows={rows} /> : extra ? <p className="an-club">{extra}</p> : null}
              {revealed && ballot && leader > 0 ? (
                <BallotRail
                  points={toNum(ballot.points)}
                  leader={leader}
                  fpv={toNum(ballot.first_place_votes)}
                />
              ) : null}
            </article>
          );
        })}
      </div>
    </div>
  );
}

function RosterBody({ slide, ev, phase }) {
  const revealed = isRevealedPhase(phase);
  const slots = rosterSlotsFor(slide, ev);
  if (!slots.length) {
    return (
      <div className="an-card">
        <p className="an-muted">Selections weren't recorded for this season.</p>
      </div>
    );
  }
  return (
    <div className="an-card">
      <div
        className="an-table"
        style={{ "--an-cols": "44px minmax(0, 1.4fr) minmax(0, 1.3fr) minmax(0, 0.9fr)" }}
      >
        <div className="an-table-head">
          <span>Pos</span>
          <span>Player</span>
          <span>Team</span>
          <span>Line</span>
        </div>
        <div className="an-table-body">
          {slots.map((slot, i) => (
            <div
              key={`${slot.slot}-${i}`}
              className={`an-table-row${revealed ? " is-in" : ""}`}
              style={phase === PHASE.REVEAL ? { animationDelay: `${i * 220}ms` } : undefined}
            >
              <span className="an-cell-muted">{slot.slot}</span>
              <span className={revealed ? "an-cell-name" : "an-cell-muted"}>
                {revealed ? slot.label : "Sealed"}
              </span>
              <span className="an-cell-team">
                {revealed ? (
                  <>
                    <TeamLogo src={slot.teamLogoSrc} size="sm" />
                    <span>{slot.teamName || ""}</span>
                  </>
                ) : null}
              </span>
              <span className="an-num">
                {revealed && slot.stat ? `${slot.stat.value} ${slot.stat.label}` : ""}
              </span>
            </div>
          ))}
        </div>
      </div>
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
    <div className="an-order an-votes">
      <h4>{ballot ? (hasFirst ? "Voting results" : "How the race finished") : "Final order"}</h4>
      {ballot && hasFirst && format ? (
        <p className="an-votes__fmt">
          {[format.voters ? `${format.voters} ballots` : "", format.body || "", curve ? `${curve} points` : ""]
            .filter(Boolean)
            .join(" · ")}
        </p>
      ) : null}
      <div className="an-votes__row an-votes__head">
        <span>#</span>
        <span>Player</span>
        <span>{ballot && hasFirst ? "1st" : ""}</span>
        <span>{ballot ? (hasFirst ? "Pts" : "Score") : "Total"}</span>
      </div>
      {rows.map((r, i) => {
        const who = team ? null : resolvePlayer({ ...r, player_id: r.player_id ?? r.entity_id }, playerIndex, r.name);
        const sub = [r.team_abbr || r.team_name, r.position, r.summary].filter(Boolean).join(" · ");
        return (
          <div key={`${r.entity_id || r.name}-${i}`} className={`an-votes__row${r.is_winner ? " is-winner" : ""}`}>
            <span className="an-num">{toNum(r.rank) ?? i + 1}</span>
            <span className="an-votes__who">
              {who ? <PlayerHeadshot player={who} size="xs" mood="neutral" showFlag={false} /> : null}
              <span className="an-votes__name">
                <strong>{r.name}</strong>
                {sub ? <small>{sub}</small> : null}
              </span>
            </span>
            <em className="an-num">{ballot && hasFirst ? toNum(r.first_place_votes) ?? 0 : ""}</em>
            <em className="an-num">
              {ballot ? formatPoints(toNum(r.points) ?? 0) : formatStat(r.value) || ""}
            </em>
          </div>
        );
      })}
    </div>
  );
}

function CitationBody({ slide, ev, finalists, playerIndex, rule = "" }) {
  const team = isTeamSlide(slide);
  const winnerCard = finalists.find((card) => isWinnerCard(card, slide)) || null;
  const player = team
    ? null
    : resolvePlayer(slide?.winnerPlayer || winnerCard?.player, playerIndex, slide?.winnerLabel);
  const logo = winnerLogo(slide) || cardLogo(winnerCard);
  const teamName = slide?.winnerTeamName || cardTeam(winnerCard);
  const position = cardPosition(winnerCard) || player?.position || "";
  const why = cleanLines(safeArray(ev?.why).length ? ev.why : slide?.whyTheyWon);
  const stats = winnerStatRows(slide, ev);
  const components = safeArray(ev?.winner?.components).filter(
    (c) => c?.label && toNum(c.pct) !== null
  );
  const ballot = ev?.winner?.ballot || null;
  const race = ev?.race || null;
  const order = [...finalists].sort((a, b) => (toNum(a?.rank) ?? 99) - (toNum(b?.rank) ?? 99));
  const hasBallot = Boolean(ballot) && toNum(ballot.first_place_votes) !== null;
  const hasRace = Boolean(race) && toNum(race.leader_value) !== null;
  const voteTable = safeArray(ev?.voting_table).filter((r) => r && r.name);
  const hasResult = hasBallot || hasRace || order.length > 1 || voteTable.length > 0;
  const hasCase = why.length || stats.length || components.length;

  return (
    <div className="an-card">
      <div className={`an-card-grid${hasResult ? "" : " is-two"}`}>
        <section className="an-z-identity">
          {team ? <TeamLogo src={logo} size="hero" /> : <Headshot player={player} size="lg" />}
          <p className="an-pos">{slide?.awardLabel}</p>
          <h2 className="an-name">{slide?.winnerLabel}</h2>
          {!team && teamName ? (
            <p className="an-vitals">
              <TeamLogo src={logo} size="sm" />
              <span>{teamName}</span>
            </p>
          ) : null}
          {!team && position ? <p className="an-club">{position}</p> : null}
        </section>

        <section className="an-z-eval">
          {why.length ? (
            <div className="an-why">
              <h4>Why they won</h4>
              {why.slice(0, 5).map((line) => (
                <p key={line}>{line}</p>
              ))}
            </div>
          ) : null}
          {stats.length ? (
            <div>
              <h4>Season line</h4>
              <StatTable rows={stats} />
            </div>
          ) : null}
          {components.length ? (
            <div>
              <h4>Against the field</h4>
              <ComponentList items={components} />
            </div>
          ) : null}
          {!hasCase ? (
            <p className="an-muted">Detailed results weren't recorded for this season.</p>
          ) : null}
          {rule ? (
            <p className="an-criteria">
              <b>How you win it</b>
              {rule}
            </p>
          ) : null}
        </section>

        {hasResult ? (
          <section className="an-z-result">
            <p className="an-result-title">Result</p>

            {hasBallot ? (
              <>
                <div className="an-result-peak">
                  <span>First-place votes</span>
                  <strong className="an-num">
                    {toNum(ballot.first_place_votes)}
                    {toNum(ballot.voter_count) !== null ? <small>/ {toNum(ballot.voter_count)}</small> : null}
                  </strong>
                </div>
                <div className="an-result-meta">
                  {toNum(ballot.points) !== null ? (
                    <div>
                      <span>Ballot points</span>
                      <strong className="an-num">{formatPoints(toNum(ballot.points))}</strong>
                    </div>
                  ) : null}
                  {toNum(ballot.margin) !== null ? (
                    <div>
                      <span>Margin</span>
                      <strong className="an-num">+{formatPoints(toNum(ballot.margin))}</strong>
                    </div>
                  ) : null}
                  {ballot.runner_up_name ? (
                    <div>
                      <span>Runner-up</span>
                      <strong>{ballot.runner_up_name}</strong>
                    </div>
                  ) : null}
                </div>
                {ev?.closest_of_night ? <p className="an-result-note">Closest vote of the night.</p> : null}
              </>
            ) : hasRace ? (
              <>
                <div className="an-result-peak">
                  <span>{race.metric_label || "Leader"}</span>
                  <strong className="an-num">{formatStat(race.leader_value)}</strong>
                </div>
                <div className="an-result-meta">
                  {toNum(race.margin) !== null ? (
                    <div>
                      <span>Lead</span>
                      <strong className="an-num">+{formatStat(race.margin)}</strong>
                    </div>
                  ) : null}
                  {toNum(race.runner_up_value) !== null ? (
                    <div>
                      <span>Runner-up</span>
                      <strong className="an-num">{formatStat(race.runner_up_value)}</strong>
                    </div>
                  ) : null}
                </div>
              </>
            ) : null}

            {voteTable.length ? (
              <VoteTable table={voteTable} format={ev?.ballot_format} playerIndex={playerIndex} team={team} />
            ) : order.length > 1 ? (
              <div className="an-order">
                <h4>Final order</h4>
                {order.map((card, i) => {
                  const b = ballotFor(card, ev);
                  const tail = b
                    ? `${formatPoints(toNum(b.points))} pts`
                    : formatStat(card?.displayValue ?? card?.display_value) || "";
                  return (
                    <div
                      key={`${card?.label}-${i}`}
                      className={`an-order-row${isWinnerCard(card, slide) ? " is-winner" : ""}`}
                    >
                      <span>{toNum(card?.rank) ?? i + 1}</span>
                      <strong>{card?.label}</strong>
                      <em className="an-num">{tail}</em>
                    </div>
                  );
                })}
              </div>
            ) : null}
          </section>
        ) : null}
      </div>
    </div>
  );
}

function SummaryBody({ slides, playerIndex }) {
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
  const stats = pick && !roster ? winnerStatRows(pick, ev).slice(0, 5) : [];
  const why = pick ? cleanLines(safeArray(ev?.why).length ? ev.why : pick?.whyTheyWon).slice(0, 2) : [];
  const runnersUp = [...finalists]
    .filter((card) => !isWinnerCard(card, pick))
    .sort((a, b) => (toNum(a?.rank) ?? 99) - (toNum(b?.rank) ?? 99));
  const slots = roster ? rosterSlotsFor(pick, ev) : [];
  const voteRows = safeArray(ev?.voting_table).filter((r) => r && r.name).slice(0, 5);
  const groups = [
    ["Major awards", honours.filter(isMajorSlide)],
    ["More honours", honours.filter((x) => !isMajorSlide(x) && !isRosterSlide(x))],
    ["League selections", honours.filter(isRosterSlide)],
  ].filter(([, list]) => list.length);

  return (
    <div className="an-card an-recap">
      {pick ? (
        <section className={`an-spot${roster ? " is-roster" : ""}`} key={pick.id}>
          {roster ? null : (
            <div className="an-spot-face">
              {team ? <TeamLogo src={winnerLogo(pick)} size="hero" /> : <Headshot player={player} size="lg" />}
            </div>
          )}
          <div className="an-spot-main">
            <small className="an-spot-award">{roster ? "League selection" : pick.awardLabel}</small>
            {pick.stageLine ? <span className="an-spot-line">{pick.stageLine}</span> : null}
            <strong className="an-spot-name">{roster ? pick.awardLabel : pick.winnerLabel}</strong>
            {!team && !roster && pick.winnerTeamName ? (
              <span className="an-spot-team">
                <TeamLogo src={winnerLogo(pick)} size="sm" />
                {pick.winnerTeamName}
              </span>
            ) : null}
            {stats.length ? (
              <div className="an-spot-stats">
                {stats.map((row) => (
                  <div key={row.key}>
                    <b>{row.value}</b>
                    <i>{row.label}</i>
                  </div>
                ))}
              </div>
            ) : null}
            {why.length ? <p className="an-spot-why">{why.join(" ")}</p> : null}
            {runnersUp.length ? (
              <p className="an-spot-runners">
                <i>Also in the running</i> {runnersUp.map((c) => `${c.label}${cardTeam(c) ? ` (${cardTeam(c)})` : ""}`).join(" · ")}
              </p>
            ) : null}
            {slots.length ? (
              <div className="an-spot-lineup">
                {slots.map((slot, i) => (
                  <span key={`${slot?.slot || i}-${slot?.label || i}`}>
                    <i>{slot?.slot || ""}</i>
                    <b>{slot?.label || ""}</b>
                    {slot?.teamName ? <em>{slot.teamName}</em> : null}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
          {!roster && voteRows.length > 1 ? (
            <div className="an-spot-votes">
              <VoteTable table={voteRows} format={ev?.ballot_format} playerIndex={playerIndex} team={team} />
            </div>
          ) : null}
        </section>
      ) : null}

      {groups.map(([label, list]) => (
        <section key={label} className="an-honours">
          <h4 className="an-gate-head">{label}</h4>
          <div className={`an-honour-grid${label === "Major awards" ? " is-major" : ""}`}>
            {list.map((slide) => (
              <button
                key={slide.id}
                type="button"
                className={`an-honour${pick && slide.id === pick.id ? " is-on" : ""}`}
                aria-pressed={Boolean(pick && slide.id === pick.id)}
                onClick={() => setPickId(slide.id)}
              >
                {isRosterSlide(slide) ? null : <TeamLogo src={winnerLogo(slide)} size="sm" />}
                <span>
                  <i>{slide.awardLabel}</i>
                  <b>{isRosterSlide(slide) ? "See the lineup" : slide.winnerLabel}</b>
                </span>
              </button>
            ))}
          </div>
        </section>
      ))}

      {cup ? (
        <section className="an-champ">
          <span className="an-champ-rays" aria-hidden="true" />
          <TeamLogo src={winnerLogo(cup)} size="hero" />
          <div>
            <small>Stanley Cup champions</small>
            <strong>{cup.winnerLabel}</strong>
          </div>
        </section>
      ) : null}
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
    headTitle = "Awards Night";
    headMeta.push({ text: `${slides.length} awards on tonight's card`, tone: "strong" });
    headMeta.push({ text: seasonLabel(franchiseState) });
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
      <div className="an-card">
        <p className="an-muted">
          No award results were found. Complete the playoffs to reveal the season's hardware.
        </p>
      </div>
    );
  } else if (phase === PHASE.GATE) {
    body = <GateBody slides={slides} playerIndex={playerIndex} userTeamName={userTeamName} onPick={jumpToAward} />;
  } else if (phase === PHASE.SUMMARY) {
    body = <SummaryBody slides={slides} playerIndex={playerIndex} />;
  } else if (phase === PHASE.TITLE || (phase === PHASE.SEAL && !hasFinalistPhase && !roster)) {
    body = (
      <TitleBody
        key={`title-${activeSlide.id}`}
        slide={activeSlide}
        ev={ev}
        finalistsCount={roster ? 0 : finalists.length}
        rule={ruleByKey[activeSlide?.awardKey] || ""}
      />
    );
  } else if (roster) {
    body = <RosterBody key={`roster-${activeSlide.id}`} slide={activeSlide} ev={ev} phase={phase} />;
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

  return (
    <section className="an-root an-broadcast an-dd" aria-label="Awards Night">
      <header className="an-topbar">
        <div className="an-topbar-main">
          <h1 className="an-page-title">Awards Night</h1>
          <span className="an-season">{seasonLabel(franchiseState)}</span>
        </div>
        <nav className="an-milestones" aria-label="Offseason timeline">
          {safeArray(SEASON_MILESTONES).map((milestone) => (
            <span key={milestone.id} className={milestone.id === "awards" ? "is-current" : ""}>
              {milestone.label}
            </span>
          ))}
        </nav>
        <div className="an-topbar-actions">
          {typeof onBack === "function" ? (
            <button type="button" className="an-ghost-btn an-back-btn" onClick={onBack}>
              Leave to Hub
            </button>
          ) : null}
{null}
          <button
            type="button"
            className={`an-ghost-btn${paused ? " is-on" : ""}`}
            onClick={() => setPaused((v) => !v)}
            disabled={!inCeremony}
          >
            {paused ? "Play" : "Pause"}
          </button>
          {phase !== PHASE.SUMMARY && slides.length ? (
            <button type="button" className="an-ghost-btn" onClick={goToSummary}>
              Skip to summary
            </button>
          ) : null}
        </div>
      </header>

      <div className="an-floor">
        <div className="an-floor-grid">
          <CeremonyLog
            groups={indexGroups}
            slidesCount={slides.length}
            activeIndex={inCeremony ? activeIndex : -1}
            revealedIds={displayRevealed}
            onSelect={jumpToAward}
          />

          <section className="an-stage" aria-live="polite">
            <header className="an-stagehead">
              <span className="an-stagehead-num">{headNum}</span>
              <div className="an-stagehead-identity">
                <div className={`an-stagehead-state${headLive ? " is-live" : ""}`}>{headState}</div>
                <h2 className="an-stagehead-title">{headTitle}</h2>
                {headMeta.length ? (
                  <div className="an-stagehead-meta">
                    {headMeta.map((item) => (
                      <span
                        key={item.text}
                        className={item.tone === "gold" ? "is-gold" : item.tone === "strong" ? "is-strong" : ""}
                      >
                        {item.text}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
              {upNext.length ? (
                <div className="an-stagehead-next">
                  <span className="an-order-label">Up next</span>
                  {upNext.map((slide, i) => (
                    <span key={slide.id} className="an-next-item">
                      <b>{pad2(upNextStart + i + 1)}</b>
                      {slide.awardLabel}
                    </span>
                  ))}
                </div>
              ) : null}
            </header>

            <div className="an-stage-body">{body}</div>

            <div className="an-dock">
              {phase === PHASE.SUMMARY || !slides.length ? (
                <>
                  <div className="an-dock-state">
                    {slides.length ? `All ${slides.length} awards presented` : "Nothing to present"}
                  </div>
                  <div className="an-dock-actions">
                    <button
                      type="button"
                      className="an-cta-btn an-dock-primary"
                      onClick={handleContinue}
                      disabled={continuing}
                    >
                      {continuing ? "Continuing…" : slides.length ? "Continue to Retirements" : "Continue"}
                    </button>
                  </div>
                  {continueError ? <p className="an-dock-error">{continueError}</p> : null}
                </>
              ) : phase === PHASE.GATE ? (
                <>
                  <div className="an-dock-state">
                    <b>Ceremony ready</b> · {slides.length} awards
                  </div>
                  <div className="an-dock-actions">
                    <button type="button" className="an-cta-btn an-dock-primary" onClick={beginCeremony}>
                      Begin ceremony
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <div className="an-dock-state">
                    <b>{PHASE_LABEL[phase]}</b> · Award {activeIndex + 1} of {slides.length}
                    {paused ? " · Paused" : ""}
                  </div>
                  <div className="an-dock-actions">
                    <button
                      type="button"
                      className="an-ghost-btn"
                      onClick={revealNow}
                      disabled={isRevealedPhase(phase)}
                    >
                      Reveal now
                    </button>
                    <button type="button" className="an-ghost-btn" onClick={skipAward}>
                      Skip award
                    </button>
                    <button type="button" className="an-cta-btn an-dock-primary" onClick={advancePhase}>
                      {phase === PHASE.CITATION ? (isLast ? "Finish ceremony" : "Next award") : "Continue"}
                    </button>
                  </div>
                </>
              )}
            </div>
          </section>

          <FanPulse
            mode={pulseMode}
            items={pulseItems}
            emptyText={pulseEmpty}
            feedKey={`${pulseStage}-${activeSlide?.id || "gate"}`}
          />
        </div>
      </div>
    </section>
  );
}