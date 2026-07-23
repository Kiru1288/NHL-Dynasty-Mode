import React, {
  Suspense,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import "./WorldJuniorsMenu.css";

import worldJuniorsTheme from "./Hockey Canada TSN IIHF World & Junior Championships theme song - Mordakan.mp3";
import {
  buildWjcBroadcastLines,
  buildWjcDraftStockRows,
  buildWjcShowcaseCards,
  buildWjcStatLeaders,
} from "./wjcBroadcastBuilder";
import {
  DeskControls,
  DraftStockSidebar,
  GameResultModal,
  GamesBrowser,
  NationFlagsBar,
  ProspectDetailModal,
  ShowcaseOverlay,
  StatLeadersSidebar,
} from "./WorldJuniorsBroadcastPanels";
import { WJC_HOSTS } from "./wjcBroadcastScripts";
import { wjcFlagUrl } from "../../utils/countryFlags";

import bg1 from "./gettyimages-136461654-612x612.jpg";
import bg2 from "./gettyimages-136295976-612x612.jpg";
import bg3 from "./gettyimages-95597561-612x612.jpg";
import bg4 from "./gettyimages-84179721-612x612.jpg";
import bg5 from "./gettyimages-2254829655-612x612.jpg";

const WJC_HERO_BACKGROUNDS = [bg1, bg2, bg3, bg4, bg5];
const HERO_BG_ROTATE_MS = 10000;
const HERO_BG_FADE_MS = 2800;

function BroadcastHeroBackdrop() {
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    const timerId = window.setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % WJC_HERO_BACKGROUNDS.length);
    }, HERO_BG_ROTATE_MS);
    return () => window.clearInterval(timerId);
  }, []);

  return (
    <div className="wjc-hero-backdrop" aria-hidden="true">
      {WJC_HERO_BACKGROUNDS.map((src, index) => (
        <div
          key={src}
          className="wjc-hero-backdrop__slide"
          style={{
            backgroundImage: `url(${src})`,
            opacity: index === activeIndex ? 0.32 : 0,
            transition: `opacity ${HERO_BG_FADE_MS}ms ease-in-out`,
          }}
        />
      ))}
      <div className="wjc-hero-backdrop__scrim" />
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Payload resolver — primary backend data path                               */
/* -------------------------------------------------------------------------- */

function asArray(v) {
  return Array.isArray(v) ? v : [];
}

function isWjcPayload(obj) {
  if (!obj || typeof obj !== "object") return false;
  return obj.kind === "wjc_tournament" || obj.wjc_live === true || Boolean(obj.wjc_phase);
}

function findActiveWjcPopup(franchiseState) {
  const pops = [
    ...asArray(franchiseState?.pending_ui_popups),
    ...asArray(franchiseState?.pendingUiPopups),
  ];
  return pops.find((pop) => pop && isWjcPayload(pop)) || null;
}

function findArchivedWjc(franchiseState) {
  const arch = asArray(franchiseState?.showcase_archive);
  for (let i = arch.length - 1; i >= 0; i -= 1) {
    if (isWjcPayload(arch[i])) return arch[i];
  }
  return null;
}

function normalizePlayoffs(raw) {
  const po = raw && typeof raw === "object" ? raw : {};
  return {
    quarterfinals: asArray(po.quarterfinals),
    semifinals: asArray(po.semifinals),
    bronze: po.bronze && typeof po.bronze === "object" ? po.bronze : null,
    gold: po.gold && typeof po.gold === "object" ? po.gold : null,
  };
}

function normalizeWjcFields(raw) {
  const src = raw && typeof raw === "object" ? raw : {};
  const wjcDay = src.wjc_day ?? src.day ?? null;
  const wjcDaysTotal = src.wjc_days_total ?? src.days_total ?? 11;
  const wjcPhase = String(src.wjc_phase || src.phase || "").toLowerCase();
  const medalsFinal = Boolean(
    src.medals_final || wjcPhase === "complete" || (wjcDay != null && wjcDay >= wjcDaysTotal)
  );

  return {
    wjc_phase: medalsFinal ? "complete" : wjcPhase || (wjcDay != null ? "live" : ""),
    calendar_iso: String(src.calendar_iso || src.iso || ""),
    wjc_day: wjcDay != null ? Number(wjcDay) : null,
    wjc_days_total: Number(wjcDaysTotal) || 11,
    title: String(src.title || ""),
    season_label: String(src.season_label || ""),
    countries: asArray(src.countries),
    round_robin_games: asArray(src.round_robin_games),
    round_robin_total: Number(src.round_robin_total) || asArray(src.round_robin_games).length || 0,
    standings: asArray(src.standings),
    playoffs: normalizePlayoffs(src.playoffs),
    medal_labels:
      src.medal_labels && typeof src.medal_labels === "object" ? { ...src.medal_labels } : {},
    medals_final: medalsFinal,
    user_prospects: asArray(src.user_prospects),
    tournament_prospects: asArray(src.tournament_prospects),
    player_stats: asArray(src.player_stats),
    all_games: asArray(src.all_games),
    games_today: asArray(src.games_today),
    all_games_total: Number(src.all_games_total) || asArray(src.all_games).length || 0,
    rr_days_total: Number(src.rr_days_total) || 9,
  };
}

function buildCalendarFallback(franchiseState) {
  const hud = franchiseState?.draft_class_hud?.events?.wjc || {};
  const anchors = asArray(franchiseState?.season_anchor_events);
  const wjcAnchor =
    anchors.find((a) => String(a?.key || "").includes("wjc_start")) ||
    anchors.find((a) => String(a?.type || a?.id || "").includes("wjc")) ||
    null;

  const countdown = hud.display || hud.date || "";
  const daysUntil = hud.days_until ?? hud.daysUntil ?? null;
  const startDate = hud.date || wjcAnchor?.date || "";

  return {
    ...normalizeWjcFields({}),
    countdown_display: String(countdown || ""),
    countdown_days: daysUntil,
    start_date: String(startDate || ""),
    anchor_title: String(wjcAnchor?.title || wjcAnchor?.label || "World Juniors"),
  };
}

/** @returns {{ source: string, hasData: boolean, isPreTournament: boolean, raw: object }} */
export function resolveWorldJuniorsPayload(franchiseState, eventData) {
  const empty = {
    source: "none",
    hasData: false,
    isPreTournament: true,
    raw: normalizeWjcFields({}),
    ...normalizeWjcFields({}),
    countdown_display: "",
    countdown_days: null,
    start_date: "",
    anchor_title: "World Juniors",
  };

  if (!franchiseState && !eventData) return empty;

  let source = "calendar";
  let raw = null;

  const activePopup = findActiveWjcPopup(franchiseState);
  if (activePopup) {
    source = "live";
    raw = activePopup;
  } else if (eventData && isWjcPayload(eventData)) {
    source = "eventData";
    raw = eventData;
  } else {
    const archived = findArchivedWjc(franchiseState);
    if (archived) {
      source = "archive";
      raw = archived;
    }
  }

  const calendarMeta = buildCalendarFallback(franchiseState);

  if (raw) {
    const normalized = normalizeWjcFields(raw);
    return {
      source,
      hasData: true,
      isPreTournament: false,
      raw,
      ...normalized,
      countdown_display: calendarMeta.countdown_display,
      countdown_days: calendarMeta.countdown_days,
      start_date: calendarMeta.start_date,
      anchor_title: calendarMeta.anchor_title,
    };
  }

  const hasCountdown =
    calendarMeta.countdown_display ||
    calendarMeta.countdown_days != null ||
    calendarMeta.start_date;

  return {
    source: hasCountdown ? "calendar" : "none",
    hasData: false,
    isPreTournament: true,
    raw: null,
    ...calendarMeta,
  };
}

/* -------------------------------------------------------------------------- */
/* Display helpers                                                            */
/* -------------------------------------------------------------------------- */

function getYear(payload, franchiseState) {
  const label = payload?.season_label || "";
  const match = label.match(/(\d{4})/);
  if (match) return match[1];
  return (
    franchiseState?.season_year ||
    franchiseState?.seasonYear ||
    new Date().getFullYear()
  );
}

function getUserTeamName(franchiseState) {
  return (
    franchiseState?.team?.name ||
    franchiseState?.team?.full_name ||
    franchiseState?.team?.fullName ||
    franchiseState?.team?.abbreviation ||
    franchiseState?.team?.abbr ||
    "FRANCHISE"
  );
}

function gameCode(g, side) {
  return String(g?.[`${side}`] || g?.[`${side}_label`] || "?").slice(0, 3).toUpperCase();
}

function formatScoreLine(g) {
  const home = gameCode(g, "home");
  const away = gameCode(g, "away");
  const hg = g?.home_goals;
  const ag = g?.away_goals;
  if (hg != null && ag != null) {
    return `${home} ${hg} — ${away} ${ag}`;
  }
  return `${home} vs ${away}`;
}

function formatTickerGame(g, prefix = "FINAL") {
  const home = gameCode(g, "home");
  const away = gameCode(g, "away");
  const hg = g?.home_goals;
  const ag = g?.away_goals;
  if (hg != null && ag != null) {
    return `${prefix}: ${home} ${hg}, ${away} ${ag}`;
  }
  return `${home} vs ${away}`;
}

function buildTickerItems(payload) {
  if (!payload?.hasData) return [];

  const items = [];
  const day = payload.wjc_day;
  const po = payload.playoffs || {};

  asArray(payload.round_robin_games).forEach((g) => {
    items.push(formatTickerGame(g, "FINAL"));
  });

  const allGames = asArray(payload.all_games);
  if (allGames.length > asArray(payload.round_robin_games).length) {
    items.length = 0;
    allGames.forEach((g) => {
      const tag = g.round ? String(g.round).toUpperCase().slice(0, 8) : "FINAL";
      items.push(formatTickerGame(g, tag));
    });
  }

  if (day >= 8) {
    asArray(po.quarterfinals).forEach((g) => items.push(formatTickerGame(g, "QF FINAL")));
  }
  if (day >= 9) {
    asArray(po.semifinals).forEach((g) => items.push(formatTickerGame(g, "SF FINAL")));
  }
  if (day >= 10 && po.bronze) {
    items.push(formatTickerGame(po.bronze, "BRONZE"));
  }
  if (day >= 11 && po.gold) {
    items.push(formatTickerGame(po.gold, "GOLD"));
  }

  if (payload.medals_final && payload.medal_labels) {
    const ml = payload.medal_labels;
    items.push(
      `MEDALS: GOLD ${ml.gold || "—"} · SILVER ${ml.silver || "—"} · BRONZE ${ml.bronze || "—"}`
    );
  }

  return items;
}

function getTournamentPhaseLabel(day, complete) {
  if (complete || day >= 11) return "MEDAL ROUND";
  if (day === 10) return "BRONZE GAME";
  if (day === 9) return "SEMIFINALS";
  if (day === 8) return "QUARTERFINALS";
  if (day >= 1 && day <= 7) return "GROUP STAGE";
  return "PRE-TOURNAMENT";
}

function getFeaturedStory(payload) {
  if (!payload?.hasData) {
    if (payload?.countdown_display) {
      return {
        tag: "COUNTDOWN",
        headline: payload.anchor_title || "World Juniors",
        sub: payload.countdown_display,
      };
    }
    return { tag: "STANDBY", headline: "WJC BROADCAST DESK", sub: "AWAITING TOURNAMENT DATA" };
  }

  const day = payload.wjc_day;
  const complete = payload.medals_final;
  const top = payload.standings?.[0];

  if (complete && payload.medal_labels?.gold) {
    return {
      tag: "DRAFT BOARD SHOCK",
      headline: `${payload.medal_labels.gold} WINS GOLD`,
      sub: "PERMANENT DRAFT STOCK SHIFT",
    };
  }

  if (top) {
    return {
      tag: day >= 8 ? "NATIONAL SPOTLIGHT" : "GROUP STAGE",
      headline: `${top.code} LEADS STANDINGS`,
      sub: `${top.pts} PTS · ${top.w}-${top.l} RECORD`,
    };
  }

  return {
    tag: "LIVE",
    headline: payload.title || "WORLD JUNIORS LIVE",
    sub: day ? `DAY ${day} OF ${payload.wjc_days_total}` : "TOURNAMENT IN PROGRESS",
  };
}

function getTodayGames(payload) {
  if (!payload?.hasData) return [];
  const games = asArray(payload.round_robin_games);
  const day = payload.wjc_day || 1;
  const total = payload.round_robin_total || games.length || 1;
  const nDays = payload.wjc_days_total || 11;

  const through = Math.min(total, Math.max(1, Math.floor((day * total + nDays - 1) / nDays)));
  const prevThrough =
    day > 1
      ? Math.min(total, Math.max(0, Math.floor(((day - 1) * total + nDays - 1) / nDays)))
      : 0;

  return games.slice(prevThrough, through);
}

function collectLoanDecisions(franchiseState) {
  const decisions = [
    ...asArray(franchiseState?.pending_decisions),
    ...asArray(franchiseState?.pendingDecisions),
  ];
  return decisions.filter((d) => d && d.kind === "wjc_u20_loan");
}

/* -------------------------------------------------------------------------- */
/* Cinematic broadcast — flags, queue, speech, camera                           */
/* -------------------------------------------------------------------------- */

function countryLabelFor(code, payload) {
  const c = asArray(payload?.countries).find((x) => String(x.code) === String(code));
  return c?.label || code;
}

function renderCountryFlag(code, payload, { size = 64, className = "" } = {}) {
  const label = countryLabelFor(code, payload);
  const flagUrl = wjcFlagUrl(code, size);
  return (
    <div className={`wjc-country-flag ${className}`.trim()}>
      {flagUrl ? (
        <img
          src={flagUrl}
          alt={`${label} flag`}
          loading="lazy"
          referrerPolicy="no-referrer"
          onError={(e) => {
            e.currentTarget.style.display = "none";
            const fb = e.currentTarget.nextElementSibling;
            if (fb) fb.style.display = "flex";
          }}
        />
      ) : null}
      <span className="wjc-country-flag__fallback" style={flagUrl ? { display: "none" } : undefined}>
        {String(code || "?").slice(0, 3).toUpperCase()}
      </span>
    </div>
  );
}

const WJC_INTRO_TEMPLATES = [
  "Welcome to the World Juniors. The future of hockey is on display.",
  "Nine nations are here. Every scout has a seat.",
  "This tournament has launched careers and destroyed draft stock.",
  "The hockey world has arrived. Welcome to the World Juniors.",
  "Eleven days can change an entire draft board.",
  "For these prospects, every shift matters.",
  "Welcome to the World Juniors.",
  "Eleven days. Nine nations. Every scout is watching.",
  "For some prospects, this tournament changes everything.",
  "The World Juniors desk is live — medal-round pressure or group stage, the board moves tonight.",
  "{nation_count} nations have arrived.",
  "We are on day {wjc_day} of {wjc_days_total}.",
  "{leader_label} currently leads the tournament.",
  "Group stage hockey with global stakes — welcome to the World Juniors.",
  "Scouts are packed in. Draft boards are open. Welcome to the World Juniors.",
  "Every nation brought its best U20 talent. Welcome to the World Juniors.",
  "From opening faceoff to gold medal day — this is the World Juniors.",
  "The tournament is here. The draft board is watching.",
];

const HOST_TTS = {
  host_1: { rate: 1.08, pitch: 1.12 },
  host_2: { rate: 0.96, pitch: 1.0 },
  host_3: { rate: 0.9, pitch: 0.92 },
};

function getHostCameraMode(hostId) {
  if (hostId === "host_1") return "host-left";
  if (hostId === "host_3") return "host-right";
  if (hostId === "host_2") return "host-center";
  return "wide";
}

function inferTopicFromLine(line) {
  const id = String(line?.id || "").toLowerCase();
  const tag = String(line?.meta?.tag || "").toLowerCase();
  if (id.includes("stock") || tag.includes("stock") || id.includes("riser") || id.includes("faller")) {
    return "DRAFT STOCK";
  }
  if (id.includes("standing") || tag.includes("standing") || id.includes("leader") || id.includes("surprise")) {
    return "STANDINGS";
  }
  if (
    id.includes("score") ||
    id.includes("gold") ||
    id.includes("qf") ||
    id.includes("sf") ||
    id.includes("bronze") ||
    id.includes("upset") ||
    id.includes("hot_open")
  ) {
    return "SCORES";
  }
  if (id.includes("user_prospect")) return "USER PROSPECT";
  if (id.includes("medal") || id.includes("gold_medal")) return "MEDALS";
  if (id.includes("open") || id.includes("pretournament")) return "OPENING";
  return "GENERAL";
}

function getTransitionType(topic, graphicTakeover) {
  if (graphicTakeover) return "graphic-takeover";
  if (topic === "SCORES") return "hard-cut";
  if (topic === "STANDINGS") return "broadcast-push";
  if (topic === "OPENING") return "broadcast-push";
  if (topic === "MEDALS") return "graphic-takeover";
  return "broadcast-push";
}

function interpolateIntro(text, payload) {
  const leader = payload?.standings?.[0];
  return String(text || "")
    .replace(/\{nation_count\}/g, String(asArray(payload?.countries).length || "—"))
    .replace(/\{wjc_day\}/g, String(payload?.wjc_day ?? "—"))
    .replace(/\{wjc_days_total\}/g, String(payload?.wjc_days_total ?? 11))
    .replace(/\{leader_label\}/g, leader?.label || leader?.code || "—");
}

function buildOpeningIntroLine(payload) {
  const tpl = WJC_INTRO_TEMPLATES[Math.floor(Math.random() * WJC_INTRO_TEMPLATES.length)];
  return interpolateIntro(tpl, payload);
}

function getDailyIntroLabel(payload) {
  const day = payload?.wjc_day;
  const complete = payload?.medals_final || payload?.wjc_phase === "complete";
  if (complete) return { title: "WORLD JUNIORS", sub: "TOURNAMENT COMPLETE" };
  if (day >= 11) return { title: "WORLD JUNIORS", sub: "GOLD MEDAL DAY" };
  if (day === 10) return { title: "WORLD JUNIORS", sub: "MEDAL PRESSURE" };
  if (day === 9) return { title: "WORLD JUNIORS", sub: "SEMIFINALS" };
  if (day === 8) return { title: "WORLD JUNIORS", sub: "QUARTERFINALS" };
  if (day >= 1) return { title: "WORLD JUNIORS", sub: `DAY ${day}`, phase: "GROUP STAGE" };
  return { title: "WORLD JUNIORS", sub: payload?.season_label || "U20 CHAMPIONSHIP" };
}

function shouldUseFullIntro(payload) {
  const day = payload?.wjc_day;
  return !day || day <= 1;
}

const WJC_SESSION_KEY = "wjc_broadcast_session";

function wjcSessionKey(payload) {
  return `${payload?.season_label || "wjc"}|day:${payload?.wjc_day ?? 0}`;
}

function loadWjcSession() {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(WJC_SESSION_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveWjcSession(payload, phase) {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(
      WJC_SESSION_KEY,
      JSON.stringify({
        key: wjcSessionKey(payload),
        wjc_day: payload?.wjc_day ?? 0,
        phase,
      })
    );
  } catch {
    /* ignore quota errors */
  }
}

function findPlayerInText(text, payload) {
  const lower = String(text || "").toLowerCase();
  return (
    asArray(payload?.player_stats).find((p) => p.name && lower.includes(String(p.name).toLowerCase())) ||
    asArray(payload?.tournament_prospects).find((p) => p.name && lower.includes(String(p.name).toLowerCase())) ||
    asArray(payload?.user_prospects).find((p) => p.name && lower.includes(String(p.name).toLowerCase())) ||
    null
  );
}

function findTeamInText(text, payload) {
  const codes = [
    ...asArray(payload?.countries).map((c) => c.code),
    ...asArray(payload?.standings).map((s) => s.code),
  ].filter(Boolean);
  const upper = String(text || "").toUpperCase();
  const matched = codes.find((code) => upper.includes(String(code).toUpperCase()));
  if (!matched) return null;
  return asArray(payload?.standings).find((s) => String(s.code) === String(matched)) || {
    code: matched,
    label: countryLabelFor(matched, payload),
  };
}

function resolveCenterPopup(line, payload, draftStockRows) {
  const topic = line?.topic || inferTopicFromLine(line);
  const text = line?.text || "";

  const playerFromText = findPlayerInText(text, payload);
  if (playerFromText) {
    const prospect = draftStockRows.find(
      (r) => String(r.player_id) === String(playerFromText.player_id)
    );
    return { type: "player", data: { ...playerFromText, ...prospect } };
  }

  if (topic === "DRAFT STOCK" && line?.graphicData) {
    const stat = asArray(payload?.player_stats).find(
      (p) => String(p.player_id) === String(line.graphicData.player_id)
    );
    return { type: "player", data: { ...line.graphicData, ...stat } };
  }

  if (topic === "USER PROSPECT" && line?.graphicData) {
    const stat = asArray(payload?.player_stats).find(
      (p) => String(p.player_id) === String(line.graphicData.player_id)
    );
    return { type: "player", data: { ...line.graphicData, ...stat } };
  }

  if (topic === "SCORES") {
    const game = asArray(line?.graphicData)[0];
    if (game) return { type: "game", data: game };
    const games = asArray(payload?.all_games).length
      ? asArray(payload?.all_games)
      : asArray(payload?.round_robin_games);
    if (games.length) return { type: "game", data: games[games.length - 1] };
  }

  if (topic === "STANDINGS") {
    const team = findTeamInText(text, payload) || payload?.standings?.[0];
    if (team) return { type: "team", data: team };
  }

  if (topic === "MEDALS" && line?.graphicData) {
    return { type: "medals", data: line.graphicData };
  }

  const teamFromText = findTeamInText(text, payload);
  if (teamFromText) return { type: "team", data: teamFromText };

  return null;
}

function enrichBroadcastLine(line, payload, draftStockRows) {
  const topic = inferTopicFromLine(line);
  const cameraMode = getHostCameraMode(line.speakerId);
  let graphicType = null;
  let graphicData = null;
  let graphicTakeover = false;

  if (topic === "STANDINGS") {
    graphicType = "standings";
    graphicData = asArray(payload?.standings).slice(0, 5);
  } else if (topic === "SCORES") {
    graphicType = "scores";
    const games = asArray(payload?.all_games).length
      ? asArray(payload?.all_games)
      : asArray(payload?.round_robin_games);
    graphicData = games.slice(-3);
  } else if (topic === "DRAFT STOCK") {
    graphicType = "stock";
    const row = draftStockRows.find((r) => Math.abs(Number(r.stock_delta) || 0) >= 20) || draftStockRows[0];
    graphicData = row;
    if (row && Math.abs(Number(row.stock_delta) || 0) >= 20) {
      graphicTakeover = true;
    }
  } else if (topic === "USER PROSPECT") {
    graphicType = "user-prospect";
    graphicData = asArray(payload?.user_prospects).find((p) => p.made_wjc_team) || payload?.user_prospects?.[0];
  } else if (topic === "MEDALS") {
    graphicType = "medals";
    graphicData = payload?.medal_labels;
  }

  return {
    ...line,
    hostId: line.speakerId,
    topic,
    cameraMode,
    graphicType,
    graphicData,
    graphicTakeover,
    centerPopup: resolveCenterPopup(
      { ...line, topic, graphicData, graphicType },
      payload,
      draftStockRows
    ),
    transition: getTransitionType(topic, graphicTakeover),
  };
}

function buildBroadcastQueue(payload, draftStockRows) {
  const queue = [];
  const openingText = buildOpeningIntroLine(payload);
  const openingItem = {
    id: "opening-intro",
    hostId: "host_2",
    text: openingText,
    topic: "OPENING",
    cameraMode: "host-center",
    graphicType: "tournament",
    graphicData: null,
    graphicTakeover: false,
    transition: "broadcast-push",
    centerPopup: payload?.standings?.[0]
      ? { type: "team", data: payload.standings[0] }
      : null,
  };
  queue.push(openingItem);

  const lines = buildWjcBroadcastLines(payload).map((line) =>
    enrichBroadcastLine(line, payload, draftStockRows)
  );

  lines.forEach((line) => {
    if (line.graphicTakeover && line.graphicData) {
      queue.push({
        id: `${line.id}-takeover`,
        hostId: line.hostId,
        text: "",
        topic: "DRAFT STOCK",
        cameraMode: "graphic",
        graphicType: "stock-takeover",
        graphicData: line.graphicData,
        graphicTakeover: true,
        transition: "graphic-takeover",
        silent: true,
        holdMs: 2800,
      });
    }
    queue.push(line);
  });

  if (payload?.medals_final && payload?.medal_labels?.gold) {
    queue.push({
      id: "medal-bronze",
      hostId: "host_3",
      text: `Bronze medal: ${payload.medal_labels.bronze || "—"}.`,
      topic: "MEDALS",
      cameraMode: "graphic",
      graphicType: "medal-bronze",
      graphicData: payload.medal_labels,
      graphicTakeover: true,
      transition: "graphic-takeover",
      silent: false,
    });
    queue.push({
      id: "medal-silver",
      hostId: "host_3",
      text: `Silver medal: ${payload.medal_labels.silver || "—"}.`,
      topic: "MEDALS",
      cameraMode: "graphic",
      graphicType: "medal-silver",
      graphicData: payload.medal_labels,
      graphicTakeover: true,
      transition: "graphic-takeover",
      silent: false,
    });
    queue.push({
      id: "medal-gold",
      hostId: "host_1",
      text: `${payload.medal_labels.gold || "—"} are World Junior champions.`,
      topic: "MEDALS",
      cameraMode: "graphic",
      graphicType: "medal-gold",
      graphicData: payload.medal_labels,
      graphicTakeover: true,
      transition: "graphic-takeover",
      silent: false,
    });
    queue.push({
      id: "closing-line",
      hostId: "host_2",
      text: "The tournament is over. The draft board will never look the same.",
      topic: "OPENING",
      cameraMode: "host-center",
      graphicType: null,
      graphicData: null,
      graphicTakeover: false,
      transition: "wide-reset",
    });
  }

  return queue;
}

function pickVoiceForHost(voices, hostId) {
  const english = voices.filter(
    (v) => v.lang && v.lang.toLowerCase().startsWith("en") && !v.name.toLowerCase().includes("google")
  );
  const pool = english.length ? english : voices;
  const idx = hostId === "host_1" ? 0 : hostId === "host_3" ? 2 : 1;
  return pool[idx] || pool[0] || null;
}

const BroadcastCameraContext = createContext({
  cameraMode: "wide",
  activeHostId: null,
  reducedMotion: false,
});

function useBroadcastCamera() {
  return useContext(BroadcastCameraContext);
}

function speakBroadcastLine({ text, hostId, voiceOn, voices, onStart, onEnd }) {
  if (!text) {
    onEnd?.();
    return () => {};
  }

  if (!voiceOn || typeof window === "undefined" || !window.speechSynthesis) {
    onStart?.();
    const t = window.setTimeout(() => onEnd?.(), Math.max(2200, text.length * 42));
    return () => window.clearTimeout(t);
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  const tune = HOST_TTS[hostId] || HOST_TTS.host_2;
  utterance.rate = tune.rate;
  utterance.pitch = tune.pitch;
  utterance.volume = 1;
  const voice = pickVoiceForHost(voices, hostId);
  if (voice) utterance.voice = voice;

  utterance.onstart = () => onStart?.();
  utterance.onend = () => onEnd?.();
  utterance.onerror = () => onEnd?.();

  window.speechSynthesis.speak(utterance);
  return () => window.speechSynthesis.cancel();
}

function CinematicIntro({
  phase,
  payload,
  countries,
  fullIntro,
  onSkip,
  dailyLabel,
}) {
  const montageIndex = useRef(0);
  const [countryIdx, setCountryIdx] = useState(0);

  useEffect(() => {
    if (phase !== "intro-countries") return undefined;
    montageIndex.current = 0;
    setCountryIdx(0);
    const list = asArray(countries);
    if (!list.length) return undefined;
    const timer = window.setInterval(() => {
      montageIndex.current += 1;
      if (montageIndex.current >= list.length) {
        window.clearInterval(timer);
        return;
      }
      setCountryIdx(montageIndex.current);
    }, 280);
    return () => window.clearInterval(timer);
  }, [phase, countries]);

  if (phase === "idle" || phase === "interactive" || phase === "broadcast" || phase === "opening") {
    return null;
  }

  const currentCountry = asArray(countries)[countryIdx];

  return (
    <div className={`wjc-cinematic-intro wjc-cinematic-intro--${phase}`} aria-hidden={phase === "stage-reveal"}>
      <button type="button" className="wjc-skip-intro" onClick={onSkip} aria-label="Skip intro">
        SKIP INTRO
      </button>

      {phase === "intro-title" && (
        <div className="wjc-intro-beat wjc-intro-beat--title">
          <div className="wjc-intro-red-line" />
          <h1>WORLD JUNIORS</h1>
          <p>{fullIntro ? payload?.season_label || dailyLabel.sub : dailyLabel.sub}</p>
          {dailyLabel.phase ? <span className="wjc-intro-phase-tag">{dailyLabel.phase}</span> : null}
        </div>
      )}

      {phase === "intro-countries" && fullIntro && currentCountry && (
        <div className="wjc-intro-beat wjc-intro-beat--country" key={currentCountry.code}>
          <span className="wjc-intro-country-abbr">{currentCountry.code}</span>
          {renderCountryFlag(currentCountry.code, payload, { size: 128, className: "wjc-intro-country-flag" })}
          <h2>{(currentCountry.label || currentCountry.code).toUpperCase()}</h2>
        </div>
      )}

      {phase === "intro-tournament" && fullIntro && (
        <div className="wjc-intro-beat wjc-intro-beat--collision">
          <div className="wjc-intro-collision-flags">
            {asArray(countries)
              .slice(0, 6)
              .map((c, i) => (
                <div key={c.code} className="wjc-intro-collision-flag" style={{ "--i": i }}>
                  {renderCountryFlag(c.code, payload, { size: 48 })}
                </div>
              ))}
          </div>
          <div className="wjc-intro-collision-title">
            <h1>WORLD JUNIORS</h1>
            <p>{payload?.wjc_days_total || 11} DAYS · ONE CHAMPION</p>
          </div>
          <div className="wjc-intro-flash" />
        </div>
      )}

      {(phase === "stage-reveal" || (phase === "intro-title" && !fullIntro)) && (
        <div className="wjc-intro-beat wjc-intro-beat--stage-reveal">
          <div className="wjc-intro-stage-bug">
            <span>WORLD JUNIORS</span>
            <b>{payload?.wjc_phase === "complete" ? "FINAL" : payload?.hasData ? "LIVE" : "DESK"}</b>
          </div>
        </div>
      )}
    </div>
  );
}

function HostLowerThird({ hostId, topic, visible }) {
  const host = WJC_HOSTS[hostId] || WJC_HOSTS.host_2;
  return (
    <div
      className={`wjc-host-lower-third wjc-host-lower-third--${hostId}${visible ? " is-visible" : ""}`}
      aria-hidden={!visible}
    >
      <div className="wjc-host-lower-third__rail" />
      <div className="wjc-host-lower-third__body">
        <strong>{host.name.toUpperCase()}</strong>
        <span>{host.role.toUpperCase()}</span>
      </div>
      {topic ? <em className="wjc-host-lower-third__topic">{topic}</em> : null}
    </div>
  );
}

function TopicGraphicOverlay({ item, payload, side }) {
  if (!item?.graphicType) return null;

  const type = item.graphicType;
  const data = item.graphicData;

  if (type === "standings" && asArray(data).length) {
    return (
      <div className={`wjc-topic-graphic wjc-topic-graphic--standings wjc-topic-graphic--${side}`}>
        <header>STANDINGS</header>
        <ul>
          {data.map((row, i) => (
            <li key={row.code}>
              <span>{i + 1}</span>
              {renderCountryFlag(row.code, payload, { size: 32, className: "wjc-topic-graphic__flag" })}
              <b>{row.code}</b>
              <em>{row.pts} PTS</em>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  if (type === "scores" && asArray(data).length) {
    return (
      <div className={`wjc-topic-graphic wjc-topic-graphic--scores wjc-topic-graphic--${side}`}>
        <header>SCORES</header>
        <ul>
          {data.map((g, i) => (
            <li key={`${g.home}-${g.away}-${i}`}>
              <span>FINAL</span>
              <div className="wjc-score-card-line">
                {renderCountryFlag(g.home, payload, { size: 24 })}
                <b>{gameCode(g, "home")} {g.home_goals}</b>
                <span>—</span>
                <b>{g.away_goals} {gameCode(g, "away")}</b>
                {renderCountryFlag(g.away, payload, { size: 24 })}
              </div>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  if ((type === "stock" || type === "stock-takeover") && data) {
    const delta = Number(data.stock_delta) || 0;
    const positive = delta >= 0;
    const major = Math.abs(delta) >= 20;
    return (
      <div
        className={`wjc-topic-graphic wjc-topic-graphic--stock${major ? " is-takeover" : ""}${positive ? " is-up" : " is-down"}`}
      >
        <header>{positive ? (major ? "STOCK EXPLOSION" : "STOCK WATCH") : major ? "STOCK CRASH" : "STOCK WATCH"}</header>
        <strong>{data.name}</strong>
        <div className="wjc-stock-graphic-nation">
          {renderCountryFlag(data.wjc_country, payload, { size: 40 })}
          <span>{data.wjc_country_label || data.wjc_country}</span>
        </div>
        <b className="wjc-stock-graphic-delta">
          {positive ? "+" : ""}
          {delta}
        </b>
        {major && !positive ? <em>FALLING FAST</em> : null}
        {major && positive ? <em>DRAFT BOARD SHOCK</em> : null}
      </div>
    );
  }

  if (type === "user-prospect" && data) {
    return (
      <div className={`wjc-topic-graphic wjc-topic-graphic--prospect wjc-topic-graphic--${side}`}>
        <header>YOUR PROSPECT</header>
        <strong>{data.name}</strong>
        <div className="wjc-stock-graphic-nation">
          {renderCountryFlag(data.wjc_country, payload, { size: 40 })}
          <span>{data.wjc_country_label || data.wjc_country}</span>
        </div>
        <p>
          Age {data.age ?? "—"} · {data.roster || "U20"}
        </p>
      </div>
    );
  }

  if (type?.startsWith("medal-") && data) {
    const medalKey = type.replace("medal-", "");
    const label = data[medalKey] || "—";
    return (
      <div className={`wjc-topic-graphic wjc-topic-graphic--medal wjc-topic-graphic--${medalKey} is-takeover`}>
        <header>{medalKey.toUpperCase()}</header>
        <div className="wjc-medal-graphic-badge">{medalKey.charAt(0).toUpperCase() + medalKey.slice(1)}</div>
        <strong>{label}</strong>
        {medalKey === "gold" ? <p>WORLD JUNIORS CHAMPIONS</p> : null}
      </div>
    );
  }

  return null;
}

function BroadcastControlBar({ onPause, onResume, onSkipLine, onSkipShow, isPaused, visible }) {
  if (!visible) return null;
  return (
    <div className="wjc-broadcast-controls" role="group" aria-label="Broadcast controls">
      {!isPaused ? (
        <button type="button" onClick={onPause} aria-label="Pause broadcast">
          PAUSE
        </button>
      ) : (
        <button type="button" onClick={onResume} aria-label="Resume broadcast">
          RESUME
        </button>
      )}
      <button type="button" onClick={onSkipLine} aria-label="Skip line">
        SKIP LINE
      </button>
      <button type="button" onClick={onSkipShow} aria-label="Skip show">
        SKIP SHOW
      </button>
    </div>
  );
}

function EnterBroadcastOverlay({ onEnter, visible }) {
  if (!visible) return null;
  return (
    <div className="wjc-enter-broadcast">
      <h1>WORLD JUNIORS</h1>
      <button type="button" onClick={onEnter} aria-label="Enter broadcast">
        ENTER BROADCAST
      </button>
    </div>
  );
}

function CinematicSubtitle({ text, hostId, visible }) {
  if (!text || !visible) return null;
  const host = WJC_HOSTS[hostId] || WJC_HOSTS.host_2;
  const side =
    hostId === "host_1" ? "left" : hostId === "host_3" ? "right" : "center";
  return (
    <div
      className={`wjc-subtitle-bubble wjc-subtitle-bubble--${side} wjc-subtitle-bubble--${hostId}`}
      aria-live="polite"
      key={text.slice(0, 48)}
    >
      <div className="wjc-subtitle-bubble__tail" aria-hidden="true" />
      <div className="wjc-subtitle-bubble__content">
        <span className="wjc-subtitle-bubble__host">{host.name}</span>
        <p>{text}</p>
      </div>
    </div>
  );
}

function CenterStagePopup({ popup, payload, visible }) {
  if (!visible || !popup?.type || !popup?.data) return null;

  if (popup.type === "player") {
    const p = popup.data;
    return (
      <div className="wjc-center-popup wjc-center-popup--player" key={`player-${p.player_id || p.name}`}>
        <header>ON-AIR STAT LINE</header>
        <strong>{p.name}</strong>
        <div className="wjc-center-popup__nation">
          {renderCountryFlag(p.wjc_country, payload, { size: 36 })}
          <span>{p.wjc_country_label || p.wjc_country || "—"}</span>
        </div>
        <div className="wjc-center-popup__stats">
          <div>
            <em>G</em>
            <b>{p.g ?? p.tournament_g ?? 0}</b>
          </div>
          <div>
            <em>A</em>
            <b>{p.a ?? 0}</b>
          </div>
          <div>
            <em>PTS</em>
            <b>{p.pts ?? p.tournament_pts ?? 0}</b>
          </div>
          <div>
            <em>GP</em>
            <b>{p.gp ?? p.tournament_gp ?? 0}</b>
          </div>
          <div>
            <em>+/-</em>
            <b>{p.plus_minus ?? 0}</b>
          </div>
        </div>
        {p.stock_delta != null ? (
          <div className={`wjc-center-popup__stock${Number(p.stock_delta) >= 0 ? " is-up" : " is-down"}`}>
            STOCK {Number(p.stock_delta) >= 0 ? "+" : ""}
            {p.stock_delta}
          </div>
        ) : null}
      </div>
    );
  }

  if (popup.type === "team") {
    const t = popup.data;
    return (
      <div className="wjc-center-popup wjc-center-popup--team" key={`team-${t.code}`}>
        <header>TEAM REPORT</header>
        <div className="wjc-center-popup__nation">
          {renderCountryFlag(t.code, payload, { size: 48 })}
          <strong>{t.label || t.code}</strong>
        </div>
        <div className="wjc-center-popup__stats wjc-center-popup__stats--team">
          <div>
            <em>RECORD</em>
            <b>
              {t.w ?? 0}-{t.l ?? 0}
            </b>
          </div>
          <div>
            <em>PTS</em>
            <b>{t.pts ?? 0}</b>
          </div>
          <div>
            <em>GF</em>
            <b>{t.gf ?? 0}</b>
          </div>
          <div>
            <em>GA</em>
            <b>{t.ga ?? 0}</b>
          </div>
        </div>
      </div>
    );
  }

  if (popup.type === "game") {
    const g = popup.data;
    return (
      <div className="wjc-center-popup wjc-center-popup--game" key={`game-${g.home}-${g.away}`}>
        <header>{g.round || "FINAL"}</header>
        <div className="wjc-center-popup__scoreline">
          <div>
            {renderCountryFlag(g.home, payload, { size: 40 })}
            <strong>{gameCode(g, "home")}</strong>
            <b>{g.home_goals ?? "—"}</b>
          </div>
          <span className="wjc-center-popup__vs">—</span>
          <div>
            {renderCountryFlag(g.away, payload, { size: 40 })}
            <strong>{gameCode(g, "away")}</strong>
            <b>{g.away_goals ?? "—"}</b>
          </div>
        </div>
      </div>
    );
  }

  return null;
}

/* -------------------------------------------------------------------------- */
/* 3D stage (compact broadcast desk)                                          */
/* -------------------------------------------------------------------------- */

const CAMERA_TARGETS = {
  wide: { x: 0, y: 2.48, z: 5.15, lookX: 0, lookY: 2.05, lookZ: 0, sway: 0.1 },
  "host-left": { x: -1.08, y: 2.62, z: 3.55, lookX: -1.72, lookY: 2.42, lookZ: 0.17, sway: 0.025 },
  "host-center": { x: 0, y: 2.68, z: 3.35, lookX: 0, lookY: 2.52, lookZ: 0.02, sway: 0.02 },
  "host-right": { x: 1.08, y: 2.62, z: 3.55, lookX: 1.72, lookY: 2.42, lookZ: 0.17, sway: 0.025 },
  graphic: { x: 0, y: 2.55, z: 7.1, lookX: 0, lookY: 1.85, lookZ: 0, sway: 0.02 },
  desk: { x: 0, y: 2.05, z: 5.8, lookX: 0, lookY: 1.55, lookZ: 0, sway: 0.08 },
};

function FloatingCamera() {
  const { cameraMode, reducedMotion } = useBroadcastCamera();
  const current = useRef({ x: 0, y: 2.35, z: 6.35, lookX: 0, lookY: 1.55, lookZ: 0 });

  useFrame(({ clock, camera }) => {
    const t = clock.getElapsedTime();
    const target = CAMERA_TARGETS[cameraMode] || CAMERA_TARGETS.wide;
    const lerp = reducedMotion ? 0.22 : 0.08;
    current.current.x += (target.x - current.current.x) * lerp;
    current.current.y += (target.y - current.current.y) * lerp;
    current.current.z += (target.z - current.current.z) * lerp;
    current.current.lookX += (target.lookX - current.current.lookX) * lerp;
    current.current.lookY += (target.lookY - current.current.lookY) * lerp;
    current.current.lookZ += (target.lookZ - current.current.lookZ) * lerp;

    const sway = reducedMotion ? 0 : target.sway;
    camera.position.x = current.current.x + Math.sin(t * 0.18) * sway;
    camera.position.y = current.current.y + Math.sin(t * 0.22) * (sway * 0.35);
    camera.position.z = current.current.z + Math.sin(t * 0.16) * (sway * 0.85);
    camera.lookAt(current.current.lookX, current.current.lookY, current.current.lookZ);
  });

  return null;
}

function BroadcastLights() {
  const redRef = useRef();
  const blueRef = useRef();
  const whiteRef = useRef();

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();

    if (redRef.current) {
      redRef.current.intensity = 5.3 + Math.sin(t * 1.35) * 0.75;
    }

    if (blueRef.current) {
      blueRef.current.intensity = 4.9 + Math.sin(t * 1.18 + 1.2) * 0.65;
    }

    if (whiteRef.current) {
      whiteRef.current.intensity = 2.8 + Math.sin(t * 0.8) * 0.28;
    }
  });

  return (
    <>
      <ambientLight intensity={0.82} />

      <spotLight
        ref={redRef}
        position={[4.7, 6.2, 4.2]}
        angle={0.42}
        penumbra={0.72}
        intensity={5.6}
        color="#d71920"
        castShadow
      />

      <spotLight
        ref={blueRef}
        position={[-4.7, 6.2, 4.2]}
        angle={0.42}
        penumbra={0.72}
        intensity={5.1}
        color="#1d4ed8"
        castShadow
      />

      <pointLight ref={whiteRef} position={[0, 4.4, 3.5]} intensity={2.9} color="#ffffff" />
      <pointLight position={[0, 0.65, 2.7]} intensity={1.25} color="#f8fafc" />
    </>
  );
}

function CylinderFigure({
  x = 0,
  z = 0,
  scale = 1,
  jacket = "#e5e7eb",
  stripe = "#d71920",
  skin = "#efc29c",
  hair = "#111827",
  delay = 0,
}) {
  const groupRef = useRef();
  const leftArmRef = useRef();
  const rightArmRef = useRef();

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime() + delay;

    if (groupRef.current) {
      groupRef.current.position.y = Math.sin(t * 1.2) * 0.028;
      groupRef.current.rotation.y = Math.sin(t * 0.42) * 0.045;
    }

    if (leftArmRef.current) {
      leftArmRef.current.rotation.z = -0.34 + Math.sin(t * 1.05) * 0.035;
    }

    if (rightArmRef.current) {
      rightArmRef.current.rotation.z = 0.34 + Math.sin(t * 1.1 + 0.4) * 0.035;
    }
  });

  return (
    <group ref={groupRef} position={[x, 0, z]} scale={[scale, scale, scale]}>
      <mesh position={[-0.18, 0.39, 0]} castShadow>
        <cylinderGeometry args={[0.088, 0.105, 0.78, 20]} />
        <meshStandardMaterial color="#111827" roughness={0.78} />
      </mesh>

      <mesh position={[0.18, 0.39, 0]} castShadow>
        <cylinderGeometry args={[0.088, 0.105, 0.78, 20]} />
        <meshStandardMaterial color="#111827" roughness={0.78} />
      </mesh>

      <mesh position={[-0.18, -0.03, 0.11]} castShadow>
        <boxGeometry args={[0.28, 0.09, 0.36]} />
        <meshStandardMaterial color="#030712" roughness={0.85} />
      </mesh>

      <mesh position={[0.18, -0.03, 0.11]} castShadow>
        <boxGeometry args={[0.28, 0.09, 0.36]} />
        <meshStandardMaterial color="#030712" roughness={0.85} />
      </mesh>

      <mesh position={[0, 1.12, 0]} castShadow>
        <cylinderGeometry args={[0.4, 0.48, 1.1, 28]} />
        <meshStandardMaterial color={jacket} roughness={0.7} metalness={0.03} />
      </mesh>

      <mesh position={[0, 1.16, 0.405]} castShadow>
        <boxGeometry args={[0.15, 0.92, 0.032]} />
        <meshStandardMaterial color={stripe} roughness={0.58} metalness={0.04} />
      </mesh>

      <mesh position={[0, 1.55, 0.43]} castShadow>
        <boxGeometry args={[0.44, 0.12, 0.035]} />
        <meshStandardMaterial color="#020617" roughness={0.7} />
      </mesh>

      <mesh position={[0, 1.93, 0]} castShadow>
        <sphereGeometry args={[0.315, 28, 20]} />
        <meshStandardMaterial color={skin} roughness={0.72} />
      </mesh>

      <mesh position={[0, 2.14, 0]} castShadow>
        <sphereGeometry args={[0.322, 28, 12, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color={hair} roughness={0.88} />
      </mesh>

      <mesh position={[-0.095, 1.98, 0.293]}>
        <sphereGeometry args={[0.023, 10, 10]} />
        <meshStandardMaterial color="#020617" />
      </mesh>

      <mesh position={[0.095, 1.98, 0.293]}>
        <sphereGeometry args={[0.023, 10, 10]} />
        <meshStandardMaterial color="#020617" />
      </mesh>

      <mesh position={[0, 1.885, 0.306]}>
        <boxGeometry args={[0.115, 0.018, 0.012]} />
        <meshStandardMaterial color="#7f1d1d" />
      </mesh>

      <mesh ref={leftArmRef} position={[-0.5, 1.16, 0]} rotation={[0, 0, -0.34]} castShadow>
        <cylinderGeometry args={[0.07, 0.088, 0.9, 18]} />
        <meshStandardMaterial color={jacket} roughness={0.7} />
      </mesh>

      <mesh ref={rightArmRef} position={[0.5, 1.16, 0]} rotation={[0, 0, 0.34]} castShadow>
        <cylinderGeometry args={[0.07, 0.088, 0.9, 18]} />
        <meshStandardMaterial color={jacket} roughness={0.7} />
      </mesh>

      <mesh position={[-0.69, 0.8, 0]} castShadow>
        <sphereGeometry args={[0.085, 16, 12]} />
        <meshStandardMaterial color={skin} roughness={0.72} />
      </mesh>

      <mesh position={[0.69, 0.8, 0]} castShadow>
        <sphereGeometry args={[0.085, 16, 12]} />
        <meshStandardMaterial color={skin} roughness={0.72} />
      </mesh>
    </group>
  );
}

function BroadcastStage() {
  const backLogoRef = useRef();

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();

    if (backLogoRef.current) {
      backLogoRef.current.material.opacity = 0.09 + Math.sin(t * 1.1) * 0.025;
      backLogoRef.current.rotation.z = Math.sin(t * 0.15) * 0.015;
    }
  });

  return (
    <group>
      <BroadcastLights />

      <mesh position={[0, 2.25, -2.08]} receiveShadow>
        <boxGeometry args={[10.3, 4.85, 0.18]} />
        <meshStandardMaterial color="#05070d" roughness={0.88} />
      </mesh>

      <mesh position={[-2.95, 2.24, -1.965]}>
        <boxGeometry args={[4.25, 4.35, 0.045]} />
        <meshStandardMaterial color="#123b7a" transparent opacity={0.42} roughness={0.8} />
      </mesh>

      <mesh position={[2.95, 2.24, -1.96]}>
        <boxGeometry args={[4.25, 4.35, 0.045]} />
        <meshStandardMaterial color="#a00020" transparent opacity={0.38} roughness={0.8} />
      </mesh>

      <mesh ref={backLogoRef} position={[0, 2.55, -1.91]}>
        <circleGeometry args={[1.1, 72]} />
        <meshStandardMaterial color="#f8fafc" transparent opacity={0.11} />
      </mesh>

      <mesh position={[0, 0.02, -0.1]} receiveShadow>
        <cylinderGeometry args={[4.45, 4.88, 0.28, 72]} />
        <meshStandardMaterial color="#151924" roughness={0.55} metalness={0.12} />
      </mesh>

      <mesh position={[0, 0.19, -0.1]} receiveShadow>
        <cylinderGeometry args={[4.18, 4.18, 0.045, 72]} />
        <meshStandardMaterial color="#202938" roughness={0.48} metalness={0.12} />
      </mesh>

      <mesh position={[0, 0.215, 1.72]} receiveShadow>
        <boxGeometry args={[7.15, 0.28, 0.36]} />
        <meshStandardMaterial color="#06080f" roughness={0.5} metalness={0.12} />
      </mesh>

      <mesh position={[0, 0.37, 1.91]}>
        <boxGeometry args={[6.4, 0.08, 0.05]} />
        <meshStandardMaterial color="#d71920" roughness={0.32} metalness={0.25} />
      </mesh>

      <CylinderFigure x={-1.72} z={0.17} scale={1.08} jacket="#e5e7eb" stripe="#1d4ed8" delay={0.1} />
      <CylinderFigure x={0} z={0.02} scale={1.24} jacket="#f8fafc" stripe="#d71920" hair="#0a0a0a" delay={0.6} />
      <CylinderFigure x={1.72} z={0.17} scale={1.08} jacket="#d1d5db" stripe="#facc15" delay={1.1} />

      <mesh position={[0, -0.23, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[12.5, 8.5]} />
        <meshStandardMaterial color="#03050b" roughness={0.92} />
      </mesh>
    </group>
  );
}

/* -------------------------------------------------------------------------- */
/* Ticker                                                                     */
/* -------------------------------------------------------------------------- */

function WjcScoreTicker({ items }) {
  const trackRef = useRef(null);
  const text = items.length ? items.join("   ·   ") : "WORLD JUNIORS · AWAITING SCORE FEED";

  return (
    <div className="wjc-ticker" aria-label="WJC score ticker">
      <div className="wjc-ticker__bug">WJC</div>
      <div className="wjc-ticker__track-wrap">
        <div ref={trackRef} className="wjc-ticker__track" aria-hidden="true">
          <span>{text}</span>
          <span>{text}</span>
        </div>
      </div>
    </div>
  );
}


/* -------------------------------------------------------------------------- */
/* Main screen                                                                */
/* -------------------------------------------------------------------------- */

export default function WorldJuniorsMenu({
  eventData,
  franchiseState,
  onClose,
  onBackToHub,
  onSimNextTournamentDay,
  onOpenDraftBoard,
}) {
  const audioRef = useRef(null);
  const queueIndexRef = useRef(0);
  const introTimersRef = useRef([]);
  const speechCleanupRef = useRef(null);
  const queueRef = useRef([]);

  const [isMuted, setIsMuted] = useState(false);
  const [voiceOn, setVoiceOn] = useState(true);
  const [simBusy, setSimBusy] = useState(false);
  const [selectedGame, setSelectedGame] = useState(null);
  const [selectedProspect, setSelectedProspect] = useState(null);
  const [showcaseCard, setShowcaseCard] = useState(null);

  const [broadcastPhase, setBroadcastPhase] = useState("idle");
  const [cameraMode, setCameraMode] = useState("wide");
  const [activeHostId, setActiveHostId] = useState(null);
  const [activeSpeechText, setActiveSpeechText] = useState("");
  const [activeTopic, setActiveTopic] = useState(null);
  const [showHostLowerThird, setShowHostLowerThird] = useState(false);
  const [currentQueueItem, setCurrentQueueItem] = useState(null);
  const [cameraTransition, setCameraTransition] = useState("broadcast-push");
  const [speechVoices, setSpeechVoices] = useState([]);
  const [speechPaused, setSpeechPaused] = useState(false);
  const [needsUserStart, setNeedsUserStart] = useState(true);
  const [graphicTakeover, setGraphicTakeover] = useState(false);
  const [centerPopup, setCenterPopup] = useState(null);
  const [showCenterPopup, setShowCenterPopup] = useState(false);
  const sessionRestoredRef = useRef(false);

  const reducedMotion = useMemo(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }, []);

  const payload = useMemo(
    () => resolveWorldJuniorsPayload(franchiseState, eventData),
    [franchiseState, eventData]
  );

  const draftStockRows = useMemo(
    () => buildWjcDraftStockRows(payload, franchiseState),
    [payload, franchiseState]
  );

  const broadcastQueue = useMemo(
    () => buildBroadcastQueue(payload, draftStockRows),
    [payload, draftStockRows]
  );

  const statLeaders = useMemo(() => buildWjcStatLeaders(payload), [payload]);
  const showcaseCards = useMemo(() => buildWjcShowcaseCards(payload), [payload]);
  const tickerItems = useMemo(() => buildTickerItems(payload), [payload]);
  const dailyIntroLabel = useMemo(() => getDailyIntroLabel(payload), [payload]);
  const fullIntro = useMemo(() => shouldUseFullIntro(payload), [payload]);

  const tournamentGames = useMemo(() => {
    if (asArray(payload.all_games).length) return asArray(payload.all_games);
    const games = [...asArray(payload.round_robin_games)];
    const po = payload.playoffs || {};
    games.push(...asArray(po.quarterfinals));
    games.push(...asArray(po.semifinals));
    if (po.bronze) games.push(po.bronze);
    if (po.gold) games.push(po.gold);
    return games;
  }, [payload]);

  queueRef.current = broadcastQueue;

  const clearIntroTimers = useCallback(() => {
    introTimersRef.current.forEach((id) => window.clearTimeout(id));
    introTimersRef.current = [];
  }, []);

  const cancelSpeech = useCallback(() => {
    if (speechCleanupRef.current) {
      speechCleanupRef.current();
      speechCleanupRef.current = null;
    }
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  }, []);

  const finishBroadcast = useCallback(() => {
    cancelSpeech();
    clearIntroTimers();
    setBroadcastPhase("interactive");
    setCameraMode("wide");
    setActiveHostId(null);
    setActiveSpeechText("");
    setActiveTopic(null);
    setShowHostLowerThird(false);
    setCurrentQueueItem(null);
    setGraphicTakeover(false);
    setCenterPopup(null);
    setShowCenterPopup(false);
    saveWjcSession(payload, "interactive");
  }, [cancelSpeech, clearIntroTimers, payload]);

  const resetBroadcastState = useCallback(() => {
    cancelSpeech();
    clearIntroTimers();
    queueIndexRef.current = 0;
    setBroadcastPhase("idle");
    setCameraMode("wide");
    setActiveHostId(null);
    setActiveSpeechText("");
    setActiveTopic(null);
    setShowHostLowerThird(false);
    setCurrentQueueItem(null);
    setGraphicTakeover(false);
    setSpeechPaused(false);
    setShowcaseCard(null);
    setCenterPopup(null);
    setShowCenterPopup(false);
  }, [cancelSpeech, clearIntroTimers]);

  const applyQueueVisuals = useCallback((item) => {
    if (!item) return;
    setCurrentQueueItem(item);
    setCameraTransition(item.transition || "broadcast-push");
    setCameraMode(item.cameraMode || getHostCameraMode(item.hostId));
    setActiveTopic(item.topic || null);
    setGraphicTakeover(Boolean(item.graphicTakeover));
    if (item.centerPopup) {
      setCenterPopup(item.centerPopup);
      setShowCenterPopup(true);
    } else {
      setShowCenterPopup(false);
    }
    if (item.hostId && !item.silent) {
      setActiveHostId(item.hostId);
    }
  }, []);

  const advanceQueueRef = useRef(null);

  const advanceQueue = useCallback(() => {
    const queue = queueRef.current;
    if (!queue.length || queueIndexRef.current >= queue.length) {
      finishBroadcast();
      return;
    }

    const item = queue[queueIndexRef.current];
    queueIndexRef.current += 1;
    applyQueueVisuals(item);

    if (item.silent) {
      setActiveSpeechText("");
      setShowHostLowerThird(false);
      if (item.centerPopup) {
        setCenterPopup(item.centerPopup);
        setShowCenterPopup(true);
      }
      const hold = item.holdMs || 2400;
      const timer = window.setTimeout(() => advanceQueueRef.current?.(), hold);
      introTimersRef.current.push(timer);
      return;
    }

    const cleanup = speakBroadcastLine({
      text: item.text,
      hostId: item.hostId,
      voiceOn,
      voices: speechVoices,
      onStart: () => {
        setActiveHostId(item.hostId);
        setActiveSpeechText(item.text);
        setShowHostLowerThird(true);
        if (item.centerPopup) {
          setCenterPopup(item.centerPopup);
          setShowCenterPopup(true);
        }
      },
      onEnd: () => {
        setShowHostLowerThird(false);
        setActiveSpeechText("");
        setShowCenterPopup(false);
        const resetTimer = window.setTimeout(() => {
          if (queueIndexRef.current < queue.length) {
            setActiveHostId(null);
            setCameraMode("wide");
          }
          advanceQueueRef.current?.();
        }, reducedMotion ? 200 : 550);
        introTimersRef.current.push(resetTimer);
      },
    });
    speechCleanupRef.current = cleanup;
  }, [applyQueueVisuals, finishBroadcast, reducedMotion, speechVoices, voiceOn]);

  advanceQueueRef.current = advanceQueue;

  const startBroadcastQueue = useCallback(() => {
    cancelSpeech();
    clearIntroTimers();
    queueIndexRef.current = 0;
    setBroadcastPhase("broadcast");
    advanceQueue();
  }, [advanceQueue, cancelSpeech, clearIntroTimers]);

  const runIntroSequence = useCallback(() => {
    clearIntroTimers();
    setBroadcastPhase("intro-title");

    const schedule = (fn, ms) => {
      const id = window.setTimeout(fn, ms);
      introTimersRef.current.push(id);
    };

    if (fullIntro) {
      schedule(() => setBroadcastPhase("intro-countries"), reducedMotion ? 1200 : 2200);
      const montageMs = Math.max(1800, asArray(payload.countries).length * 280 + 400);
      schedule(() => setBroadcastPhase("intro-tournament"), reducedMotion ? 1800 : 2200 + montageMs);
      schedule(
        () => setBroadcastPhase("stage-reveal"),
        reducedMotion ? 2600 : 2200 + montageMs + (reducedMotion ? 1200 : 2000)
      );
      schedule(
        () => {
          setBroadcastPhase("opening");
          startBroadcastQueue();
        },
        reducedMotion ? 3200 : 2200 + montageMs + 2000 + 1600
      );
    } else {
      schedule(() => setBroadcastPhase("stage-reveal"), reducedMotion ? 900 : 1600);
      schedule(
        () => {
          setBroadcastPhase("opening");
          startBroadcastQueue();
        },
        reducedMotion ? 1500 : 2800
      );
    }
  }, [clearIntroTimers, fullIntro, payload.countries, reducedMotion, startBroadcastQueue]);

  const handleEnterBroadcast = useCallback(async () => {
    resetBroadcastState();
    setNeedsUserStart(false);
    const audio = audioRef.current;
    if (audio) {
      try {
        audio.loop = true;
        audio.volume = 0.2;
        audio.muted = isMuted;
        await audio.play();
      } catch (error) {
        console.warn("World Juniors music could not start:", error);
      }
    }
    runIntroSequence();
  }, [isMuted, resetBroadcastState, runIntroSequence]);

  const handleReplayBroadcast = useCallback(() => {
    handleEnterBroadcast();
  }, [handleEnterBroadcast]);

  const handleSkipIntro = useCallback(() => {
    clearIntroTimers();
    setBroadcastPhase("opening");
    startBroadcastQueue();
  }, [clearIntroTimers, startBroadcastQueue]);

  const handlePauseBroadcast = useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.pause();
      setSpeechPaused(true);
    }
  }, []);

  const handleResumeBroadcast = useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.resume();
      setSpeechPaused(false);
    }
  }, []);

  const handleSkipLine = useCallback(() => {
    cancelSpeech();
    setShowHostLowerThird(false);
    setActiveSpeechText("");
    setShowCenterPopup(false);
    setActiveHostId(null);
    setCameraMode("wide");
    advanceQueue();
  }, [advanceQueue, cancelSpeech]);

  const handleSkipShow = useCallback(() => {
    queueIndexRef.current = queueRef.current.length;
    finishBroadcast();
  }, [finishBroadcast]);

  const prospectTournamentStat = useMemo(() => {
    if (!selectedProspect) return null;
    return asArray(payload.player_stats).find(
      (p) => String(p.player_id) === String(selectedProspect.player_id)
    );
  }, [selectedProspect, payload.player_stats]);

  const handleSelectProspect = useCallback(
    (row) => {
      const full =
        draftStockRows.find((r) => String(r.player_id) === String(row.player_id)) ||
        asArray(payload.tournament_prospects).find(
          (r) => String(r.player_id) === String(row.player_id)
        ) ||
        row;
      setSelectedProspect(full);
    },
    [draftStockRows, payload.tournament_prospects]
  );

  const onAirLabel = payload.hasData
    ? payload.wjc_phase === "complete"
      ? "FINAL"
      : "LIVE"
    : "DESK";

  const isIntroActive = ["intro-title", "intro-countries", "intro-tournament", "stage-reveal"].includes(
    broadcastPhase
  );
  const isBroadcastActive = ["opening", "broadcast"].includes(broadcastPhase);
  const showHubInteractive = broadcastPhase === "interactive" || broadcastPhase === "idle";

  const topicGraphicSide =
    activeHostId === "host_1" ? "right" : activeHostId === "host_3" ? "left" : "right";

  useEffect(() => {
    document.body.classList.add("wjc-stage-open");
    return () => document.body.classList.remove("wjc-stage-open");
  }, []);

  useEffect(() => {
    if (sessionRestoredRef.current) return;
    const session = loadWjcSession();
    const key = wjcSessionKey(payload);
    sessionRestoredRef.current = true;

    if (session?.key === key && session.phase === "interactive") {
      setNeedsUserStart(false);
      setBroadcastPhase("interactive");
      return;
    }

    if (payload?.hasData && payload?.wjc_day > 1) {
      setNeedsUserStart(false);
      setBroadcastPhase("interactive");
      saveWjcSession(payload, "interactive");
    }
  }, [payload]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.loop = true;
    audio.volume = 0.2;
    audio.muted = isMuted;
  }, [isMuted]);

  useEffect(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return undefined;
    const loadVoices = () => setSpeechVoices(window.speechSynthesis.getVoices());
    loadVoices();
    window.speechSynthesis.addEventListener("voiceschanged", loadVoices);
    return () => {
      window.speechSynthesis.removeEventListener("voiceschanged", loadVoices);
      window.speechSynthesis.cancel();
    };
  }, []);

  useEffect(() => {
    return () => {
      cancelSpeech();
      clearIntroTimers();
    };
  }, [cancelSpeech, clearIntroTimers]);

  useEffect(() => {
    if (!showcaseCards.length || !isBroadcastActive) return undefined;
    const pick = () => showcaseCards[Math.floor(Math.random() * showcaseCards.length)];
    setShowcaseCard(pick());
    const timer = window.setInterval(() => setShowcaseCard(pick()), 3200);
    return () => window.clearInterval(timer);
  }, [isBroadcastActive, showcaseCards, currentQueueItem?.id]);

  const handleLeave = useCallback(() => {
    cancelSpeech();
    clearIntroTimers();
    if (typeof onClose === "function") onClose();
    else if (typeof onBackToHub === "function") onBackToHub();
  }, [onBackToHub, onClose, cancelSpeech, clearIntroTimers]);

  const handleSimDay = useCallback(async () => {
    if (simBusy || typeof onSimNextTournamentDay !== "function") return;
    setSimBusy(true);
    cancelSpeech();
    clearIntroTimers();
    try {
      await onSimNextTournamentDay();
      queueIndexRef.current = 0;
      setBroadcastPhase("interactive");
      setNeedsUserStart(false);
      setCameraMode("wide");
      setActiveHostId(null);
      setActiveSpeechText("");
      setActiveTopic(null);
      setShowHostLowerThird(false);
      setCurrentQueueItem(null);
      setGraphicTakeover(false);
      setCenterPopup(null);
      setShowCenterPopup(false);
      sessionRestoredRef.current = false;
    } catch (error) {
      console.warn("WJC sim day failed:", error);
    } finally {
      setSimBusy(false);
    }
  }, [onSimNextTournamentDay, simBusy, cancelSpeech, clearIntroTimers]);

  useEffect(() => {
    if (!payload?.hasData) return;
    const session = loadWjcSession();
    const key = wjcSessionKey(payload);
    if (session?.key !== key && broadcastPhase === "interactive") {
      saveWjcSession(payload, "interactive");
    }
  }, [payload, broadcastPhase]);

  const cameraContextValue = useMemo(
    () => ({ cameraMode, activeHostId, reducedMotion }),
    [cameraMode, activeHostId, reducedMotion]
  );

  return (
    <BroadcastCameraContext.Provider value={cameraContextValue}>
      <section
        className={`wjc-stage-root wjc-stage-root--broadcast wjc-broadcast-phase--${broadcastPhase}${graphicTakeover ? " is-graphic-takeover" : ""}`}
        aria-label="World Juniors broadcast hub"
      >
        <audio ref={audioRef} src={worldJuniorsTheme} preload="auto" />

        <EnterBroadcastOverlay
          visible={needsUserStart && broadcastPhase === "idle"}
          onEnter={handleEnterBroadcast}
        />

        <CinematicIntro
          phase={broadcastPhase}
          payload={payload}
          countries={payload.countries}
          fullIntro={fullIntro}
          onSkip={handleSkipIntro}
          dailyLabel={dailyIntroLabel}
        />

        <header className={`wjc-broadcast-header wjc-broadcast-header--slim${isIntroActive ? " is-dimmed" : ""}`}>
          <NationFlagsBar standings={payload.standings} countries={payload.countries} />
          <DeskControls
            audioRef={audioRef}
            isMuted={isMuted}
            setIsMuted={setIsMuted}
            voiceOn={voiceOn}
            setVoiceOn={setVoiceOn}
            onLeave={handleLeave}
            onSimDay={onSimNextTournamentDay ? handleSimDay : null}
            simBusy={simBusy}
            onOpenDraftBoard={onOpenDraftBoard}
          />
        </header>

        <div className={`wjc-broadcast-main${isIntroActive ? " is-intro-active" : ""}`}>
          <DraftStockSidebar rows={draftStockRows} onSelectPlayer={handleSelectProspect} />

          <div className="wjc-broadcast-center" aria-label="Broadcast stage">
            <BroadcastHeroBackdrop />

            <div
              className={`wjc-broadcast-camera viewport camera--${cameraMode} transition--${cameraTransition}${activeHostId ? ` is-speaking-${activeHostId}` : ""}`}
            >
              <div className={`wjc-broadcast-stage broadcast-stage camera--${cameraMode}`}>
                <div className="wjc-broadcast-hero__stage">
                  <div className="wjc-stage-screen-top" aria-hidden="true">
                    <span>{onAirLabel}</span>
                    <b>WORLD JUNIORS</b>
                    <em>{WJC_HOSTS[activeHostId]?.name || "BROADCAST TEAM"}</em>
                  </div>

                  <div className="wjc-intro-stage-bug wjc-intro-stage-bug--persistent">
                    <span>WORLD JUNIORS</span>
                    <b>{onAirLabel}</b>
                  </div>

                  <Canvas
                    className={`wjc-stage-canvas${activeHostId ? ` is-speaking-${activeHostId}` : ""}`}
                    camera={{ position: [0, 2.35, 6.35], fov: 37 }}
                    dpr={[1, 1.5]}
                    shadows
                  >
                    <Suspense fallback={null}>
                      <FloatingCamera />
                      <BroadcastStage />
                    </Suspense>
                  </Canvas>

                  {!graphicTakeover ? (
                    <ShowcaseOverlay card={showcaseCard} onSelectPlayer={handleSelectProspect} />
                  ) : null}

                  <TopicGraphicOverlay item={currentQueueItem} payload={payload} side={topicGraphicSide} />

                  <CenterStagePopup popup={centerPopup} payload={payload} visible={showCenterPopup} />

                  <HostLowerThird hostId={activeHostId} topic={activeTopic} visible={showHostLowerThird} />

                  <CinematicSubtitle
                    text={activeSpeechText}
                    hostId={activeHostId}
                    visible={showHostLowerThird}
                  />

                  <div className="wjc-stage-host-labels" aria-hidden="true">
                    <span
                      className={`wjc-stage-host-labels__left${activeHostId === "host_1" ? " is-active" : activeHostId ? " is-inactive" : ""}`}
                    >
                      Marcus Cole
                    </span>
                    <span
                      className={`wjc-stage-host-labels__center${activeHostId === "host_2" ? " is-active" : activeHostId ? " is-inactive" : ""}`}
                    >
                      Jordan Hayes
                    </span>
                    <span
                      className={`wjc-stage-host-labels__right${activeHostId === "host_3" ? " is-active" : activeHostId ? " is-inactive" : ""}`}
                    >
                      Dr. Elena Park
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <BroadcastControlBar
              visible={isBroadcastActive}
              isPaused={speechPaused}
              onPause={handlePauseBroadcast}
              onResume={handleResumeBroadcast}
              onSkipLine={handleSkipLine}
              onSkipShow={handleSkipShow}
            />

            {showHubInteractive ? (
              <button
                type="button"
                className="wjc-replay-broadcast"
                onClick={handleReplayBroadcast}
                aria-label="Replay broadcast"
              >
                REPLAY BROADCAST
              </button>
            ) : null}
          </div>

          <StatLeadersSidebar leaders={statLeaders} />
        </div>

        <div className={`wjc-hub-row${showHubInteractive ? "" : " wjc-hub-dimmed"}`}>
          <GamesBrowser
            games={tournamentGames}
            onSelectGame={setSelectedGame}
            formatScoreLine={formatScoreLine}
          />
        </div>

        <WjcScoreTicker items={tickerItems} />

        <GameResultModal
          game={selectedGame}
          onClose={() => setSelectedGame(null)}
          formatScoreLine={formatScoreLine}
          gameCode={gameCode}
        />

        <ProspectDetailModal
          prospect={selectedProspect}
          tournamentStats={prospectTournamentStat}
          franchiseState={franchiseState}
          onClose={() => setSelectedProspect(null)}
          onOpenDraftBoard={onOpenDraftBoard}
        />
      </section>
    </BroadcastCameraContext.Provider>
  );
}
