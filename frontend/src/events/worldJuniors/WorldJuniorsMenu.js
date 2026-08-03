import React, { useCallback, useMemo, useState } from "react";
import "./WorldJuniorsMenu.css";

import {
  buildWjcDraftStockRows,
  buildWjcShowcaseCards,
  buildWjcStatLeaders,
} from "./wjcBroadcastBuilder";

import {
  DraftStockSidebar,
  GameResultModal,
  GamesBrowser,
  NationFlagsBar,
  ProspectDetailModal,
  StatLeadersSidebar,
} from "./WorldJuniorsBroadcastPanels";

import { wjcFlagUrl } from "../../utils/countryFlags";

import bg1 from "./gettyimages-136461654-612x612.jpg";
import bg2 from "./gettyimages-136295976-612x612.jpg";
import bg3 from "./gettyimages-95597561-612x612.jpg";
import bg4 from "./gettyimages-84179721-612x612.jpg";
import bg5 from "./gettyimages-2254829655-612x612.jpg";

const WJC_HERO_BACKGROUNDS = [bg1, bg2, bg3, bg4, bg5];

/* -------------------------------------------------------------------------- */
/* Payload resolver                                                           */
/* -------------------------------------------------------------------------- */

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function isWjcPayload(value) {
  if (!value || typeof value !== "object") return false;

  return (
    value.kind === "wjc_tournament" ||
    value.wjc_live === true ||
    Boolean(value.wjc_phase)
  );
}

function findActiveWjcPopup(franchiseState) {
  const popups = [
    ...asArray(franchiseState?.pending_ui_popups),
    ...asArray(franchiseState?.pendingUiPopups),
  ];

  return popups.find((popup) => popup && isWjcPayload(popup)) || null;
}

function findArchivedWjc(franchiseState) {
  const archive = asArray(franchiseState?.showcase_archive);
  const seasonYear = Number(
    franchiseState?.season_year || franchiseState?.seasonYear || 0
  );

  for (let index = archive.length - 1; index >= 0; index -= 1) {
    const entry = archive[index];
    if (!isWjcPayload(entry)) continue;
    if (!seasonYear) return entry;
    const label = String(entry.season_label || "");
    // Prefer this season's archive; skip prior-year WJC desks.
    if (!label || label.startsWith(String(seasonYear))) return entry;
  }

  return null;
}

function normalizePlayoffs(rawPlayoffs) {
  const playoffs =
    rawPlayoffs && typeof rawPlayoffs === "object" ? rawPlayoffs : {};

  return {
    quarterfinals: asArray(playoffs.quarterfinals),
    semifinals: asArray(playoffs.semifinals),
    bronze:
      playoffs.bronze && typeof playoffs.bronze === "object"
        ? playoffs.bronze
        : null,
    gold:
      playoffs.gold && typeof playoffs.gold === "object"
        ? playoffs.gold
        : null,
  };
}

function normalizeWjcFields(raw) {
  const source = raw && typeof raw === "object" ? raw : {};

  const wjcDay = source.wjc_day ?? source.day ?? null;
  const wjcDaysTotal = source.wjc_days_total ?? source.days_total ?? 11;
  const rawPhase = String(source.wjc_phase || source.phase || "").toLowerCase();

  const medalsFinal = Boolean(
    source.medals_final ||
      rawPhase === "complete" ||
      (wjcDay != null && Number(wjcDay) >= Number(wjcDaysTotal))
  );

  return {
    wjc_phase: medalsFinal
      ? "complete"
      : rawPhase || (wjcDay != null ? "live" : ""),

    calendar_iso: String(source.calendar_iso || source.iso || ""),
    wjc_day: wjcDay != null ? Number(wjcDay) : null,
    wjc_days_total: Number(wjcDaysTotal) || 11,

    title: String(source.title || ""),
    season_label: String(source.season_label || ""),

    countries: asArray(source.countries),

    round_robin_games: asArray(source.round_robin_games),
    round_robin_total:
      Number(source.round_robin_total) ||
      asArray(source.round_robin_games).length ||
      0,

    standings: asArray(source.standings),
    playoffs: normalizePlayoffs(source.playoffs),

    medal_labels:
      source.medal_labels && typeof source.medal_labels === "object"
        ? { ...source.medal_labels }
        : {},

    medals_final: medalsFinal,

    user_prospects: asArray(source.user_prospects),
    tournament_prospects: asArray(source.tournament_prospects),
    player_stats: asArray(source.player_stats),

    all_games: asArray(source.all_games),
    games_today: asArray(source.games_today),

    all_games_total:
      Number(source.all_games_total) ||
      asArray(source.all_games).length ||
      0,

    rr_days_total: Number(source.rr_days_total) || 9,
  };
}

function buildCalendarFallback(franchiseState) {
  const hud = franchiseState?.draft_class_hud?.events?.wjc || {};
  const anchors = asArray(franchiseState?.season_anchor_events);

  const wjcAnchor =
    anchors.find((anchor) =>
      String(anchor?.key || "").toLowerCase().includes("wjc_start")
    ) ||
    anchors.find((anchor) =>
      String(anchor?.type || anchor?.id || "")
        .toLowerCase()
        .includes("wjc")
    ) ||
    null;

  const countdown = hud.display || hud.date || "";
  const daysUntil = hud.days_until ?? hud.daysUntil ?? null;
  const startDate = hud.date || wjcAnchor?.date || "";
  const nations =
    asArray(franchiseState?.wjc_nations).length > 0
      ? asArray(franchiseState.wjc_nations)
      : asArray(franchiseState?.wjc_tournament?.countries);

  return {
    ...normalizeWjcFields({ countries: nations }),
    countdown_display: String(countdown || ""),
    countdown_days: daysUntil,
    start_date: String(startDate || ""),
    anchor_title: String(
      wjcAnchor?.title || wjcAnchor?.label || "World Juniors"
    ),
  };
}

function wjcPayloadHasTournamentData(raw) {
  if (!raw || typeof raw !== "object") return false;
  const phase = String(raw.wjc_phase || raw.phase || "").toLowerCase();
  if (phase === "live" || phase === "complete") return true;
  if (raw.wjc_day != null || raw.medals_final) return true;
  if (asArray(raw.all_games).length > 0) return true;
  if (asArray(raw.player_stats).length > 0) return true;
  if (asArray(raw.round_robin_games).length > 0) return true;
  if (asArray(raw.standings).length > 0) return true;
  if (asArray(raw.tournament_prospects).length > 0) return true;
  return false;
}

function isPreTournamentPayload(raw) {
  if (!raw || typeof raw !== "object") return true;
  const phase = String(raw.wjc_phase || raw.phase || "").toLowerCase();
  if (phase === "upcoming") return true;
  if (phase === "live" || phase === "complete") return false;
  if (raw.wjc_day != null || raw.medals_final) return false;
  if (asArray(raw.all_games).length > 0 || asArray(raw.player_stats).length > 0) {
    return false;
  }
  return true;
}

export function resolveWorldJuniorsPayload(franchiseState, eventData) {
  const emptyPayload = {
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

  if (!franchiseState && !eventData) {
    return emptyPayload;
  }

  let source = "calendar";
  let rawPayload = null;

  const activePopup = findActiveWjcPopup(franchiseState);
  const stateTournament =
    franchiseState?.wjc_tournament && isWjcPayload(franchiseState.wjc_tournament)
      ? franchiseState.wjc_tournament
      : null;

  if (activePopup) {
    source = "live";
    rawPayload = activePopup;
  } else if (eventData && isWjcPayload(eventData) && wjcPayloadHasTournamentData(eventData)) {
    source = "eventData";
    rawPayload = eventData;
  } else if (stateTournament && wjcPayloadHasTournamentData(stateTournament)) {
    source = "state";
    rawPayload = stateTournament;
  } else {
    const archivedPayload = findArchivedWjc(franchiseState);

    if (archivedPayload) {
      source = "archive";
      rawPayload = archivedPayload;
    } else if (stateTournament) {
      source = "state";
      rawPayload = stateTournament;
    } else if (eventData && isWjcPayload(eventData)) {
      source = "eventData";
      rawPayload = eventData;
    }
  }

  const calendarMeta = buildCalendarFallback(franchiseState);

  if (rawPayload) {
    const normalized = normalizeWjcFields(rawPayload);
    const countries =
      asArray(normalized.countries).length > 0
        ? normalized.countries
        : calendarMeta.countries;
    const hasData = wjcPayloadHasTournamentData({ ...normalized, countries });
    const isPreTournament = isPreTournamentPayload(normalized);

    return {
      source,
      hasData,
      isPreTournament,
      raw: rawPayload,
      ...normalized,
      countries,
      countdown_display: calendarMeta.countdown_display,
      countdown_days: calendarMeta.countdown_days,
      start_date: calendarMeta.start_date,
      anchor_title: calendarMeta.anchor_title,
    };
  }

  const hasCountdown = Boolean(
    calendarMeta.countdown_display ||
      calendarMeta.countdown_days != null ||
      calendarMeta.start_date
  );

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
  const seasonLabel = payload?.season_label || "";
  const match = seasonLabel.match(/(\d{4})/);

  if (match) {
    return match[1];
  }

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

function gameCode(game, side) {
  return String(
    game?.[side] || game?.[`${side}_label`] || "?"
  )
    .slice(0, 3)
    .toUpperCase();
}

function formatScoreLine(game) {
  const home = gameCode(game, "home");
  const away = gameCode(game, "away");

  const homeGoals = game?.home_goals;
  const awayGoals = game?.away_goals;

  if (homeGoals != null && awayGoals != null) {
    return `${home} ${homeGoals} — ${away} ${awayGoals}`;
  }

  return `${home} vs ${away}`;
}

function formatTickerGame(game, prefix = "FINAL") {
  const home = gameCode(game, "home");
  const away = gameCode(game, "away");

  const homeGoals = game?.home_goals;
  const awayGoals = game?.away_goals;

  if (homeGoals != null && awayGoals != null) {
    return `${prefix}: ${home} ${homeGoals}, ${away} ${awayGoals}`;
  }

  return `${home} vs ${away}`;
}

function buildTickerItems(payload) {
  if (!payload?.hasData) {
    return [];
  }

  const items = [];
  const playoffs = payload.playoffs || {};

  const allGames = asArray(payload.all_games);

  if (allGames.length) {
    allGames.forEach((game) => {
      const prefix = game?.round
        ? String(game.round).toUpperCase().slice(0, 12)
        : "FINAL";

      items.push(formatTickerGame(game, prefix));
    });
  } else {
    asArray(payload.round_robin_games).forEach((game) => {
      items.push(formatTickerGame(game, "FINAL"));
    });

    asArray(playoffs.quarterfinals).forEach((game) => {
      items.push(formatTickerGame(game, "QF FINAL"));
    });

    asArray(playoffs.semifinals).forEach((game) => {
      items.push(formatTickerGame(game, "SF FINAL"));
    });

    if (playoffs.bronze) {
      items.push(formatTickerGame(playoffs.bronze, "BRONZE"));
    }

    if (playoffs.gold) {
      items.push(formatTickerGame(playoffs.gold, "GOLD"));
    }
  }

  if (payload.medals_final && payload.medal_labels) {
    const medals = payload.medal_labels;

    items.push(
      `MEDALS: GOLD ${medals.gold || "—"} · SILVER ${
        medals.silver || "—"
      } · BRONZE ${medals.bronze || "—"}`
    );
  }

  return items;
}

function getTournamentPhaseLabel(day, complete) {
  if (complete || Number(day) >= 11) return "Tournament Complete";
  if (Number(day) === 10) return "Bronze Medal Day";
  if (Number(day) === 9) return "Semifinals";
  if (Number(day) === 8) return "Quarterfinals";
  if (Number(day) >= 1) return "Group Stage";
  return "Pre-Tournament";
}

function getTournamentProgressSteps(payload) {
  const day = Number(payload?.wjc_day) || 0;
  const complete = Boolean(payload?.medals_final);
  const steps = [
    { id: "group", label: "Group Stage", start: 1, end: 7 },
    { id: "qf", label: "Quarterfinals", start: 8, end: 8 },
    { id: "sf", label: "Semifinals", start: 9, end: 9 },
    { id: "medal", label: "Medal Games", start: 10, end: 11 },
  ];

  return steps.map((step) => {
    let state = "upcoming";
    if (complete || day > step.end) state = "complete";
    else if (day >= step.start && day <= step.end) state = "current";
    else if (!payload?.hasData || day < 1) state = step.id === "group" ? "upcoming" : "upcoming";
    return { ...step, state };
  });
}

function formatProspectStatus(prospect) {
  if (prospect?.made_wjc_team === false) return "Cut";
  if (prospect?.injured || prospect?.injury) return "Injured";
  if (prospect?.eliminated) return "Eliminated";
  const raw = String(prospect?.roster || prospect?.role || prospect?.status || "").trim();
  if (!raw) return "Active";
  const lower = raw.toLowerCase();
  if (lower.includes("cut")) return "Cut";
  if (lower.includes("inj")) return "Injured";
  if (lower.includes("elimin")) return "Eliminated";
  if (lower.includes("dnp") || lower.includes("did not")) return "Did Not Play";
  if (lower.includes("select")) return "Selected";
  return raw.length <= 18 ? raw : "Active";
}

function goalDiff(row) {
  const gf = Number(row?.gf);
  const ga = Number(row?.ga);
  if (!Number.isFinite(gf) || !Number.isFinite(ga)) return null;
  return gf - ga;
}

function formatDiff(value) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const n = Number(value);
  if (n > 0) return `+${n}`;
  return String(n);
}

function getFeaturedStory(payload) {
  if (!payload?.hasData) {
    if (payload?.countdown_display) {
      return {
        tag: "UPCOMING",
        headline: payload.anchor_title || "World Juniors",
        sub: payload.countdown_display,
      };
    }

    return {
      tag: "TOURNAMENT",
      headline: "WORLD JUNIORS",
      sub: "Tournament data is not available yet.",
    };
  }

  if (payload.medals_final && payload.medal_labels?.gold) {
    return {
      tag: "CHAMPIONS",
      headline: `${payload.medal_labels.gold} WINS GOLD`,
      sub: "The tournament is complete.",
    };
  }

  const firstStanding = payload.standings?.[0];

  if (firstStanding) {
    return {
      tag:
        Number(payload.wjc_day) >= 8
          ? "MEDAL ROUND"
          : "TOURNAMENT LEADER",

      headline: `${firstStanding.label || firstStanding.code} LEADS`,

      sub: `${firstStanding.pts ?? 0} PTS · ${
        firstStanding.w ?? 0
      }-${firstStanding.l ?? 0} RECORD`,
    };
  }

  return {
    tag: "WORLD JUNIORS",
    headline: payload.title || "TOURNAMENT IN PROGRESS",
    sub: payload.wjc_day
      ? `Day ${payload.wjc_day} of ${payload.wjc_days_total}`
      : "Tournament overview",
  };
}

function collectLoanDecisions(franchiseState) {
  const decisions = asArray(
    franchiseState?.pending_decisions ??
      franchiseState?.pendingDecisions
  );

  return decisions.filter(
    (decision) => decision && decision.kind === "wjc_u20_loan"
  );
}

function collectTournamentGames(payload) {
  if (asArray(payload?.all_games).length) {
    return asArray(payload.all_games);
  }

  const games = [...asArray(payload?.round_robin_games)];
  const playoffs = payload?.playoffs || {};

  games.push(...asArray(playoffs.quarterfinals));
  games.push(...asArray(playoffs.semifinals));

  if (playoffs.bronze) {
    games.push(playoffs.bronze);
  }

  if (playoffs.gold) {
    games.push(playoffs.gold);
  }

  return games;
}

function countryLabelFor(code, payload) {
  const country = asArray(payload?.countries).find(
    (item) => String(item?.code) === String(code)
  );

  return country?.label || code || "Unknown";
}

function safeNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

/* -------------------------------------------------------------------------- */
/* Shared visual components                                                   */
/* -------------------------------------------------------------------------- */

function CountryFlag({
  code,
  payload,
  size = 48,
  className = "",
}) {
  const label = countryLabelFor(code, payload);
  const flagUrl = wjcFlagUrl(code, size);

  return (
    <div
      className={`wjc-page-flag ${className}`.trim()}
      aria-label={`${label} flag`}
    >
      {flagUrl ? (
        <img
          src={flagUrl}
          alt=""
          loading="lazy"
          referrerPolicy="no-referrer"
          onError={(event) => {
            event.currentTarget.style.display = "none";

            const fallback =
              event.currentTarget.nextElementSibling;

            if (fallback) {
              fallback.style.display = "flex";
            }
          }}
        />
      ) : null}

      <span
        className="wjc-page-flag__fallback"
        style={flagUrl ? { display: "none" } : undefined}
      >
        {String(code || "?").slice(0, 3).toUpperCase()}
      </span>
    </div>
  );
}

function WjcScoreTicker({ items }) {
  const text = items.length
    ? items.join("   ·   ")
    : "World Juniors · Tournament Centre";
  const shouldAnimate = items.length > 0;

  return (
    <div className="wjc-ticker" aria-label="World Juniors score ticker">
      <div className="wjc-ticker__bug">WJC</div>
      <div className="wjc-ticker__track-wrap">
        <div
          className={`wjc-ticker__track${shouldAnimate ? " is-animated" : ""}`}
          aria-hidden="true"
        >
          <span>{text}</span>
          {shouldAnimate ? <span>{text}</span> : null}
        </div>
      </div>
    </div>
  );
}

function WjcPageToolbar({
  activeSection,
  onSectionChange,
  onLeave,
  onSimDay,
  onOpenDraftBoard,
  simBusy,
  canSim,
  simLabel,
}) {
  const sections = [
    { id: "overview", label: "Overview" },
    { id: "games", label: "Games" },
    { id: "prospects", label: "Prospects" },
    { id: "playoffs", label: "Bracket" },
  ];

  return (
    <div className="wjc-page-toolbar">
      <nav
        className="wjc-page-tabs"
        role="tablist"
        aria-label="World Juniors sections"
      >
        {sections.map((section) => {
          const selected = activeSection === section.id;
          return (
            <button
              key={section.id}
              type="button"
              role="tab"
              aria-selected={selected}
              className={selected ? "is-active" : ""}
              onClick={() => onSectionChange(section.id)}
            >
              {section.label}
            </button>
          );
        })}
      </nav>

      <div className="wjc-page-actions">
        {typeof onOpenDraftBoard === "function" ? (
          <button
            type="button"
            className="wjc-page-action wjc-page-action--secondary"
            onClick={onOpenDraftBoard}
          >
            Draft Board
          </button>
        ) : null}

        {typeof onSimDay === "function" ? (
          <button
            type="button"
            className="wjc-page-action wjc-page-action--primary"
            onClick={onSimDay}
            disabled={simBusy || !canSim}
            aria-busy={simBusy ? "true" : undefined}
            title={
              canSim
                ? "Advance one calendar day"
                : "Tournament complete"
            }
          >
            {simBusy ? "Simulating..." : simLabel}
          </button>
        ) : null}

        <button
          type="button"
          className="wjc-page-action wjc-page-action--exit"
          onClick={onLeave}
        >
          Back
        </button>
      </div>
    </div>
  );
}

function WjcSummaryHero({
  payload,
  franchiseState,
  heroImage,
  loanDecisionCount,
  gamesCount,
  compact = false,
}) {
  const story = getFeaturedStory(payload);

  const phase = getTournamentPhaseLabel(
    payload.wjc_day,
    payload.medals_final
  );

  const status = payload.isPreTournament
    ? "Upcoming"
    : payload.medals_final
      ? "Final"
      : "Active";

  const dayValue =
    payload.wjc_day != null
      ? `${payload.wjc_day}/${payload.wjc_days_total}`
      : payload.countdown_days != null
        ? `T−${payload.countdown_days}`
        : payload.countdown_display ||
          payload.start_date ||
          "—";

  return (
    <section
      className={`wjc-page-hero${compact ? " is-compact" : ""}`}
      style={{
        backgroundImage: `
          linear-gradient(
            90deg,
            rgba(4, 16, 26, 0.96) 0%,
            rgba(4, 16, 26, 0.82) 48%,
            rgba(4, 16, 26, 0.35) 100%
          ),
          url(${heroImage})
        `,
      }}
    >
      <div className="wjc-page-hero__content">
        <div className="wjc-page-hero__status">
          <span>{status}</span>
          <b>{phase}</b>
        </div>

        {!compact ? (
          <p className="wjc-page-hero__tag">{story.tag}</p>
        ) : null}

        <h1>{story.headline}</h1>

        {!compact ? (
          <p className="wjc-page-hero__sub">{story.sub}</p>
        ) : (
          <p className="wjc-page-hero__sub is-compact-sub">
            Day {dayValue}
            <span aria-hidden="true"> · </span>
            {gamesCount} games
          </p>
        )}

        {!compact ? (
          <div className="wjc-page-hero__facts">
            <div>
              <span>Season</span>
              <strong>{getYear(payload, franchiseState)}</strong>
            </div>
            <div>
              <span>Day</span>
              <strong>{dayValue}</strong>
            </div>
            <div>
              <span>Games</span>
              <strong>{gamesCount}</strong>
            </div>
            <div>
              <span>Your Club</span>
              <strong title={getUserTeamName(franchiseState)}>
                {getUserTeamName(franchiseState)}
              </strong>
            </div>
            {loanDecisionCount > 0 ? (
              <div className="is-alert">
                <span>Loan Decisions</span>
                <strong>{loanDecisionCount}</strong>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Overview                                                                   */
/* -------------------------------------------------------------------------- */

function WjcStandingsTable({ standings, payload }) {
  const rows = asArray(standings);

  return (
    <section className="wjc-page-card wjc-page-standings">
      <header className="wjc-page-card__header">
        <div>
          <span>Tournament Table</span>
          <h2>Standings</h2>
        </div>
        <strong>{rows.length} teams</strong>
      </header>

      {!rows.length ? (
        <div className="wjc-page-empty">
          Standings will appear when tournament games begin.
        </div>
      ) : (
        <div className="wjc-page-table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Team</th>
                <th>GP</th>
                <th>W</th>
                <th>L</th>
                <th>GF</th>
                <th>GA</th>
                <th>Diff</th>
                <th>Pts</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => {
                const diff = goalDiff(row);
                return (
                  <tr
                    key={`${row.code || "team"}-${index}`}
                    className={index === 0 ? "is-leader" : ""}
                  >
                    <td>{index + 1}</td>
                    <td>
                      <div className="wjc-page-team-cell">
                        <CountryFlag
                          code={row.code}
                          payload={payload}
                          size={32}
                        />
                        <div>
                          <strong>{row.code || "—"}</strong>
                          <span>
                            {row.label ||
                              countryLabelFor(row.code, payload)}
                          </span>
                        </div>
                      </div>
                    </td>
                    <td>{row.gp ?? 0}</td>
                    <td>{row.w ?? 0}</td>
                    <td>{row.l ?? 0}</td>
                    <td>{row.gf ?? 0}</td>
                    <td>{row.ga ?? 0}</td>
                    <td
                      className={
                        diff > 0
                          ? "is-positive"
                          : diff < 0
                            ? "is-negative"
                            : ""
                      }
                    >
                      {formatDiff(diff)}
                    </td>
                    <td className="wjc-page-standings__points">
                      {row.pts ?? 0}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function WjcMedalPanel({ payload }) {
  if (!payload?.medals_final) {
    return null;
  }

  const medals = payload.medal_labels || {};

  const entries = [
    { key: "gold", label: "GOLD", team: medals.gold },
    { key: "silver", label: "SILVER", team: medals.silver },
    { key: "bronze", label: "BRONZE", team: medals.bronze },
  ];

  return (
    <section className="wjc-page-card wjc-page-medals">
      <header className="wjc-page-card__header">
        <div>
          <span>FINAL RESULTS</span>
          <h2>Medal Winners</h2>
        </div>
      </header>

      <div className="wjc-page-medals__grid">
        {entries.map((entry) => (
          <div
            key={entry.key}
            className={`wjc-page-medal wjc-page-medal--${entry.key}`}
          >
            <span>{entry.label}</span>
            <strong>{entry.team || "—"}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function getShowcaseName(card) {
  return (
    card?.name ||
    card?.player_name ||
    card?.prospect_name ||
    card?.title ||
    "Tournament Spotlight"
  );
}

function getShowcaseText(card) {
  return (
    card?.summary ||
    card?.description ||
    card?.subtitle ||
    card?.sub ||
    card?.note ||
    "A tournament performance worth monitoring."
  );
}

function WjcShowcaseGrid({
  cards,
  payload,
  onSelectProspect,
}) {
  const visibleCards = asArray(cards).slice(0, 4);

  return (
    <section className="wjc-page-card wjc-page-showcase">
      <header className="wjc-page-card__header">
        <div>
          <span>SCOUTING DESK</span>
          <h2>Tournament Spotlight</h2>
        </div>

        <strong>{visibleCards.length} STORIES</strong>
      </header>

      {!visibleCards.length ? (
        <div className="wjc-page-empty">
          Tournament spotlights will appear as players establish themselves.
        </div>
      ) : (
        <div className="wjc-page-showcase__grid">
          {visibleCards.map((card, index) => {
            const country =
              card?.wjc_country ||
              card?.country ||
              card?.nation ||
              card?.code;

            const stockDelta =
              card?.stock_delta ?? card?.delta ?? null;

            return (
              <button
                key={`${card?.player_id || getShowcaseName(card)}-${index}`}
                type="button"
                className="wjc-page-showcase-card"
                onClick={() => onSelectProspect(card)}
              >
                <div className="wjc-page-showcase-card__top">
                  <CountryFlag
                    code={country}
                    payload={payload}
                    size={40}
                  />

                  <span>
                    {card?.tag ||
                      card?.category ||
                      "TOURNAMENT WATCH"}
                  </span>
                </div>

                <strong>{getShowcaseName(card)}</strong>
                <p>{getShowcaseText(card)}</p>

                {stockDelta != null ? (
                  <div
                    className={`wjc-page-showcase-card__stock ${
                      Number(stockDelta) >= 0
                        ? "is-positive"
                        : "is-negative"
                    }`}
                  >
                    STOCK {Number(stockDelta) >= 0 ? "+" : ""}
                    {stockDelta}
                  </div>
                ) : null}
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}

function WjcProgressStrip({ payload }) {
  const steps = getTournamentProgressSteps(payload);
  return (
    <section className="wjc-page-card wjc-progress-strip" aria-label="Tournament progress">
      <header className="wjc-page-card__header">
        <div>
          <span>Tournament Path</span>
          <h2>Stage Progress</h2>
        </div>
      </header>
      <ol className="wjc-progress-strip__list">
        {steps.map((step) => (
          <li key={step.id} className={`is-${step.state}`}>
            <strong>{step.label}</strong>
            <span>
              {step.state === "complete"
                ? "Complete"
                : step.state === "current"
                  ? "Current"
                  : "Upcoming"}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}

function WjcTodayGamesModule({ payload, onSelectGame }) {
  const today = asArray(payload?.games_today);
  const upcoming = asArray(payload?.all_games)
    .filter((g) => g?.home_goals == null || g?.away_goals == null)
    .slice(0, 4);
  const games = today.length ? today : upcoming;

  return (
    <section className="wjc-page-card wjc-today-games">
      <header className="wjc-page-card__header">
        <div>
          <span>{today.length ? "Today" : "Next"}</span>
          <h2>{today.length ? "Today's Games" : "Upcoming Games"}</h2>
        </div>
        <strong>{games.length} listed</strong>
      </header>
      {!games.length ? (
        <div className="wjc-page-empty">No games scheduled.</div>
      ) : (
        <div className="wjc-today-games__list">
          {games.map((game, index) => {
            const complete =
              game.home_goals != null && game.away_goals != null;
            return (
              <button
                key={`${game.home}-${game.away}-${game.game_day}-${index}`}
                type="button"
                className="wjc-today-games__row"
                onClick={() => onSelectGame?.(game)}
              >
                <em>{game.round || "Game"}</em>
                <strong>
                  {String(game.away || "?").slice(0, 3).toUpperCase()}
                  {" "}
                  {complete ? game.away_goals : "—"}
                  {" — "}
                  {complete ? game.home_goals : "—"}
                  {" "}
                  {String(game.home || "?").slice(0, 3).toUpperCase()}
                </strong>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}

function WjcOverviewSection({
  payload,
  showcaseCards,
  onSelectProspect,
  onSelectGame,
  userProspectCount,
  gamesCount,
}) {
  return (
    <div className="wjc-page-section-stack">
      <div className="wjc-overview-summary">
        <div>
          <span>Stage</span>
          <strong>
            {getTournamentPhaseLabel(payload.wjc_day, payload.medals_final)}
          </strong>
        </div>
        <div>
          <span>Day</span>
          <strong>
            {payload.wjc_day != null
              ? `${payload.wjc_day}/${payload.wjc_days_total}`
              : "—"}
          </strong>
        </div>
        <div>
          <span>Games</span>
          <strong>{gamesCount}</strong>
        </div>
        <div>
          <span>Your Prospects</span>
          <strong>{userProspectCount}</strong>
        </div>
      </div>

      <WjcProgressStrip payload={payload} />
      <WjcMedalPanel payload={payload} />
      <WjcStandingsTable
        standings={payload.standings}
        payload={payload}
      />
      <WjcTodayGamesModule
        payload={payload}
        onSelectGame={onSelectGame}
      />
      <WjcShowcaseGrid
        cards={showcaseCards}
        payload={payload}
        onSelectProspect={onSelectProspect}
      />
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Prospect page                                                              */
/* -------------------------------------------------------------------------- */

function mergeUserProspectsWithStats(payload) {
  const stats = asArray(payload?.player_stats);
  const tournamentProspects = asArray(
    payload?.tournament_prospects
  );

  return asArray(payload?.user_prospects).map((prospect) => {
    const tournamentProfile = tournamentProspects.find(
      (candidate) =>
        String(candidate?.player_id) ===
        String(prospect?.player_id)
    );

    const tournamentStats = stats.find(
      (candidate) =>
        String(candidate?.player_id) ===
        String(prospect?.player_id)
    );

    return {
      ...prospect,
      ...(tournamentProfile || {}),
      ...(tournamentStats || {}),
    };
  });
}

function WjcProspectsSection({
  payload,
  prospects,
  onSelectProspect,
  onOpenDraftBoard,
}) {
  const sorted = useMemo(() => {
    const list = [...asArray(prospects)];
    list.sort((a, b) => {
      const aCut = a.made_wjc_team === false ? 1 : 0;
      const bCut = b.made_wjc_team === false ? 1 : 0;
      if (aCut !== bCut) return aCut - bCut;
      const aPts = Number(a.pts ?? a.tournament_pts) || 0;
      const bPts = Number(b.pts ?? b.tournament_pts) || 0;
      if (bPts !== aPts) return bPts - aPts;
      const aRank = Number(a.stock_after ?? a.stock_before) || 999;
      const bRank = Number(b.stock_after ?? b.stock_before) || 999;
      return aRank - bRank;
    });
    return list;
  }, [prospects]);

  const activeCount = sorted.filter(
    (p) => formatProspectStatus(p) === "Active" || formatProspectStatus(p) === "Selected"
  ).length;
  const cutCount = sorted.filter((p) => formatProspectStatus(p) === "Cut").length;

  return (
    <section className="wjc-page-card wjc-page-prospects">
      <header className="wjc-page-card__header">
        <div>
          <span>Organization Tracker</span>
          <h2>Your WJC Prospects</h2>
        </div>
        <div className="wjc-page-prospects__actions">
          <strong>
            {sorted.length} selected
            {cutCount ? ` · ${cutCount} cut` : ""}
            {activeCount ? ` · ${activeCount} active` : ""}
          </strong>
          {typeof onOpenDraftBoard === "function" ? (
            <button
              type="button"
              className="wjc-page-inline-action"
              onClick={onOpenDraftBoard}
            >
              Open Draft Board
            </button>
          ) : null}
        </div>
      </header>

      {!sorted.length ? (
        <div className="wjc-page-empty wjc-page-empty--structured">
          <strong>No organization prospects are participating.</strong>
          <p>Review tournament draft risers in the side board, or open the draft class.</p>
          {typeof onOpenDraftBoard === "function" ? (
            <button
              type="button"
              className="wjc-page-inline-action"
              onClick={onOpenDraftBoard}
            >
              Open Draft Board
            </button>
          ) : null}
        </div>
      ) : sorted.length === 1 ? (
        <div className="wjc-prospect-spotlight">
          {(() => {
            const prospect = sorted[0];
            const goals = prospect.g ?? prospect.tournament_g ?? 0;
            const assists = prospect.a ?? prospect.tournament_a ?? 0;
            const points =
              prospect.pts ??
              prospect.tournament_pts ??
              safeNumber(goals) + safeNumber(assists);
            const gamesPlayed = prospect.gp ?? prospect.tournament_gp ?? 0;
            const stockDelta =
              prospect.stock_delta ?? prospect.draft_stock_delta;
            const status = formatProspectStatus(prospect);
            return (
              <button
                type="button"
                className="wjc-prospect-spotlight__card"
                onClick={() => onSelectProspect(prospect)}
              >
                <div className="wjc-prospect-spotlight__top">
                  <CountryFlag
                    code={
                      prospect.wjc_country ||
                      prospect.country ||
                      prospect.nation
                    }
                    payload={payload}
                    size={48}
                  />
                  <div>
                    <strong>{prospect.name || "Unknown Player"}</strong>
                    <span>
                      {prospect.position || "—"} · Age {prospect.age ?? "—"}
                    </span>
                    <em className="wjc-status-pill">{status}</em>
                  </div>
                </div>
                <div className="wjc-prospect-spotlight__stats">
                  <div><span>GP</span><strong>{gamesPlayed}</strong></div>
                  <div><span>G</span><strong>{goals}</strong></div>
                  <div><span>A</span><strong>{assists}</strong></div>
                  <div><span>PTS</span><strong>{points}</strong></div>
                  <div>
                    <span>Stock</span>
                    <strong>
                      {stockDelta == null
                        ? "—"
                        : `${Number(stockDelta) >= 0 ? "+" : ""}${stockDelta}`}
                    </strong>
                  </div>
                </div>
              </button>
            );
          })()}
        </div>
      ) : (
        <div className="wjc-page-prospect-list">
          {sorted.map((prospect) => {
            const goals = prospect.g ?? prospect.tournament_g ?? 0;
            const assists = prospect.a ?? prospect.tournament_a ?? 0;
            const points =
              prospect.pts ??
              prospect.tournament_pts ??
              safeNumber(goals) + safeNumber(assists);
            const gamesPlayed = prospect.gp ?? prospect.tournament_gp ?? 0;
            const stockDelta =
              prospect.stock_delta ?? prospect.draft_stock_delta;
            const status = formatProspectStatus(prospect);

            return (
              <button
                key={prospect.player_id || prospect.name}
                type="button"
                className="wjc-page-prospect-row"
                onClick={() => onSelectProspect(prospect)}
              >
                <div className="wjc-page-prospect-row__identity">
                  <CountryFlag
                    code={
                      prospect.wjc_country ||
                      prospect.country ||
                      prospect.nation
                    }
                    payload={payload}
                    size={44}
                  />
                  <div>
                    <strong>{prospect.name || "Unknown Player"}</strong>
                    <span>
                      {prospect.position || "—"} · Age {prospect.age ?? "—"}
                      {prospect.stock_after != null || prospect.stock_before != null
                        ? ` · #${prospect.stock_after ?? prospect.stock_before}`
                        : ""}
                    </span>
                  </div>
                </div>
                <div className="wjc-page-prospect-row__status">
                  <span>Status</span>
                  <strong>{status}</strong>
                </div>
                <div className="wjc-page-prospect-row__stat">
                  <span>GP</span>
                  <strong>{gamesPlayed}</strong>
                </div>
                <div className="wjc-page-prospect-row__stat">
                  <span>G</span>
                  <strong>{goals}</strong>
                </div>
                <div className="wjc-page-prospect-row__stat">
                  <span>A</span>
                  <strong>{assists}</strong>
                </div>
                <div className="wjc-page-prospect-row__stat">
                  <span>PTS</span>
                  <strong>{points}</strong>
                </div>
                <div
                  className={`wjc-page-prospect-row__stock ${
                    stockDelta == null
                      ? ""
                      : Number(stockDelta) >= 0
                        ? "is-positive"
                        : "is-negative"
                  }`}
                >
                  <span>Stock</span>
                  <strong>
                    {stockDelta == null
                      ? "—"
                      : `${Number(stockDelta) >= 0 ? "+" : ""}${stockDelta}`}
                  </strong>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Playoff page                                                               */
/* -------------------------------------------------------------------------- */

function WjcPlayoffGame({
  game,
  payload,
  label,
  onSelectGame,
}) {
  if (!game) {
    return (
      <div className="wjc-page-playoff-game is-empty">
        <span>{label}</span>
        <strong>TBD</strong>
      </div>
    );
  }

  const homeGoals = game.home_goals;
  const awayGoals = game.away_goals;

  const complete =
    homeGoals != null && awayGoals != null;
  const homeWins = complete && Number(homeGoals) > Number(awayGoals);
  const awayWins = complete && Number(awayGoals) > Number(homeGoals);

  return (
    <button
      type="button"
      className="wjc-page-playoff-game"
      onClick={() => onSelectGame(game)}
    >
      <span className="wjc-page-playoff-game__label">
        {label}
      </span>

      <div className={`wjc-page-playoff-game__team${homeWins ? " is-winner" : ""}`}>
        <CountryFlag
          code={game.home}
          payload={payload}
          size={30}
        />

        <strong>{gameCode(game, "home")}</strong>
        <b>{complete ? homeGoals : "—"}</b>
      </div>

      <div className={`wjc-page-playoff-game__team${awayWins ? " is-winner" : ""}`}>
        <CountryFlag
          code={game.away}
          payload={payload}
          size={30}
        />

        <strong>{gameCode(game, "away")}</strong>
        <b>{complete ? awayGoals : "—"}</b>
      </div>
    </button>
  );
}

function WjcPlayoffsSection({
  payload,
  onSelectGame,
}) {
  const playoffs = payload.playoffs || {};
  const phase = getTournamentPhaseLabel(
    payload.wjc_day,
    payload.medals_final
  );
  const seeded =
    asArray(playoffs.quarterfinals).length > 0 ||
    asArray(playoffs.semifinals).length > 0 ||
    Boolean(playoffs.bronze) ||
    Boolean(playoffs.gold);

  return (
    <section className="wjc-page-card wjc-page-playoffs">
      <header className="wjc-page-card__header">
        <div>
          <span>Medal Round</span>
          <h2>Bracket</h2>
        </div>
        <strong>{phase}</strong>
      </header>

      {!seeded ? (
        <div className="wjc-bracket-banner">
          Seeding pending while group play continues.
        </div>
      ) : null}

      <div className="wjc-page-bracket">
        <div className="wjc-page-bracket__round">
          <h3>Quarterfinals</h3>
          {asArray(playoffs.quarterfinals).length ? (
            asArray(playoffs.quarterfinals).map((game, index) => (
              <WjcPlayoffGame
                key={`quarterfinal-${index}`}
                game={game}
                payload={payload}
                label={`QF ${index + 1}`}
                onSelectGame={onSelectGame}
              />
            ))
          ) : (
            <>
              <WjcPlayoffGame label="QF 1" />
              <WjcPlayoffGame label="QF 2" />
              <WjcPlayoffGame label="QF 3" />
              <WjcPlayoffGame label="QF 4" />
            </>
          )}
        </div>

        <div className="wjc-page-bracket__round">
          <h3>Semifinals</h3>
          {asArray(playoffs.semifinals).length ? (
            asArray(playoffs.semifinals).map((game, index) => (
              <WjcPlayoffGame
                key={`semifinal-${index}`}
                game={game}
                payload={payload}
                label={`SF ${index + 1}`}
                onSelectGame={onSelectGame}
              />
            ))
          ) : (
            <>
              <WjcPlayoffGame label="SF 1" />
              <WjcPlayoffGame label="SF 2" />
            </>
          )}
        </div>

        <div className="wjc-page-bracket__round wjc-page-bracket__round--medals">
          <h3>Medal Games</h3>
          <WjcPlayoffGame
            game={playoffs.bronze}
            payload={payload}
            label="Bronze"
            onSelectGame={onSelectGame}
          />
          <WjcPlayoffGame
            game={playoffs.gold}
            payload={payload}
            label="Gold"
            onSelectGame={onSelectGame}
          />
        </div>
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Main page                                                                  */
/* -------------------------------------------------------------------------- */

export default function WorldJuniorsMenu({
  eventData,
  franchiseState,
  onClose,
  onBackToHub,
  onSimNextTournamentDay,
  onOpenDraftBoard,
}) {
  const [activeSection, setActiveSection] =
    useState("overview");

  const [selectedGame, setSelectedGame] =
    useState(null);

  const [selectedProspect, setSelectedProspect] =
    useState(null);

  const [simBusy, setSimBusy] =
    useState(false);

  const payload = useMemo(
    () =>
      resolveWorldJuniorsPayload(
        franchiseState,
        eventData
      ),
    [franchiseState, eventData]
  );

  const draftStockRows = useMemo(
    () =>
      buildWjcDraftStockRows(
        payload,
        franchiseState
      ),
    [payload, franchiseState]
  );

  const statLeaders = useMemo(
    () => buildWjcStatLeaders(payload),
    [payload]
  );

  const showcaseCards = useMemo(
    () => buildWjcShowcaseCards(payload),
    [payload]
  );

  const tickerItems = useMemo(
    () => buildTickerItems(payload),
    [payload]
  );

  const tournamentGames = useMemo(
    () => collectTournamentGames(payload),
    [payload]
  );

  const userProspects = useMemo(
    () => mergeUserProspectsWithStats(payload),
    [payload]
  );

  const loanDecisions = useMemo(
    () => collectLoanDecisions(franchiseState),
    [franchiseState]
  );

  const heroImage = useMemo(() => {
    const index = Math.max(
      0,
      (safeNumber(payload.wjc_day, 1) - 1) %
        WJC_HERO_BACKGROUNDS.length
    );

    return WJC_HERO_BACKGROUNDS[index];
  }, [payload.wjc_day]);

  const prospectTournamentStats = useMemo(() => {
    if (!selectedProspect) {
      return null;
    }

    return asArray(payload.player_stats).find(
      (player) =>
        String(player?.player_id) ===
        String(selectedProspect?.player_id)
    );
  }, [payload.player_stats, selectedProspect]);

  const handleLeave = useCallback(() => {
    if (typeof onClose === "function") {
      onClose();
      return;
    }

    if (typeof onBackToHub === "function") {
      onBackToHub();
    }
  }, [onBackToHub, onClose]);

  const handleSelectProspect = useCallback(
    (row) => {
      if (!row) return;

      const fullProfile =
        draftStockRows.find(
          (candidate) =>
            String(candidate?.player_id) ===
            String(row?.player_id)
        ) ||
        asArray(payload.tournament_prospects).find(
          (candidate) =>
            String(candidate?.player_id) ===
            String(row?.player_id)
        ) ||
        asArray(payload.user_prospects).find(
          (candidate) =>
            String(candidate?.player_id) ===
            String(row?.player_id)
        ) ||
        row;

      setSelectedProspect({
        ...row,
        ...fullProfile,
      });
    },
    [
      draftStockRows,
      payload.tournament_prospects,
      payload.user_prospects,
    ]
  );

  const handleSimDay = useCallback(async () => {
    if (
      simBusy ||
      typeof onSimNextTournamentDay !== "function"
    ) {
      return;
    }

    setSimBusy(true);

    try {
      await onSimNextTournamentDay();
    } catch (error) {
      console.warn(
        "World Juniors simulation failed:",
        error
      );
    } finally {
      setSimBusy(false);
    }
  }, [onSimNextTournamentDay, simBusy]);

  const canSim =
    typeof onSimNextTournamentDay === "function" &&
    !payload.medals_final;

  const simLabel =
    payload.isPreTournament || !payload.hasData
      ? "Sim Day"
      : "Sim Next Day";

  const heroCompact = activeSection !== "overview";
  const completedGames = tournamentGames.filter(
    (g) => g?.home_goals != null && g?.away_goals != null
  ).length;

  return (
    <section
      className={`wjc-page-root wjc-page-root--${activeSection}`}
      data-register="ops"
      aria-label="World Juniors tournament centre"
    >
      <header className="wjc-page-header">
        <div className="wjc-page-brand">
          <span className="wjc-page-brand__mark">WJC</span>
          <div>
            <p>IIHF · U20 Championship</p>
            <h1>World Juniors</h1>
          </div>
        </div>
        <div className="wjc-page-header__flags">
          <NationFlagsBar
            standings={payload.standings}
            countries={payload.countries}
          />
        </div>
      </header>

      <WjcPageToolbar
        activeSection={activeSection}
        onSectionChange={setActiveSection}
        onLeave={handleLeave}
        onSimDay={handleSimDay}
        onOpenDraftBoard={onOpenDraftBoard}
        simBusy={simBusy}
        canSim={canSim}
        simLabel={simLabel}
      />

      <main className="wjc-page-main">
        <WjcSummaryHero
          payload={payload}
          franchiseState={franchiseState}
          heroImage={heroImage}
          loanDecisionCount={loanDecisions.length}
          gamesCount={tournamentGames.length}
          compact={heroCompact}
        />

        {payload.isPreTournament ? (
          <section className="wjc-page-countdown">
            <div>
              <span>Next Tournament</span>
              <strong>
                {payload.countdown_display ||
                  payload.start_date ||
                  "Schedule unavailable"}
              </strong>
            </div>
            {payload.countdown_days != null ? (
              <div>
                <span>Days Until Start</span>
                <strong>{payload.countdown_days}</strong>
              </div>
            ) : null}
          </section>
        ) : null}

        <div className="wjc-page-workspace">
          <aside className="wjc-page-rail wjc-page-rail--left">
            <DraftStockSidebar
              rows={draftStockRows}
              onSelectPlayer={handleSelectProspect}
            />
          </aside>

          <section className="wjc-page-content" role="tabpanel">
            {activeSection === "overview" ? (
              <WjcOverviewSection
                payload={payload}
                showcaseCards={showcaseCards}
                onSelectProspect={handleSelectProspect}
                onSelectGame={setSelectedGame}
                userProspectCount={userProspects.length}
                gamesCount={tournamentGames.length}
              />
            ) : null}

            {activeSection === "games" ? (
              <section className="wjc-page-card wjc-page-games">
                <header className="wjc-page-card__header">
                  <div>
                    <span>Tournament Schedule</span>
                    <h2>Games and Results</h2>
                  </div>
                  <strong>
                    {completedGames} completed
                    <span aria-hidden="true"> · </span>
                    {tournamentGames.length} total
                    {payload.wjc_day != null
                      ? ` · Day ${payload.wjc_day}`
                      : ""}
                  </strong>
                </header>
                <GamesBrowser
                  games={tournamentGames}
                  onSelectGame={setSelectedGame}
                  formatScoreLine={formatScoreLine}
                />
              </section>
            ) : null}

            {activeSection === "prospects" ? (
              <WjcProspectsSection
                payload={payload}
                prospects={userProspects}
                onSelectProspect={handleSelectProspect}
                onOpenDraftBoard={onOpenDraftBoard}
              />
            ) : null}

            {activeSection === "playoffs" ? (
              <WjcPlayoffsSection
                payload={payload}
                onSelectGame={setSelectedGame}
              />
            ) : null}
          </section>

          <aside className="wjc-page-rail wjc-page-rail--right">
            <StatLeadersSidebar leaders={statLeaders} />
          </aside>
        </div>
      </main>

      <WjcScoreTicker items={tickerItems} />

      <GameResultModal
        game={selectedGame}
        onClose={() => setSelectedGame(null)}
        formatScoreLine={formatScoreLine}
        gameCode={gameCode}
      />

      <ProspectDetailModal
        prospect={selectedProspect}
        tournamentStats={prospectTournamentStats}
        franchiseState={franchiseState}
        onClose={() => setSelectedProspect(null)}
        onOpenDraftBoard={onOpenDraftBoard}
      />
    </section>
  );
}