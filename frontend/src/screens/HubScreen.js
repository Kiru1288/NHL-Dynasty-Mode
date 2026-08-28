import React, { useCallback, useEffect, useMemo } from "react";
import { useGameUI } from "../game/GameUIContext";
import { HUB_MENU, SCREENS } from "../game/constants";
import { GameFooter } from "../components/game/GameFooter";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";
import { getFranchisePhaseCta } from "../events/FranchiseEventOverlay";
import "../styles/motion-control.css";
import FirstPersonOfficeHub, {
  FRANCHISE_COMMAND_REGISTRY,
  OFFICE_NAV_TARGETS,
  OFFICE_PANEL_IDS,
  resolveCommandTarget,
} from "./FirstPersonOfficeHub";

const ROUTE_SCREENS = { ...SCREENS };

const menuIndexById = (id, fallback = 0) => {
  const idx = HUB_MENU.findIndex(
    (m) => String(m.id).toLowerCase() === String(id).toLowerCase()
  );
  return idx >= 0 ? idx : fallback;
};

const HUB_INDEX_FOR_SCREEN = {
  [ROUTE_SCREENS.ROSTER]: menuIndexById("roster", 0),
  [ROUTE_SCREENS.CALENDAR]: menuIndexById("calendar", 1),
  [ROUTE_SCREENS.CHEMISTRY]: menuIndexById("chemistry", 2),
  [ROUTE_SCREENS.EDIT_LINES]: menuIndexById("roster", 0),
  [ROUTE_SCREENS.POWER_PLAY]: menuIndexById("roster", 0),
  [ROUTE_SCREENS.PENALTY_KILL]: menuIndexById("roster", 0),
  [ROUTE_SCREENS.STATS]: menuIndexById("stats", 2),
  [ROUTE_SCREENS.TRADE]: menuIndexById("ops", 3),
  [ROUTE_SCREENS.DRAFT_CLASS]: menuIndexById("draft_class", 4),
  [ROUTE_SCREENS.DRAFT_LOTTERY]: menuIndexById("draft_class", 4),
  [ROUTE_SCREENS.TEAM_NEEDS]: menuIndexById("draft_class", 4),
  [ROUTE_SCREENS.SCOUTING]: menuIndexById("scouting", 5),
  [ROUTE_SCREENS.HUB]: menuIndexById("office", 6),
  [ROUTE_SCREENS.OFFICE]: menuIndexById("office", 6),
  [ROUTE_SCREENS.GM_WORLD]: menuIndexById("office", 6),
  [ROUTE_SCREENS.LEAGUE_OPERATIONS]: menuIndexById("office", 6),
  [ROUTE_SCREENS.CAP_LEDGER]: menuIndexById("office", 6),
  [ROUTE_SCREENS.SETTINGS]: menuIndexById("settings", 7),
  [ROUTE_SCREENS.STORYLINES]: menuIndexById("stats", 2),
  [ROUTE_SCREENS.PLACEHOLDER]: menuIndexById("office", 6),
};

const OFFICE_PANEL_TO_SCREEN = {
  [OFFICE_PANEL_IDS.DASHBOARD]: ROUTE_SCREENS.HUB,
  [OFFICE_PANEL_IDS.CALENDAR]: ROUTE_SCREENS.CALENDAR,
  [OFFICE_PANEL_IDS.SCOUTING]: ROUTE_SCREENS.SCOUTING,
  [OFFICE_PANEL_IDS.CONTRACTS]: ROUTE_SCREENS.CAP_LEDGER,
  [OFFICE_PANEL_IDS.STATS]: ROUTE_SCREENS.STATS,
  [OFFICE_PANEL_IDS.LINES]: ROUTE_SCREENS.EDIT_LINES,
  [OFFICE_PANEL_IDS.NEWS]: ROUTE_SCREENS.STORYLINES,
  [OFFICE_PANEL_IDS.DRAFT]: ROUTE_SCREENS.DRAFT_CLASS,
  [OFFICE_PANEL_IDS.DRAFT_CLASS]: ROUTE_SCREENS.DRAFT_CLASS,
  [OFFICE_PANEL_IDS.ROSTER]: ROUTE_SCREENS.ROSTER,
  [OFFICE_PANEL_IDS.STANDINGS]: ROUTE_SCREENS.STATS,
  [OFFICE_PANEL_IDS.MESSAGES]: ROUTE_SCREENS.TRADE,
  [OFFICE_PANEL_IDS.AWARDS]: ROUTE_SCREENS.STATS,
  [OFFICE_PANEL_IDS.GAME_DAY]: ROUTE_SCREENS.PLACEHOLDER,
  [OFFICE_PANEL_IDS.TEAM_IDENTITY]: ROUTE_SCREENS.PLACEHOLDER,
  [OFFICE_PANEL_IDS.TASKS]: ROUTE_SCREENS.PLACEHOLDER,
  [OFFICE_PANEL_IDS.LEAGUE_CENTRAL]: ROUTE_SCREENS.LEAGUE_OPERATIONS,
};

function syncHubIndexForScreen(screen, setHubMenuIndex) {
  const idx = HUB_INDEX_FOR_SCREEN[screen];
  if (idx != null) {
    setHubMenuIndex(idx);
  }
}

function navigateFranchiseCommand(
  target,
  {
    setScreen,
    setHubMenuIndex,
    setCapLedgerTab,
    setStatsCentralTab,
    openCommandPlaceholder,
    openFranchiseEvent,
    onReopenOffseasonStage,
  }
) {
  const resolution = resolveCommandTarget(target);
  if (!resolution) {
    console.warn("[OfficeNav] Unknown franchise command target:", target);
    openCommandPlaceholder?.({
      title: "Unknown Command",
      subtitle: "This navigation target is not registered.",
      description: `Target id: ${String(target || "empty")}`,
      targetId: target,
    });
    return false;
  }

  if (resolution.type === "placeholder") {
    openCommandPlaceholder?.({
      ...resolution.placeholder,
      targetId: resolution.placeholder?.targetId || target,
    });
    syncHubIndexForScreen(ROUTE_SCREENS.PLACEHOLDER, setHubMenuIndex);
    return true;
  }

  if (resolution.type === "hub") {
    syncHubIndexForScreen(ROUTE_SCREENS.HUB, setHubMenuIndex);
    setScreen(ROUTE_SCREENS.HUB);
    return true;
  }

  if (resolution.type === "franchise_event") {
    const stage = String(resolution.stage || "free_agency");
    syncHubIndexForScreen(ROUTE_SCREENS.HUB, setHubMenuIndex);
    setScreen(ROUTE_SCREENS.HUB);
    const open = () => {
      if (typeof openFranchiseEvent === "function") openFranchiseEvent();
    };
    // Only step the server stage back when Roster Check (or later) is blocking
    // access to the Free Agency Wire.
    if (typeof onReopenOffseasonStage === "function") {
      Promise.resolve(onReopenOffseasonStage(stage))
        .catch(() => null)
        .finally(open);
    } else {
      open();
    }
    return true;
  }

  if (resolution.type === "screen") {
    if (resolution.capTab && setCapLedgerTab) {
      setCapLedgerTab(resolution.capTab);
    }
    if (resolution.statsTab && setStatsCentralTab) {
      setStatsCentralTab(resolution.statsTab);
    }
    syncHubIndexForScreen(resolution.screen, setHubMenuIndex);
    setScreen(resolution.screen);
    return true;
  }

  console.warn("[OfficeNav] Unhandled command resolution:", target, resolution);
  return false;
}

function validateOfficeRouteMaps() {
  if (process.env.NODE_ENV === "production") return;

  FRANCHISE_COMMAND_REGISTRY.forEach((cmd) => {
    if (cmd.type === "navigate" && !resolveCommandTarget(cmd.target)) {
      console.warn("[OfficeNav] Registry command missing route:", cmd.id, cmd.target);
    }
  });

  Object.values(OFFICE_NAV_TARGETS).forEach((target) => {
    if (
      target.startsWith("sim-") ||
      target === OFFICE_NAV_TARGETS.SIM_GAME ||
      target === OFFICE_NAV_TARGETS.SIM_NEXT_GAME ||
      target === OFFICE_NAV_TARGETS.SIM_TO_DATE
    ) {
      return;
    }
    if (!resolveCommandTarget(target)) {
      console.warn("[OfficeNav] Missing route for office target:", target);
    }
  });
}

function highlightOfficePanel(panelId, setHubMenuIndex) {
  const screen = OFFICE_PANEL_TO_SCREEN[panelId];
  if (!screen) {
    console.warn("[OfficeNav] Unknown office panel:", panelId);
    return false;
  }
  syncHubIndexForScreen(screen, setHubMenuIndex);
  return true;
}

function safeNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function fmtMoney(value) {
  if (value == null || value === "—") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);

  if (Math.abs(n) >= 1000000) {
    return `$${(n / 1000000).toFixed(2)}M`;
  }

  return `$${n.toFixed(2)}M`;
}

function fmtRecord(record) {
  if (!record) return null;
  if (typeof record === "string") return record;

  const w = record.w ?? record.wins;
  const l = record.l ?? record.losses;
  const otl = record.otl ?? record.ot ?? record.overtime_losses;
  if (w == null && l == null && otl == null) return null;

  return `${w ?? 0}-${l ?? 0}-${otl ?? 0}`;
}

function titleCase(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (m) => m.toUpperCase());
}

function phaseLabel(franchiseState) {
  const nhlToday = franchiseState?.nhl_today || {};
  const ph = franchiseState?.phase;

  if (ph === "complete") return "Season Complete";
  if (ph === "regular" && nhlToday?.ui_phase) return String(nhlToday.ui_phase);
  if (ph === "regular") return "Regular Season";
  if (!ph) return null;

  return titleCase(ph);
}

function getNextLeagueDay(franchiseState) {
  return (
    franchiseState?.calendar_summary ||
    franchiseState?.next_league_day ||
    franchiseState?.nhl_today?.date_label ||
    franchiseState?.nhl_today?.iso ||
    franchiseState?.current_date ||
    franchiseState?.date ||
    null
  );
}

function getCapSpace(team, franchiseState) {
  const snap =
    team?.cap_snapshot
    || franchiseState?.team?.cap_snapshot
    || franchiseState?.cap_snapshot
    || null;

  const seasonYear = safeNumber(
    franchiseState?.season_year ?? franchiseState?.season_calendar_year,
    2025
  );

  let limit = safeNumber(
    snap?.upper_limit_m
      ?? team?.cap_limit
      ?? team?.salary_cap
      ?? franchiseState?.salary_cap,
    NaN
  );
  const hit = safeNumber(
    snap?.total_cap_hit_m ?? team?.cap_hit ?? team?.payroll ?? franchiseState?.cap_hit,
    NaN
  );

  // Stale 2024–25 $88M ceiling on a 2025+ franchise — never trust usable space
  // that was computed under $88 (that early-return was the Pulse/briefing split).
  const stale88 =
    Number.isFinite(limit) && Math.abs(limit - 88) < 0.05 && seasonYear >= 2025;
  if (stale88) {
    limit = 95.5;
  }

  const snapSpace = safeNumber(
    snap?.usable_cap_space_m ?? snap?.usableCapSpace ?? snap?._raw?.usableCapSpace,
    NaN
  );
  if (Number.isFinite(snapSpace) && !stale88) {
    return snapSpace;
  }

  const explicit = safeNumber(
    team?.cap_space ?? team?.capSpace ?? franchiseState?.cap_space ?? franchiseState?.team_cap_space,
    NaN
  );

  if (Number.isFinite(limit) && Number.isFinite(hit)) {
    const recomputed = limit - hit;
    // If explicit space looks like it was computed under $88 (within ~0.1 of
    // 88-hit) while we corrected the ceiling, prefer the recomputed room.
    if (
      Number.isFinite(explicit)
      && seasonYear >= 2025
      && Math.abs(explicit - (88 - hit)) < 0.15
      && Math.abs(recomputed - explicit) > 0.2
    ) {
      return recomputed;
    }
    if (!Number.isFinite(explicit) || stale88) return recomputed;
  }

  if (Number.isFinite(explicit) && !stale88) return explicit;
  if (Number.isFinite(limit) && Number.isFinite(hit)) return limit - hit;
  return null;
}

function getStandingsLine(franchiseState, team) {
  const standings = franchiseState?.standings || [];
  const myId = String(team?.id ?? franchiseState?.user_team_id ?? "");
  const row = standings.find((r) => String(r.team_id ?? r.id) === myId);

  if (row?.division_rank && row?.division) {
    return `${row.division_rank} ${row.division}`;
  }

  if (row?.conference_rank && row?.conference) {
    return `${row.conference_rank} ${row.conference}`;
  }

  if (team?.standings_position) return team.standings_position;
  if (franchiseState?.standings_summary) return franchiseState.standings_summary;

  return null;
}

function findNextGame(franchiseState, team) {
  const blocks =
    franchiseState?.schedule_upcoming ||
    franchiseState?.upcoming_schedule ||
    franchiseState?.upcoming_games ||
    [];

  if (!Array.isArray(blocks) || !blocks.length) {
    return (
      franchiseState?.next_game ||
      franchiseState?.nextGame ||
      franchiseState?.nhl_today?.next_game ||
      null
    );
  }

  const firstBlock = blocks[0];
  const game = Array.isArray(firstBlock?.games) ? firstBlock.games[0] : firstBlock;

  if (!game) return null;

  const userId = String(team?.id ?? franchiseState?.user_team_id ?? "");
  const homeId = String(game?.home_id ?? game?.home_team_id ?? "");
  const awayId = String(game?.away_id ?? game?.away_team_id ?? "");

  const isHome = userId && homeId && userId === homeId;
  const isAway = userId && awayId && userId === awayId;

  if (isHome) {
    return `vs ${game?.away_name || game?.away_team || "Opponent"}`;
  }

  if (isAway) {
    return `@ ${game?.home_name || game?.home_team || "Opponent"}`;
  }

  if (game?.home_name && game?.away_name) {
    return `${game.away_name} @ ${game.home_name}`;
  }

  return null;
}

function countUnreadMessages(franchiseState) {
  return safeNumber(
    franchiseState?.unreadMessages ??
      franchiseState?.unread_messages ??
      franchiseState?.inbox_unread ??
      franchiseState?.messages_unread,
    0
  );
}

function countPendingTasks(franchiseState) {
  return safeNumber(
    franchiseState?.pendingTasks ??
      franchiseState?.pending_tasks ??
      franchiseState?.tasks_pending ??
      franchiseState?.urgent_decisions,
    0
  );
}

function countStorylines(franchiseState) {
  const direct =
    franchiseState?.activeStorylines ??
    franchiseState?.active_storylines ??
    franchiseState?.storyline_count;

  if (direct != null) return safeNumber(direct, 0);

  const events =
    franchiseState?.storyline_events ||
    franchiseState?.storylineEvents ||
    franchiseState?.storylines ||
    franchiseState?.league_storylines;
  if (Array.isArray(events)) return events.length;

  return 0;
}

function collectHubRosterPlayers(franchiseState) {
  const direct = franchiseState?.roster;
  if (Array.isArray(direct) && direct.length) return direct;

  const userTeamId = String(
    franchiseState?.user_team_id ||
      franchiseState?.team?.id ||
      franchiseState?.team?.team_id ||
      ""
  );

  const orgs = franchiseState?.roster_browser?.organizations || [];
  const userOrg =
    orgs.find((org) => String(org?.team_id || "") === userTeamId) ||
    orgs.find(
      (org) =>
        String(org?.name || "").toLowerCase() ===
        String(franchiseState?.team?.name || "").toLowerCase()
    ) ||
    orgs[0];

  if (Array.isArray(userOrg?.nhl) && userOrg.nhl.length) return userOrg.nhl;

  const teamPlayers = franchiseState?.team?.players || franchiseState?.user_team?.players;
  return Array.isArray(teamPlayers) ? teamPlayers : [];
}


export function HubScreen() {
  const {
    franchiseState,
    sessionBootstrapping,
    setScreen,
    setHubMenuIndex,
    setCapLedgerTab,
    setStatsCentralTab,
    openCommandPlaceholder,
    error,
    setError,
    onAdvanceDay,
    onAdvanceFranchise,
    advancing,
    openFranchiseEvent,
    onReopenOffseasonStage,
    refreshFranchise,
  } = useGameUI();

  useEffect(() => {
    validateOfficeRouteMaps();
  }, []);

  const team = franchiseState?.team || {};
  const teamName = team?.name || team?.team_name || "Franchise";
  const teamLogo = resolveFranchiseTeamLogo(team, teamName);
  const rec = fmtRecord(team?.record || franchiseState?.record);
  const capSpace = getCapSpace(team, franchiseState);
  const standingsLine = getStandingsLine(franchiseState, team);
  const currentPhase = phaseLabel(franchiseState);
  const currentDate = getNextLeagueDay(franchiseState);
  const nextGame = findNextGame(franchiseState, team);
  const seasonYear =
    franchiseState?.nhl_season_label ||
    franchiseState?.season_year ||
    franchiseState?.seasonYear ||
    currentPhase ||
    "Season";

  const canAdvance =
    franchiseState?.flags?.can_advance == null
      ? true
      : Boolean(franchiseState?.flags?.can_advance);

  const blockBulkSim =
    franchiseState?.flags?.is_terminal_dead_end ||
    String(franchiseState?.phase) === "complete";

  const canSimRegularSeason =
    !blockBulkSim &&
    !advancing &&
    ["regular", "preseason"].includes(
      String(franchiseState?.phase || franchiseState?.season_phase || "")
    );

  const phaseCta = getFranchisePhaseCta(franchiseState);

  const unreadMessages = countUnreadMessages(franchiseState);
  const pendingTasks = countPendingTasks(franchiseState);
  const activeStorylines = countStorylines(franchiseState);
  const rosterPlayers = useMemo(
    () => collectHubRosterPlayers(franchiseState),
    [franchiseState]
  );

  const simActions = useMemo(
    () => ({
      "sim-next-game": () =>
        onAdvanceFranchise?.({ mode: "next_game", count: 1, auto_resolve: true }),
      "sim-game": () =>
        onAdvanceFranchise?.({ mode: "next_game", count: 1, auto_resolve: true }),
      "sim-to-date": () =>
        onAdvanceFranchise?.({ mode: "days", count: 7, auto_resolve: true }),
    }),
    [onAdvanceFranchise]
  );

  const handlePhaseCta = useCallback(() => {
    const cta = String(phaseCta || "").toLowerCase();
    if (
      cta.includes("playoff") ||
      cta.includes("award") ||
      cta.includes("draft") ||
      cta.includes("bracket") ||
      cta.includes("lottery") ||
      cta.includes("combine") ||
      cta.includes("free agency") ||
      cta.includes("salary") ||
      cta.includes("development") ||
      cta.includes("roster") ||
      cta.includes("generate") ||
      cta.includes("preseason") ||
      cta.includes("retirement") ||
      cta.includes("continue to") ||
      cta.includes("resume") ||
      cta.includes("rights") ||
      cta.includes("re-sign") ||
      cta.includes("resign") ||
      cta.includes("contract") ||
      cta.includes("timeline") ||
      cta.includes("offseason")
    ) {
      openFranchiseEvent?.();
      return;
    }
    onAdvanceDay?.();
  }, [phaseCta, openFranchiseEvent, onAdvanceDay]);

  const handleOpenPanel = useCallback(
    (panelId) => {
      highlightOfficePanel(panelId, setHubMenuIndex);
    },
    [setHubMenuIndex]
  );

  const handleNavigate = useCallback(
    (target) => {
      if (simActions[target] && !advancing && canAdvance) {
        simActions[target]();
        return;
      }

      navigateFranchiseCommand(target, {
        setScreen,
        setHubMenuIndex,
        setCapLedgerTab,
        setStatsCentralTab,
        openCommandPlaceholder,
        openFranchiseEvent,
        onReopenOffseasonStage,
      });
    },
    [
      advancing,
      canAdvance,
      openCommandPlaceholder,
      openFranchiseEvent,
      onReopenOffseasonStage,
      setCapLedgerTab,
      setHubMenuIndex,
      setScreen,
      setStatsCentralTab,
      simActions,
    ]
  );

  if (sessionBootstrapping && !franchiseState?.team?.name && !franchiseState?.team?.team_name) {
    return (
      <div className="game-screen hub-screen">
        <div
          style={{
            minHeight: "100%",
            display: "grid",
            placeItems: "center",
            color: "rgba(201,168,106,0.85)",
            fontFamily: 'var(--font-office-display, "Archivo Black", sans-serif)',
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            fontSize: 12,
            fontWeight: 800,
          }}
        >
          Restoring franchise…
        </div>
      </div>
    );
  }

  return (
    <div className="game-screen hub-screen">
      <FirstPersonOfficeHub
        teamName={teamName}
        teamLogo={teamLogo}
        seasonYear={seasonYear}
        currentDate={currentDate || "Today"}
        record={rec || "0-0-0"}
        capSpace={fmtMoney(capSpace)}
        capSpaceMillions={Number.isFinite(capSpace) ? capSpace : null}
        nextGame={nextGame || "No game listed"}
        standingsRank={standingsLine || "Standings"}
        unreadMessages={unreadMessages}
        pendingTasks={pendingTasks}
        activeStorylines={activeStorylines}
        franchiseState={franchiseState}
        team={team}
        players={rosterPlayers}
        onOpenPanel={handleOpenPanel}
        onNavigate={handleNavigate}
        onSimNextGame={() => handleNavigate("sim-next-game")}
        simDisabled={!canAdvance || advancing}
      />

      {error ? (
        <div
          style={{
            position: "absolute",
            left: 18,
            top: 190,
            zIndex: 20,
            maxWidth: 420,
            padding: "12px 14px",
            borderRadius: 14,
            background: "rgba(120, 0, 20, 0.86)",
            color: "white",
            fontWeight: 800,
            boxShadow: "0 18px 45px rgba(0,0,0,0.35)",
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          <div>{error}</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button
              type="button"
              onClick={() => refreshFranchise?.()}
              style={{
                border: "1px solid rgba(255,255,255,0.35)",
                background: "rgba(255,255,255,0.12)",
                color: "white",
                fontWeight: 800,
                borderRadius: 10,
                padding: "6px 10px",
                cursor: "pointer",
              }}
            >
              Retry connection
            </button>
            <button
              type="button"
              onClick={() => setError?.(null)}
              style={{
                border: "1px solid rgba(255,255,255,0.35)",
                background: "transparent",
                color: "white",
                fontWeight: 800,
                borderRadius: 10,
                padding: "6px 10px",
                cursor: "pointer",
              }}
            >
              Dismiss
            </button>
          </div>
        </div>
      ) : null}

      {/* Desk console: an engraved instrument cluster rather than a row of
          web pills. The clock advance is the primary control; bulk simulation
          reads as one segmented dial so it cannot wrap onto a second row. */}
      <div
        style={{
          position: "absolute",
          right: 22,
          bottom: 54,
          zIndex: 5,
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-end",
          gap: 6,
          pointerEvents: "auto",
          maxWidth: "min(96vw, 720px)",
        }}
      >
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
          <button
            type="button"
            disabled={!canAdvance || advancing}
            onClick={onAdvanceDay}
            style={{
              minHeight: 34,
              padding: "0 16px",
              borderRadius: 3,
              border: "1px solid rgba(201,168,106,0.55)",
              background: "rgba(201,168,106,0.14)",
              color: "#f4e3bd",
              fontWeight: 900,
              fontSize: 12,
              letterSpacing: "0.1em",
              cursor: advancing ? "not-allowed" : "pointer",
            }}
          >
            {advancing ? "ADVANCING..." : "Advance one day"}
          </button>

          {(franchiseState?.flags?.can_enter_playoffs || phaseCta) && phaseCta !== "Advance Day" ? (
            <button
              type="button"
              disabled={advancing}
              onClick={handlePhaseCta}
              style={{
                minHeight: 34,
                padding: "0 14px",
                borderRadius: 3,
                border: "1px solid rgba(201,168,106,0.34)",
                background: "rgba(10,12,16,0.86)",
                color: "#e7d7b4",
                fontWeight: 900,
                fontSize: 12,
                letterSpacing: "0.1em",
                cursor: advancing ? "not-allowed" : "pointer",
              }}
            >
              {(phaseCta || "ENTER PLAYOFFS").toUpperCase()}
            </button>
          ) : null}

          <button
            type="button"
            onClick={() => setScreen(ROUTE_SCREENS.CHEMISTRY)}
            style={{
              minHeight: 34,
              padding: "0 14px",
              borderRadius: 3,
              border: "1px solid rgba(201,168,106,0.24)",
              background: "rgba(10,12,16,0.86)",
              color: "#cbbfa6",
              fontWeight: 900,
              fontSize: 12,
              letterSpacing: "0.1em",
            }}
          >
            Team chemistry
          </button>
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "stretch",
            border: "1px solid rgba(201,168,106,0.24)",
            borderRadius: 3,
            background: "rgba(10,12,16,0.86)",
            overflow: "hidden",
          }}
        >
          <span
            style={{
              display: "flex",
              alignItems: "center",
              padding: "0 10px",
              color: "rgba(201,168,106,0.75)",
              fontSize: 11,
              fontWeight: 900,
              letterSpacing: "0.16em",
            }}
          >
            SIM DAYS
          </span>

          <button
            type="button"
            disabled={!canSimRegularSeason}
            onClick={() =>
              onAdvanceFranchise?.({ mode: "season", count: 1, auto_resolve: true })
            }
            title="Simulate the rest of the regular season (testing)"
            style={{
              minHeight: 32,
              padding: "0 12px",
              border: 0,
              borderLeft: "1px solid rgba(201,168,106,0.18)",
              background: "transparent",
              color: "#e7d7b4",
              fontWeight: 900,
              fontSize: 11,
              letterSpacing: "0.08em",
              cursor: canSimRegularSeason ? "pointer" : "not-allowed",
              opacity: canSimRegularSeason ? 1 : 0.4,
            }}
          >
            {advancing ? "SIMMING..." : "Rest of season"}
          </button>

          {[7, 15, 30].map((days) => (
            <button
              key={days}
              type="button"
              disabled={blockBulkSim || advancing}
              onClick={() =>
                onAdvanceFranchise?.({ mode: "days", count: days, auto_resolve: true })
              }
              style={{
                minHeight: 32,
                padding: "0 12px",
                border: 0,
                borderLeft: "1px solid rgba(201,168,106,0.18)",
                background: "transparent",
                color: "#cbbfa6",
                fontWeight: 900,
                fontSize: 11,
                letterSpacing: "0.08em",
                fontVariantNumeric: "tabular-nums",
                cursor: advancing ? "not-allowed" : "pointer",
                opacity: blockBulkSim || advancing ? 0.4 : 1,
              }}
            >
              {days} days
            </button>
          ))}
        </div>
      </div>

      <GameFooter />
    </div>
  );
}
