import React, { useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { getApiBaseUrl } from "../services/api";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";

const LOTTERY_ODDS = [
  18.5, 13.5, 11.5, 9.5, 8.5, 7.5, 6.5, 6.0,
  5.0, 3.5, 3.0, 2.5, 2.0, 1.5, 0.5, 0.5,
];

function safeNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function getTeamName(team) {
  return (
    team?.team_name ||
    team?.teamName ||
    team?.name ||
    team?.full_name ||
    team?.fullName ||
    team?.franchise_name ||
    team?.franchiseName ||
    "Unknown Team"
  );
}

function getTeamAbbreviation(team) {
  return (
    team?.abbreviation ||
    team?.abbr ||
    team?.team_abbr ||
    team?.teamAbbr ||
    team?.code ||
    getTeamName(team).slice(0, 3).toUpperCase()
  );
}

function getTeamLogo(team) {
  return (
    team?.logo ||
    team?.logo_url ||
    team?.logoUrl ||
    team?.team_logo ||
    team?.teamLogo ||
    team?.image ||
    null
  );
}

function getTeamId(team) {
  return (
    team?.id ||
    team?.team_id ||
    team?.teamId ||
    team?.abbreviation ||
    team?.abbr ||
    team?.name
  );
}

function getRecord(team) {
  const wins = safeNumber(team?.wins ?? team?.w);
  const losses = safeNumber(team?.losses ?? team?.l);
  const otLosses = safeNumber(
    team?.otLosses ??
      team?.otl ??
      team?.ot ??
      team?.overtime_losses ??
      team?.overtimeLosses
  );

  return { wins, losses, otLosses };
}

function getGamesPlayed(team) {
  const explicit =
    team?.gamesPlayed ??
    team?.gp ??
    team?.games_played ??
    team?.games ??
    team?.record?.gamesPlayed;

  if (explicit !== undefined && explicit !== null) {
    return safeNumber(explicit);
  }

  const { wins, losses, otLosses } = getRecord(team);
  return wins + losses + otLosses;
}

function getPoints(team) {
  const explicit = team?.points ?? team?.pts ?? team?.standings_points;

  if (explicit !== undefined && explicit !== null) {
    return safeNumber(explicit);
  }

  const { wins, otLosses } = getRecord(team);
  return wins * 2 + otLosses;
}

function getPointsPercentage(team) {
  const gp = getGamesPlayed(team);
  const points = getPoints(team);
  const maxPoints = gp * 2;

  if (!gp || !maxPoints) return 0;
  return points / maxPoints;
}

function normalizeStandings(raw) {
  if (!raw) return [];

  const possibleArrays = [
    raw,
    raw?.standings,
    raw?.leagueStandings,
    raw?.league_standings,
    raw?.teams,
    raw?.data,
    raw?.universe?.standings,
    raw?.universe?.leagueStandings,
    raw?.franchise?.standings,
    raw?.season?.standings,
    raw?.league?.standings,
  ];

  const source = possibleArrays.find(Array.isArray);
  if (!source) return [];

  return source
    .map((team) => {
      const { wins, losses, otLosses } = getRecord(team);
      const gp = getGamesPlayed(team);
      const points = getPoints(team);
      const pct = getPointsPercentage(team);

      return {
        raw: team,
        id: getTeamId(team),
        name: getTeamName(team),
        abbreviation: getTeamAbbreviation(team),
        logo: getTeamLogo(team) || resolveFranchiseTeamLogo(team, getTeamName(team)),
        wins,
        losses,
        otLosses,
        gamesPlayed: gp,
        points,
        pointsPercentage: pct,
        conference: team?.conference || team?.conf || "",
        division: team?.division || "",
        isPlayoffTeam:
          team?.isPlayoffTeam ||
          team?.playoff_team ||
          team?.clinched_playoffs ||
          team?.clinchedPlayoffs ||
          false,
      };
    })
    .filter((team) => team.name && team.name !== "Unknown Team");
}

function buildLotteryBoard(standings) {
  if (!standings.length) return [];

  const playoffTeamsKnown = standings.some((team) => team.isPlayoffTeam);

  const eligibleTeams = playoffTeamsKnown
    ? standings.filter((team) => !team.isPlayoffTeam)
    : [...standings]
        .sort((a, b) => {
          if (a.pointsPercentage !== b.pointsPercentage) {
            return a.pointsPercentage - b.pointsPercentage;
          }

          if (a.points !== b.points) {
            return a.points - b.points;
          }

          return a.wins - b.wins;
        })
        .slice(0, 16);

  return eligibleTeams
    .sort((a, b) => {
      if (a.pointsPercentage !== b.pointsPercentage) {
        return a.pointsPercentage - b.pointsPercentage;
      }

      if (a.points !== b.points) {
        return a.points - b.points;
      }

      return a.wins - b.wins;
    })
    .slice(0, 16)
    .map((team, index) => ({
      ...team,
      lotteryRank: index + 1,
      projectedPick: index + 1,
      odds: LOTTERY_ODDS[index] ?? 0,
    }));
}

function readLocalUniverse() {
  const possibleKeys = [
    "nhl_franchise_universe",
    "franchise_universe",
    "sim_universe",
    "universe",
    "franchiseState",
    "franchise_state",
    "currentFranchise",
    "current_franchise",
  ];

  for (const key of possibleKeys) {
    try {
      const value = localStorage.getItem(key);
      if (!value) continue;

      const parsed = JSON.parse(value);
      const standings = normalizeStandings(parsed);

      if (standings.length) {
        return parsed;
      }
    } catch {
      continue;
    }
  }

  return null;
}

async function fetchFirstWorkingEndpoint() {
  const endpoints = [
    "/api/franchise/state",
    "/api/franchise/universe",
    "/api/sim/universe",
    "/api/universe",
    "/api/standings",
    "/standings",
  ];

  for (const endpoint of endpoints) {
    try {
      const response = await fetch(`${getApiBaseUrl()}${endpoint}`);

      if (!response.ok) continue;

      const data = await response.json();
      const standings = normalizeStandings(data);

      if (standings.length) {
        return data;
      }
    } catch {
      continue;
    }
  }

  return null;
}

function weightedPick(teams, blockedIds = new Set()) {
  const available = teams.filter((team) => !blockedIds.has(String(team.id)));
  const totalWeight = available.reduce((sum, team) => sum + safeNumber(team.odds), 0);

  if (!available.length || totalWeight <= 0) return null;

  let roll = Math.random() * totalWeight;

  for (const team of available) {
    roll -= safeNumber(team.odds);

    if (roll <= 0) {
      return team;
    }
  }

  return available[available.length - 1];
}

function mapBackendLotteryToBoard(payload, standingsBoard) {
  if (!payload) return null;
  const final =
    payload.final_order ||
    payload.picks ||
    payload.order ||
    null;
  if (!Array.isArray(final) || !final.length) return null;

  const byId = new Map((standingsBoard || []).map((t) => [String(t.id), t]));
  return final.map((row, index) => {
    const tid = String(row.team_id || row.id || "");
    const viaId = row.via_team_id || row.original_owner_team_id || null;
    const viaName = row.via_team_name || row.original_owner_team_name || null;
    const isTraded = Boolean(row.is_traded) || Boolean(viaId && String(viaId) !== tid);
    const base = byId.get(tid) || {
      id: tid,
      name: row.team_name || tid,
      abbr: (row.team_name || tid || "").slice(0, 3).toUpperCase(),
      logo: null,
      points: row.points ?? 0,
      wins: row.wins ?? 0,
      losses: 0,
      otl: 0,
      pointsPercentage: 0,
      odds: LOTTERY_ODDS[Math.max(0, (row.original_rank || index + 1) - 1)] ?? 0,
      lotteryRank: row.original_rank || index + 1,
      projectedPick: row.original_rank || index + 1,
    };
    const wonPick =
      row.won_pick ??
      (row.pick <= 2 && (row.movement || 0) > 0 ? row.pick : null);
    return {
      ...base,
      id: tid,
      name: row.team_name || base.name,
      finalPick: row.pick || index + 1,
      wonPick: wonPick || null,
      movement: row.movement ?? 0,
      lotterySeed: payload.lottery_seed,
      isTraded,
      viaTeamId: isTraded ? viaId : null,
      viaTeamName: isTraded ? viaName : null,
      viaAbbr: isTraded
        ? String(viaName || viaId || "")
            .replace(/^the\s+/i, "")
            .slice(0, 3)
            .toUpperCase()
        : null,
      lotteryTeamId: row.lottery_team_id || row.original_owner_team_id || tid,
    };
  });
}

function simulateLottery(board) {
  if (!board.length) return [];
  return board.map((team, index) => ({
    ...team,
    wonPick: null,
    finalPick: index + 1,
  }));
}

export default function DraftLottery({
  universe,
  standings,
  leagueStandings,
  selectedTeam,
  userTeam,
}) {
  const { franchiseState, setScreen } = useGameUI();
  const [simData, setSimData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [simResults, setSimResults] = useState(null);

  const franchisePayload = franchiseState || null;

  useEffect(() => {
    let active = true;

    async function loadSimUniverse() {
      setLoading(true);

      const propData =
        universe ||
        standings ||
        leagueStandings ||
        franchisePayload ||
        null;

      if (propData && normalizeStandings(propData).length) {
        if (active) {
          setSimData(propData);
          setLoading(false);
        }
        return;
      }

      const localData = readLocalUniverse();

      if (localData) {
        if (active) {
          setSimData(localData);
          setLoading(false);
        }
        return;
      }

      const apiData = await fetchFirstWorkingEndpoint();

      if (active) {
        setSimData(apiData || franchisePayload);
        setLoading(false);
      }
    }

    loadSimUniverse();

    return () => {
      active = false;
    };
  }, [universe, standings, leagueStandings, franchisePayload]);

  const normalizedStandings = useMemo(() => {
    return normalizeStandings(
      simData ||
        franchisePayload ||
        universe ||
        standings ||
        leagueStandings
    );
  }, [simData, franchisePayload, universe, standings, leagueStandings]);

  const lotteryBoard = useMemo(() => {
    return buildLotteryBoard(normalizedStandings);
  }, [normalizedStandings]);

  const backendLottery = useMemo(() => {
    const payload =
      franchisePayload?.draft_lottery ||
      franchisePayload?.offseason?.draft_lottery ||
      simData?.draft_lottery ||
      null;
    return mapBackendLotteryToBoard(payload, lotteryBoard);
  }, [franchisePayload, simData, lotteryBoard]);

  const displayedBoard = simResults || backendLottery || lotteryBoard;
  const hasOfficialResult = Boolean(backendLottery?.length || simResults?.length);
  const showingResults = Boolean(simResults?.length);

  useEffect(() => {
    if (backendLottery?.length) {
      setSimResults(backendLottery);
    }
  }, [backendLottery]);

  const selectedTeamId =
    selectedTeam?.id ||
    selectedTeam?.team_id ||
    selectedTeam?.teamId ||
    selectedTeam?.abbreviation ||
    selectedTeam?.abbr ||
    userTeam?.id ||
    userTeam?.team_id ||
    userTeam?.teamId ||
    userTeam?.abbreviation ||
    userTeam?.abbr ||
    franchisePayload?.user_team_id ||
    franchisePayload?.team?.id ||
    simData?.selectedTeamId ||
    simData?.selected_team_id ||
    simData?.userTeamId ||
    simData?.user_team_id ||
    null;

  const userTeamName = franchisePayload?.team?.name || userTeam?.name || "";

  function handleBack() {
    if (typeof setScreen === "function") {
      setScreen(SCREENS.DRAFT_CLASS || SCREENS.HUB);
    }
  }

  function handleSimLottery() {
    if (backendLottery?.length) {
      setSimResults(backendLottery);
      return;
    }
    setSimResults(simulateLottery(lotteryBoard));
  }

  function handleResetProjection() {
    setSimResults(null);
  }

  return (
    <main className="draft-lottery-page register-ops" data-register="ops">
      <DraftLotteryStyles />

      <header className="draft-lottery-topbar">
        <button type="button" className="draft-lottery-back" onClick={handleBack}>
          <span aria-hidden="true" className="draft-lottery-back__mark">←</span>
          <span>Back</span>
        </button>

        <div className="draft-lottery-title-block">
          <span className="type-phase-label">
            LEAGUE OPS · DRAFT LOTTERY
          </span>
          <h1>{showingResults ? "Official Order" : "Lottery Odds Board"}</h1>
          <p className="draft-lottery-sub">
            {showingResults
              ? "Post-draw pick order — not the live lottery broadcast."
              : "Pre-draw weighted odds from current standings."}
          </p>
        </div>

        <div className="draft-lottery-actions">
          {hasOfficialResult && simResults ? (
            <button type="button" className="draft-lottery-secondary" onClick={handleResetProjection}>
              Show Odds
            </button>
          ) : null}

          <button
            type="button"
            className="draft-lottery-primary"
            onClick={handleSimLottery}
            disabled={!lotteryBoard.length}
          >
            {backendLottery?.length ? "Show Official Results" : "Show Projection"}
          </button>
        </div>
      </header>

      {loading ? (
        <section className="ops-state ops-state--loading">
          <span className="ops-state__kicker">Standings sync</span>
          <h2 className="ops-state__title">Loading lottery board</h2>
          <p className="ops-state__body">Pulling eligible teams from franchise standings.</p>
        </section>
      ) : !lotteryBoard.length ? (
        <section className="ops-state">
          <span className="ops-state__kicker">No data</span>
          <h2 className="ops-state__title">Lottery board unavailable</h2>
          <p className="ops-state__body">
            Standings were not found from props, local storage, or the backend. Advance the season or load franchise state.
          </p>
        </section>
      ) : (
        <section className="draft-lottery-board" aria-label="Draft lottery odds">
          <div className="draft-lottery-table-head">
            <div>Slot</div>
            <div>Team</div>
            <div>Record</div>
            <div>PTS</div>
            <div>PTS%</div>
            <div>Win %</div>
            {showingResults ? <div>Result</div> : null}
          </div>

          <div className="draft-lottery-rows">
            {displayedBoard.map((team) => {
              const isUserTeam =
                (selectedTeamId &&
                  (String(team.id).toLowerCase() === String(selectedTeamId).toLowerCase() ||
                    String(team.lotteryTeamId || "").toLowerCase() ===
                      String(selectedTeamId).toLowerCase())) ||
                (userTeamName &&
                  String(team.name).toLowerCase() ===
                    String(userTeamName).toLowerCase());

              return (
                <article
                  key={`${team.id}-${team.lotteryRank}-${team.finalPick || team.projectedPick}`}
                  className={[
                    "draft-lottery-row",
                    isUserTeam ? "is-user-team" : "",
                    team.wonPick ? "is-winner" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                >
                  <div className="draft-lottery-pick-cell">
                    <span className="draft-lottery-slot">
                      {showingResults ? team.finalPick : team.projectedPick}
                    </span>
                    {!showingResults ? (
                      <span className="draft-lottery-odds">{team.odds}%</span>
                    ) : null}
                  </div>

                  <div className="draft-lottery-team-cell">
                    {team.logo ? (
                      <img src={team.logo} alt="" />
                    ) : (
                      <div className="draft-lottery-logo-fallback">
                        {team.abbreviation}
                      </div>
                    )}

                    <div>
                      <strong>{team.name}</strong>
                      <span>
                        {team.isTraded && team.viaAbbr
                          ? `via ${team.viaAbbr}`
                          : team.abbreviation}
                      </span>
                    </div>
                  </div>

                  <div className="draft-lottery-record type-financial">
                    {team.wins}-{team.losses}-{team.otLosses}
                  </div>

                  <div className="type-financial">{team.points}</div>

                  <div className="type-financial">{(team.pointsPercentage * 100).toFixed(1)}</div>

                  <div className="draft-lottery-odds-cell type-financial">
                    {!showingResults ? (
                      <strong>{team.odds}%</strong>
                    ) : (
                      <span className="draft-lottery-odds-muted">{team.odds}%</span>
                    )}
                    <span
                      className="draft-lottery-odds-ruler"
                      aria-hidden="true"
                      style={{
                        "--odds-share": `${Math.max(
                          0,
                          Math.min(100, (Number(team.odds) || 0) * 4)
                        )}%`,
                      }}
                    />
                  </div>

                  {showingResults ? (
                    <div>
                      {team.wonPick ? (
                        <span className="mark-seal mark-status--success">Won #{team.wonPick}</span>
                      ) : (
                        <span className="mark-status mark-status--info">Held</span>
                      )}
                    </div>
                  ) : null}
                </article>
              );
            })}
          </div>
        </section>
      )}
    </main>
  );
}

function DraftLotteryStyles() {
  return (
    <style>{`
      .draft-lottery-page {
        min-height: 100vh;
        width: 100%;
        padding: var(--space-4) var(--space-5);
        color: var(--ops-text);
        font-family: var(--font-ops-ui);
        background:
          radial-gradient(circle at 14% 0%, rgba(19, 216, 231, 0.1), transparent 28%),
          radial-gradient(circle at 88% 8%, rgba(233, 168, 60, 0.06), transparent 22%),
          linear-gradient(180deg, var(--ops-navy) 0%, var(--ops-navy-deep) 100%);
      }

      .draft-lottery-page *,
      .draft-lottery-page *::before,
      .draft-lottery-page *::after {
        box-sizing: border-box;
      }

      .draft-lottery-topbar {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr) auto;
        align-items: center;
        gap: var(--space-4);
        margin-bottom: var(--space-4);
        padding: var(--space-3) var(--space-4);
        border: 1px solid var(--ops-grid);
        border-radius: var(--radius-card);
        background: var(--ops-panel);
        box-shadow: var(--depth-overlay);
      }

      .draft-lottery-title-block {
        min-width: 0;
      }

      .draft-lottery-title-block .type-phase-label {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 4px;
        color: var(--ops-cyan);
      }

      .draft-lottery-title-block h1 {
        margin: 0;
        font-family: var(--font-broadcast-display);
        font-size: clamp(1.25rem, 2.4vw, 1.75rem);
        line-height: 1.05;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--ops-text);
      }

      .draft-lottery-sub {
        margin: 6px 0 0;
        font-size: var(--type-compact-size);
        color: var(--ops-text-secondary);
        line-height: 1.4;
      }

      .draft-lottery-actions {
        display: flex;
        justify-content: flex-end;
        gap: var(--space-2);
        flex-wrap: wrap;
      }

      .draft-lottery-back,
      .draft-lottery-primary,
      .draft-lottery-secondary {
        min-height: 38px;
        padding: 0 var(--space-4);
        border-radius: var(--radius-control);
        font-size: var(--type-control-label-size);
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        cursor: pointer;
        transition: background var(--motion-micro), border-color var(--motion-micro);
        display: inline-flex;
        align-items: center;
        gap: 6px;
      }

      .draft-lottery-back {
        color: var(--ops-text);
        background: rgba(12, 31, 47, 0.72);
        border: 1px solid var(--ops-grid);
      }

      .draft-lottery-primary {
        color: var(--ops-navy);
        background: var(--ops-cyan);
        border: 1px solid var(--ops-cyan);
      }

      .draft-lottery-secondary {
        color: var(--ops-text);
        background: rgba(12, 31, 47, 0.72);
        border: 1px solid var(--ops-grid-2);
      }

      .draft-lottery-back:hover,
      .draft-lottery-secondary:hover {
        border-color: var(--ops-grid-strong);
        background: var(--ops-cyan-soft);
      }

      .draft-lottery-primary:hover {
        background: var(--ops-ice);
      }

      .draft-lottery-primary:disabled {
        cursor: not-allowed;
        opacity: 0.45;
      }

      .draft-lottery-board {
        border: 1px solid var(--ops-grid);
        border-radius: var(--radius-card);
        background: var(--ops-panel);
        overflow: hidden;
      }

      .draft-lottery-table-head {
        display: grid;
        grid-template-columns: 88px minmax(220px, 1.4fr) 100px 64px 72px 72px 96px;
        gap: var(--space-2);
        padding: var(--space-2) var(--space-4);
        border-bottom: 1px solid var(--ops-grid);
        background: rgba(6, 21, 34, 0.55);
        color: var(--ops-text-secondary);
        font-size: var(--type-phase-label-size);
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }

      .draft-lottery-table-head:not(:has(div:nth-child(7))) {
        grid-template-columns: 88px minmax(220px, 1.4fr) 100px 64px 72px 72px;
      }

      .draft-lottery-rows {
        display: flex;
        flex-direction: column;
      }

      .draft-lottery-row {
        display: grid;
        grid-template-columns: 88px minmax(220px, 1.4fr) 100px 64px 72px 72px 96px;
        align-items: center;
        gap: var(--space-2);
        min-height: 52px;
        padding: var(--space-2) var(--space-4);
        border-bottom: 1px solid rgba(156, 218, 236, 0.08);
        color: var(--ops-text);
        background: transparent;
      }

      .draft-lottery-row:not(:has(.mark-seal)):not(:has(.mark-status)) {
        grid-template-columns: 88px minmax(220px, 1.4fr) 100px 64px 72px 72px;
      }

      .draft-lottery-row:last-child {
        border-bottom: 0;
      }

      .draft-lottery-row.is-user-team {
        background: var(--ops-cyan-soft);
      }

      .draft-lottery-row.is-winner {
        background: var(--ops-success-soft);
      }

      .draft-lottery-pick-cell {
        display: flex;
        flex-direction: column;
        gap: 2px;
      }

      .draft-lottery-slot {
        font-family: var(--font-broadcast-display);
        font-size: var(--type-score-size);
        font-weight: 400;
        color: var(--ops-gold);
        line-height: 1;
      }

      .draft-lottery-odds {
        font-size: var(--type-table-meta-size);
        font-weight: 800;
        color: var(--ops-cyan);
        letter-spacing: 0.04em;
      }

      .draft-lottery-odds-muted {
        font-size: var(--type-table-meta-size);
        color: var(--ops-text-secondary);
      }

      /* Department signature: the odds ruler. Each club's real lottery weight
         is measured against a shared scale instead of a casino graphic. */
      .draft-lottery-odds-ruler {
        display: block;
        position: relative;
        height: 4px;
        margin-top: 4px;
        background: rgba(255, 255, 255, 0.06);
        background-image: repeating-linear-gradient(
          90deg,
          rgba(255, 255, 255, 0.16) 0 1px,
          transparent 1px 25%
        );
      }

      .draft-lottery-odds-ruler::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: var(--odds-share, 0%);
        background: var(--ops-cyan);
        opacity: 0.7;
      }

      .draft-lottery-team-cell {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        min-width: 0;
      }

      .draft-lottery-team-cell img,
      .draft-lottery-logo-fallback {
        width: 32px;
        height: 32px;
        flex: 0 0 32px;
        border-radius: var(--radius-ops);
        object-fit: contain;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--ops-grid);
      }

      .draft-lottery-logo-fallback {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.6875rem;
        font-weight: 900;
        color: var(--ops-cyan);
      }

      .draft-lottery-team-cell strong {
        display: block;
        overflow: hidden;
        font-size: var(--type-compact-size);
        font-weight: 700;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .draft-lottery-team-cell span {
        display: block;
        margin-top: 1px;
        color: var(--ops-text-secondary);
        font-size: var(--type-table-meta-size);
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
      }

      .draft-lottery-record {
        font-weight: 700;
      }

      .draft-lottery-odds-cell strong {
        color: var(--ops-cyan);
        font-size: var(--type-compact-size);
      }

      @media (max-width: 1050px) {
        .draft-lottery-topbar {
          grid-template-columns: 1fr;
        }

        .draft-lottery-actions {
          justify-content: flex-start;
        }

        .draft-lottery-board {
          overflow-x: auto;
        }

        .draft-lottery-table-head,
        .draft-lottery-row,
        .draft-lottery-table-head:not(:has(div:nth-child(7))),
        .draft-lottery-row:not(:has(.mark-seal)):not(:has(.mark-status)) {
          min-width: 780px;
        }
      }

      @media (max-width: 640px) {
        .draft-lottery-page {
          padding: var(--space-3);
        }
      }
    `}</style>
  );
}
