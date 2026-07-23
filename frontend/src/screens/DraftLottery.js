import React, { useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";

const LOTTERY_ODDS = [
  18.5, 13.5, 11.5, 9.5, 8.5, 7.5, 6.5, 6.0,
  5.0, 3.5, 3.0, 2.5, 2.0, 1.5, 0.5, 0.5,
];

const API_BASE =
  process.env.REACT_APP_API_BASE_URL ||
  process.env.REACT_APP_API_URL ||
  "http://localhost:8000";

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
      const response = await fetch(`${API_BASE}${endpoint}`);

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
    };
  });
}

function simulateLottery(board) {
  // Local simulation is disabled — backend lottery is the authority.
  // Kept as a no-op fallback that returns the projected order unchanged.
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

  // When the backend lottery exists, surface it automatically (no local rerolls).
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
    // Prefer authoritative backend result; never locally re-roll a completed lottery.
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
    <main className="draft-lottery-page">
      <DraftLotteryStyles />

      <section className="draft-lottery-topbar">
        <button type="button" className="draft-lottery-back" onClick={handleBack}>
          Back
        </button>

        <div className="draft-lottery-title-block">
          <span>Draft Lottery</span>
          <h1>Lottery Odds</h1>
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
      </section>

      {loading ? (
        <section className="draft-lottery-empty">
          <h2>Loading standings...</h2>
        </section>
      ) : !lotteryBoard.length ? (
        <section className="draft-lottery-empty">
          <h2>No lottery teams found</h2>
          <p>
            This screen could not find standings from props, local storage, or the backend.
          </p>
        </section>
      ) : (
        <section className="draft-lottery-board">
          <div className="draft-lottery-table-head">
            <div>Pick</div>
            <div>Team</div>
            <div>Record</div>
            <div>PTS</div>
            <div>PTS%</div>
            <div>Odds</div>
            {hasOfficialResult ? <div>Result</div> : null}
          </div>

          <div className="draft-lottery-rows">
            {displayedBoard.map((team) => {
              const isUserTeam =
                (selectedTeamId &&
                  String(team.id).toLowerCase() ===
                    String(selectedTeamId).toLowerCase()) ||
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
                    #{simResults ? team.finalPick : team.projectedPick}
                  </div>

                  <div className="draft-lottery-team-cell">
                    {team.logo ? (
                      <img src={team.logo} alt={team.name} />
                    ) : (
                      <div className="draft-lottery-logo-fallback">
                        {team.abbreviation}
                      </div>
                    )}

                    <div>
                      <strong>{team.name}</strong>
                      <span>{team.abbreviation}</span>
                    </div>
                  </div>

                  <div className="draft-lottery-record">
                    {team.wins}-{team.losses}-{team.otLosses}
                  </div>

                  <div>{team.points}</div>

                  <div>{(team.pointsPercentage * 100).toFixed(1)}%</div>

                  <div>
                    <strong className="draft-lottery-odds">{team.odds}%</strong>
                  </div>

                  {simResults ? (
                    <div>
                      {team.wonPick ? (
                        <span className="draft-lottery-winner">
                          Won #{team.wonPick}
                        </span>
                      ) : (
                        <span className="draft-lottery-hold">Projected</span>
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
        padding: 22px;
        color: #eaf2ff;
        font-family: var(--font-motion-control, "Rajdhani", "Barlow Condensed", Arial, sans-serif);
        background:
          radial-gradient(circle at top left, rgba(59, 130, 246, 0.2), transparent 34%),
          radial-gradient(circle at bottom right, rgba(16, 185, 129, 0.1), transparent 28%),
          linear-gradient(135deg, #07111f 0%, #0d1727 44%, #111827 100%);
      }

      .draft-lottery-page *,
      .draft-lottery-page *::before,
      .draft-lottery-page *::after {
        box-sizing: border-box;
      }

      .draft-lottery-topbar {
        position: sticky;
        top: 0;
        z-index: 10;
        display: grid;
        grid-template-columns: 120px 1fr auto;
        align-items: center;
        gap: 16px;
        margin-bottom: 16px;
        padding: 14px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.84);
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(18px);
      }

      .draft-lottery-title-block {
        text-align: center;
      }

      .draft-lottery-title-block span {
        display: block;
        margin-bottom: 2px;
        color: #93c5fd;
        font-size: 0.75rem;
        font-weight: 900;
        letter-spacing: 0.16em;
        text-transform: uppercase;
      }

      .draft-lottery-title-block h1 {
        margin: 0;
        color: #ffffff;
        font-size: clamp(1.7rem, 4vw, 3rem);
        line-height: 0.95;
        letter-spacing: -0.04em;
        text-transform: uppercase;
      }

      .draft-lottery-actions {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
      }

      .draft-lottery-back,
      .draft-lottery-primary,
      .draft-lottery-secondary {
        min-height: 44px;
        padding: 0 18px;
        border: 0;
        border-radius: 15px;
        font-weight: 950;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        cursor: pointer;
        transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease;
      }

      .draft-lottery-back {
        color: #cbd5e1;
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.18);
      }

      .draft-lottery-primary {
        color: #04111f;
        background: linear-gradient(135deg, #bfdbfe, #60a5fa);
        box-shadow: 0 12px 24px rgba(96, 165, 250, 0.2);
      }

      .draft-lottery-secondary {
        color: #dbeafe;
        background: rgba(59, 130, 246, 0.18);
        border: 1px solid rgba(96, 165, 250, 0.28);
      }

      .draft-lottery-back:hover,
      .draft-lottery-primary:hover,
      .draft-lottery-secondary:hover {
        transform: translateY(-2px);
      }

      .draft-lottery-primary:disabled {
        cursor: not-allowed;
        opacity: 0.5;
        transform: none;
      }

      .draft-lottery-board {
        min-height: calc(100vh - 120px);
        padding: 14px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.72);
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(18px);
      }

      .draft-lottery-table-head {
        display: grid;
        grid-template-columns: 90px minmax(260px, 1fr) 130px 90px 100px 110px 120px;
        gap: 10px;
        padding: 0 14px 10px;
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 950;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }

      .draft-lottery-table-head:not(:has(div:nth-child(7))) {
        grid-template-columns: 90px minmax(260px, 1fr) 130px 90px 100px 110px;
      }

      .draft-lottery-rows {
        display: flex;
        flex-direction: column;
        gap: 10px;
      }

      .draft-lottery-row {
        display: grid;
        grid-template-columns: 90px minmax(260px, 1fr) 130px 90px 100px 110px 120px;
        align-items: center;
        gap: 10px;
        min-height: 68px;
        padding: 10px 14px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        color: #dbeafe;
        background:
          linear-gradient(135deg, rgba(2, 6, 23, 0.6), rgba(15, 23, 42, 0.74));
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
      }

      .draft-lottery-row:not(:has(.draft-lottery-winner)):not(:has(.draft-lottery-hold)) {
        grid-template-columns: 90px minmax(260px, 1fr) 130px 90px 100px 110px;
      }

      .draft-lottery-row.is-user-team {
        border-color: rgba(96, 165, 250, 0.56);
        background:
          linear-gradient(135deg, rgba(37, 99, 235, 0.26), rgba(15, 23, 42, 0.78));
      }

      .draft-lottery-row.is-winner {
        border-color: rgba(34, 197, 94, 0.48);
        background:
          linear-gradient(135deg, rgba(22, 163, 74, 0.22), rgba(15, 23, 42, 0.78));
      }

      .draft-lottery-pick-cell {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 58px;
        height: 40px;
        border-radius: 999px;
        color: #ffffff;
        background: rgba(96, 165, 250, 0.18);
        border: 1px solid rgba(96, 165, 250, 0.32);
        font-size: 1.05rem;
        font-weight: 950;
      }

      .draft-lottery-team-cell {
        display: flex;
        align-items: center;
        gap: 14px;
        min-width: 0;
      }

      .draft-lottery-team-cell img,
      .draft-lottery-logo-fallback {
        width: 42px;
        height: 42px;
        flex: 0 0 42px;
        border-radius: 50%;
        object-fit: contain;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
      }

      .draft-lottery-logo-fallback {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #bfdbfe;
        font-size: 0.72rem;
        font-weight: 950;
      }

      .draft-lottery-team-cell strong {
        display: block;
        overflow: hidden;
        color: #ffffff;
        font-size: 1.04rem;
        font-weight: 900;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .draft-lottery-team-cell span {
        display: block;
        margin-top: 2px;
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 900;
      }

      .draft-lottery-record {
        color: #f8fafc;
        font-weight: 800;
      }

      .draft-lottery-odds {
        color: #bfdbfe;
        font-size: 1rem;
        font-weight: 950;
      }

      .draft-lottery-winner,
      .draft-lottery-hold {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 94px;
        min-height: 34px;
        padding: 0 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }

      .draft-lottery-winner {
        color: #dcfce7;
        background: rgba(34, 197, 94, 0.18);
        border: 1px solid rgba(34, 197, 94, 0.34);
      }

      .draft-lottery-hold {
        color: #cbd5e1;
        background: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.16);
      }

      .draft-lottery-empty {
        margin-top: 18px;
        padding: 28px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.72);
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.25);
      }

      .draft-lottery-empty h2 {
        margin: 0;
        color: #ffffff;
      }

      .draft-lottery-empty p {
        margin: 8px 0 0;
        color: #aebed4;
      }

      @media (max-width: 1050px) {
        .draft-lottery-topbar {
          grid-template-columns: 1fr;
        }

        .draft-lottery-title-block {
          text-align: left;
        }

        .draft-lottery-actions {
          justify-content: stretch;
        }

        .draft-lottery-actions button,
        .draft-lottery-back {
          flex: 1;
        }

        .draft-lottery-board {
          overflow-x: auto;
        }

        .draft-lottery-table-head,
        .draft-lottery-row,
        .draft-lottery-table-head:not(:has(div:nth-child(7))),
        .draft-lottery-row:not(:has(.draft-lottery-winner)):not(:has(.draft-lottery-hold)) {
          min-width: 820px;
        }
      }

      @media (max-width: 640px) {
        .draft-lottery-page {
          padding: 14px;
        }

        .draft-lottery-topbar {
          padding: 12px;
          border-radius: 18px;
        }

        .draft-lottery-actions {
          flex-direction: column;
        }
      }
    `}</style>
  );
}