import React, { useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";

/**
 * TeamNeeds.js
 *
 * Draft landing page that reads the user's roster, prospect pool, and draft class,
 * then diagnoses what the team should draft.
 *
 * Goals:
 * - No hardcoded real players.
 * - No backend changes required.
 * - Reads many possible franchiseState shapes safely.
 * - Uses the same premium dark/cyan/gold visual style as CalendarScreen.
 * - Gives the user an actual draft strategy, not just empty cards.
 */

const EMPTY_ARRAY = Object.freeze([]);
const EMPTY_OBJECT = Object.freeze({});

const POSITION_BUCKETS = {
  C: {
    label: "Centres",
    short: "C",
    minNhl: 4,
    minOrg: 8,
    idealAvg: 79,
    idealTop: 86,
    icon: "◎",
    draftLabel: "Draft a centre",
    archetype: "Two-way centre / play-driving middle-six pivot",
  },
  LW: {
    label: "Left Wings",
    short: "LW",
    minNhl: 4,
    minOrg: 7,
    idealAvg: 78,
    idealTop: 84,
    icon: "◐",
    draftLabel: "Draft a left wing",
    archetype: "Scoring winger / forechecking top-nine winger",
  },
  RW: {
    label: "Right Wings",
    short: "RW",
    minNhl: 4,
    minOrg: 7,
    idealAvg: 78,
    idealTop: 84,
    icon: "◑",
    draftLabel: "Draft a right wing",
    archetype: "Right-shot scorer / transition winger",
  },
  D: {
    label: "Defensemen",
    short: "D",
    minNhl: 6,
    minOrg: 11,
    idealAvg: 79,
    idealTop: 86,
    icon: "▣",
    draftLabel: "Draft a defenseman",
    archetype: "Top-four defenseman / mobile two-way defender",
  },
  G: {
    label: "Goalies",
    short: "G",
    minNhl: 2,
    minOrg: 4,
    idealAvg: 78,
    idealTop: 84,
    icon: "◉",
    draftLabel: "Draft a goalie",
    archetype: "High-upside goalie / long-term starter swing",
  },
};

const NEED_COLORS = {
  CRITICAL: "critical",
  HIGH: "high",
  MEDIUM: "medium",
  LOW: "low",
  STRENGTH: "strength",
};

function TeamNeeds(props = {}) {
  const gameUI = useGameUI();

  const {
    franchiseState = gameUI?.franchiseState,
    state = null,
    gameState = null,
    data = null,
    selectedTeamId,
    teamId,
    gmTeamId,
    setScreen = gameUI?.setScreen,
    navigate = null,
    onNavigate = null,
  } = props;

  const [selectedNeed, setSelectedNeed] = useState(null);

  const rootState = useMemo(() => {
    return franchiseState || state || gameState || data || EMPTY_OBJECT;
  }, [franchiseState, state, gameState, data]);

  const normalized = useMemo(() => {
    return normalizeFranchiseState(rootState, selectedTeamId || teamId || gmTeamId);
  }, [rootState, selectedTeamId, teamId, gmTeamId]);

  const report = useMemo(() => {
    return buildTeamNeedsReport(normalized);
  }, [normalized]);

  const activeNeed = selectedNeed
    ? report.positionReports.find((row) => row.position === selectedNeed) || report.positionReports[0]
    : report.positionReports[0];

  function handleNavigate(screenKey) {
    const resolved = screenKey || SCREENS.DRAFT_CLASS;

    if (typeof setScreen === "function") {
      setScreen(resolved);
      return;
    }

    if (typeof navigate === "function") {
      navigate(resolved);
      return;
    }

    if (typeof onNavigate === "function") {
      onNavigate(resolved);
    }
  }

  return (
    <div className="teamneeds-root">
      <TeamNeedsStyles />

      <aside className="teamneeds-sidebar">
        <button
          type="button"
          className="teamneeds-brand-button"
          onClick={() => handleNavigate(SCREENS.DRAFT_CLASS)}
          title="Draft Hub"
        >
          <span>◈</span>
        </button>

        <nav className="teamneeds-side-nav" aria-label="Draft navigation">
          <SideButton icon="⌂" label="Office" onClick={() => handleNavigate(SCREENS.OFFICE)} />
          <SideButton icon="◫" label="Calendar" onClick={() => handleNavigate(SCREENS.CALENDAR)} />
          <SideButton icon="▦" label="Roster" onClick={() => handleNavigate(SCREENS.ROSTER)} />
          <SideButton active icon="◈" label="Needs" onClick={() => handleNavigate(SCREENS.TEAM_NEEDS)} />
          <SideButton icon="★" label="Scouting" onClick={() => handleNavigate(SCREENS.STATS)} />
          <SideButton icon="◎" label="Lottery" onClick={() => handleNavigate(SCREENS.DRAFT_LOTTERY)} />
        </nav>
      </aside>

      <main className="teamneeds-main">
        <header className="teamneeds-topbar">
          <section className="teamneeds-team-identity">
            <TeamLogo team={normalized.activeTeam} />
            <div>
              <p>{getTeamCity(normalized.activeTeam)}</p>
              <h1>Team Needs</h1>
              <span>{getTeamDisplayName(normalized.activeTeam)} draft intelligence</span>
            </div>
          </section>

          <section className="teamneeds-draft-board-title">
            <p>Draft Room</p>
            <h2>Roster Inefficiency Report</h2>
          </section>

          <section className="teamneeds-actions">
            <button type="button" onClick={() => handleNavigate(SCREENS.ROSTER)}>
              View Roster
            </button>
            <button type="button" onClick={() => handleNavigate(SCREENS.DRAFT_CLASS)}>
              Draft Board
            </button>
          </section>
        </header>

        <section className="teamneeds-stat-strip">
          <StatPill
            icon="▦"
            label="Roster Loaded"
            value={report.totalRoster}
            sub={`${report.nhlRosterCount} NHL / ${report.prospectCount} prospects`}
            tone="cyan"
          />
          <StatPill
            icon="!"
            label="Biggest Need"
            value={report.biggestNeed?.position || "—"}
            sub={report.biggestNeed?.headline || "No major hole detected"}
            tone={report.biggestNeed?.severity === "CRITICAL" ? "danger" : "gold"}
          />
          <StatPill
            icon="◈"
            label="Draft Strategy"
            value={report.strategyLabel}
            sub={report.strategySub}
            tone="green"
          />
          <StatPill
            icon="★"
            label="Best Fit"
            value={report.bestDraftFit?.name || "TBD"}
            sub={report.bestDraftFit?.sub || "No draft class loaded"}
            tone="blue"
          />
        </section>

        <section className="teamneeds-content-grid">
          <section className="teamneeds-board-panel">
            <header className="teamneeds-section-title">
              <div>
                <p>Priority Board</p>
                <h3>What should we draft?</h3>
              </div>
              <span>{report.positionReports.length} position groups</span>
            </header>

            <div className="teamneeds-priority-grid">
              {report.positionReports.map((need) => (
                <button
                  key={need.position}
                  type="button"
                  className={`teamneeds-priority-card tone-${need.tone} ${
                    activeNeed?.position === need.position ? "is-active" : ""
                  }`}
                  onClick={() => setSelectedNeed(need.position)}
                >
                  <div className="teamneeds-priority-icon">{POSITION_BUCKETS[need.position]?.icon || "◆"}</div>

                  <div className="teamneeds-priority-main">
                    <div className="teamneeds-priority-topline">
                      <strong>{need.position}</strong>
                      <span>{need.severity}</span>
                    </div>

                    <h4>{need.headline}</h4>
                    <p>{need.summary}</p>

                    <div className="teamneeds-meter">
                      <i style={{ width: `${Math.min(100, Math.max(0, need.needScore))}%` }} />
                    </div>

                    <footer>
                      <span>Need Score</span>
                      <b>{Math.round(need.needScore)}</b>
                    </footer>
                  </div>
                </button>
              ))}
            </div>
          </section>

          <aside className="teamneeds-right-rail">
            <section className={`teamneeds-card teamneeds-spotlight tone-${activeNeed?.tone || "medium"}`}>
              <header className="teamneeds-card-header">
                <div>
                  <p>Selected Need</p>
                  <h3>{activeNeed?.draftLabel || "Draft Strategy"}</h3>
                </div>
                <span>{activeNeed?.severity || "LOW"}</span>
              </header>

              <div className="teamneeds-spotlight-body">
                <div className="teamneeds-big-icon">{POSITION_BUCKETS[activeNeed?.position]?.icon || "◈"}</div>

                <h4>{activeNeed?.archetype || "Best player available with positional awareness"}</h4>

                <p>{activeNeed?.explanation || "The roster does not show a severe weakness here, but the draft board should still be monitored."}</p>

                <div className="teamneeds-detail-grid">
                  <Detail label="NHL Count" value={activeNeed?.nhlCount} />
                  <Detail label="Org Count" value={activeNeed?.orgCount} />
                  <Detail label="Avg OVR" value={formatNumber(activeNeed?.avgOverall, 1)} />
                  <Detail label="Top OVR" value={formatNumber(activeNeed?.topOverall, 0)} />
                  <Detail label="Avg Age" value={formatNumber(activeNeed?.avgAge, 1)} />
                  <Detail label="Pipeline" value={activeNeed?.pipelineCount} />
                </div>
              </div>
            </section>

            <section className="teamneeds-card">
              <header className="teamneeds-card-header compact">
                <div>
                  <p>Best Draft Fits</p>
                  <h3>{activeNeed?.position || "BPA"} Targets</h3>
                </div>
                <button type="button" onClick={() => handleNavigate(SCREENS.DRAFT_CLASS)}>
                  Board ›
                </button>
              </header>

              <div className="teamneeds-target-list">
                {activeNeed?.draftFits?.length ? (
                  activeNeed.draftFits.map((player, index) => (
                    <DraftTargetRow key={player.id || player.name || index} player={player} index={index} />
                  ))
                ) : (
                  <p className="teamneeds-empty">No matching draft targets found. Load a draft class or scout more players.</p>
                )}
              </div>
            </section>
          </aside>
        </section>

        <section className="teamneeds-bottom-grid">
          <section className="teamneeds-card">
            <header className="teamneeds-section-title">
              <div>
                <p>Roster Shape</p>
                <h3>Position Group Breakdown</h3>
              </div>
              <span>{getTeamDisplayName(normalized.activeTeam)}</span>
            </header>

            <div className="teamneeds-table">
              <div className="teamneeds-table-head">
                <span>Pos</span>
                <span>NHL</span>
                <span>Org</span>
                <span>Avg OVR</span>
                <span>Top</span>
                <span>Age</span>
                <span>Pipeline</span>
                <span>Verdict</span>
              </div>

              {report.positionReports.map((row) => (
                <div key={row.position} className={`teamneeds-table-row tone-${row.tone}`}>
                  <span>
                    <b>{row.position}</b>
                    <small>{POSITION_BUCKETS[row.position]?.label}</small>
                  </span>
                  <span>{row.nhlCount}</span>
                  <span>{row.orgCount}</span>
                  <span>{formatNumber(row.avgOverall, 1)}</span>
                  <span>{formatNumber(row.topOverall, 0)}</span>
                  <span>{formatNumber(row.avgAge, 1)}</span>
                  <span>{row.pipelineCount}</span>
                  <span>{row.severity}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="teamneeds-card">
            <header className="teamneeds-section-title">
              <div>
                <p>GM Notes</p>
                <h3>Recommended Draft Plan</h3>
              </div>
              <span>{report.strategyLabel}</span>
            </header>

            <div className="teamneeds-plan">
              {report.plan.map((item, index) => (
                <article key={`${item.title}-${index}`} className={`teamneeds-plan-card tone-${item.tone}`}>
                  <span>{index + 1}</span>
                  <div>
                    <strong>{item.title}</strong>
                    <p>{item.text}</p>
                  </div>
                </article>
              ))}
            </div>
          </section>
        </section>
      </main>
    </div>
  );
}

function SideButton({ active, icon, label, onClick }) {
  return (
    <button type="button" className={`teamneeds-side-button ${active ? "is-active" : ""}`} onClick={onClick}>
      <span>{icon}</span>
      <small>{label}</small>
    </button>
  );
}

function StatPill({ icon, label, value, sub, tone }) {
  return (
    <article className={`teamneeds-stat-pill tone-${tone || "neutral"}`}>
      <div>{icon}</div>
      <section>
        <span>{label}</span>
        <strong>{value ?? "—"}</strong>
        <small>{sub || "—"}</small>
      </section>
    </article>
  );
}

function Detail({ label, value }) {
  return (
    <article>
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
    </article>
  );
}

function DraftTargetRow({ player, index }) {
  const stock = Number(player.stockChange || player.stock_change || player.rank_change || 0);
  const rank = player.rank || player.overall_rank || player.draft_rank || index + 1;

  return (
    <article className="teamneeds-target-row">
      <span>{rank}</span>
      <div>
        <strong>{getPlayerName(player)}</strong>
        <small>
          {normalizePosition(getPlayerPosition(player))} · {player.league || player.current_league || player.country || "Draft eligible"}
        </small>
      </div>
      <em className={stock >= 0 ? "up" : "down"}>
        {stock >= 0 ? "+" : ""}
        {stock}
      </em>
    </article>
  );
}

function TeamLogo({ team }) {
  const abbr = getTeamAbbreviation(team);

  return (
    <div className="teamneeds-team-logo" aria-label={`${getTeamDisplayName(team)} logo placeholder`}>
      <span>{abbr}</span>
    </div>
  );
}

function normalizeFranchiseState(rootState, controlledTeamId) {
  const allTeams = normalizeArrayMerged(
    rootState.teams,
    rootState.all_teams,
    rootState.allTeams,
    rootState.league?.teams,
    rootState.franchise?.teams
  );

  const activeTeam =
    findActiveTeam(rootState, allTeams, controlledTeamId) ||
    normalizeTeam(rootState.team || rootState.user_team || rootState.selected_team || { abbreviation: "CLB", name: "Club" });

  const userTeamId = String(
    controlledTeamId ||
      rootState.user_team_id ||
      activeTeam?.id ||
      rootState.team?.id ||
      ""
  );

  const rosterBrowserOrgs = rootState.roster_browser?.organizations || [];
  const userOrg =
    rosterBrowserOrgs.find((org) => String(org?.team_id || "") === userTeamId) ||
    rosterBrowserOrgs.find(
      (org) =>
        String(org?.name || "").toLowerCase() ===
        String(activeTeam?.name || "").toLowerCase()
    ) ||
    rosterBrowserOrgs[0];

  const roster = normalizeArrayMerged(
    rootState.roster,
    rootState.players,
    rootState.team_roster,
    rootState.teamRoster,
    rootState.user_roster,
    rootState.userRoster,
    rootState.active_roster,
    rootState.activeRoster,
    rootState.team?.roster,
    rootState.user_team?.roster,
    rootState.selected_team?.roster,
    rootState.franchise?.roster,
    userOrg?.nhl,
    userOrg?.ahl,
    userOrg?.echl
  );

  const prospects = normalizeArrayMerged(
    rootState.prospects,
    rootState.pipeline,
    rootState.team_prospects,
    rootState.teamProspects,
    rootState.prospect_pool,
    rootState.prospectPool,
    rootState.farm_system,
    rootState.farmSystem,
    rootState.team?.prospects,
    rootState.user_team?.prospects
  );

  const draftClass = normalizeArrayMerged(
    rootState.draft_class_rankings?.entries,
    rootState.draft_class,
    rootState.draftClass,
    rootState.scouting?.draft_class,
    rootState.scouting?.draftClass,
    rootState.prospect_rankings,
    rootState.prospectRankings,
    rootState.draft_board,
    rootState.draftBoard
  );

  return {
    rootState,
    activeTeam,
    allTeams,
    roster: filterTeamPlayers(roster, activeTeam),
    prospects: filterTeamProspects(prospects, activeTeam),
    draftClass,
  };
}

function buildTeamNeedsReport(normalized) {
  const roster = normalized.roster || EMPTY_ARRAY;
  const prospects = normalized.prospects || EMPTY_ARRAY;
  const draftClass = normalized.draftClass || EMPTY_ARRAY;

  const nhlRoster = roster.filter((player) => isLikelyNhlRosterPlayer(player));
  const fullOrg = [...roster, ...prospects];

  const positionReports = Object.keys(POSITION_BUCKETS)
    .map((position) => {
      const config = POSITION_BUCKETS[position];

      const nhlPlayers = nhlRoster.filter((player) => normalizePosition(getPlayerPosition(player)) === position);
      const orgPlayers = fullOrg.filter((player) => normalizePosition(getPlayerPosition(player)) === position);
      const pipelinePlayers = prospects.filter((player) => normalizePosition(getPlayerPosition(player)) === position);

      const avgOverall = average(orgPlayers.map(getPlayerOverall).filter(isFiniteNumber));
      const topOverall = Math.max(0, ...orgPlayers.map(getPlayerOverall).filter(isFiniteNumber));
      const avgAge = average(orgPlayers.map(getPlayerAge).filter(isFiniteNumber));
      const topProspectScore = Math.max(0, ...pipelinePlayers.map(getPotentialScore).filter(isFiniteNumber));

      const nhlCountGap = Math.max(0, config.minNhl - nhlPlayers.length);
      const orgCountGap = Math.max(0, config.minOrg - orgPlayers.length);
      const avgGap = Math.max(0, config.idealAvg - avgOverall);
      const topGap = Math.max(0, config.idealTop - topOverall);
      const pipelineGap = pipelinePlayers.length <= 1 ? 12 : pipelinePlayers.length <= 2 ? 6 : 0;
      const ageRisk = avgAge >= 30 ? 10 : avgAge >= 28 ? 5 : 0;
      const prospectUpsideRisk = topProspectScore < 78 ? 8 : topProspectScore < 84 ? 4 : 0;

      const needScore =
        nhlCountGap * 18 +
        orgCountGap * 8 +
        avgGap * 2.2 +
        topGap * 1.8 +
        pipelineGap +
        ageRisk +
        prospectUpsideRisk;

      const severity = getSeverity(needScore);
      const tone = NEED_COLORS[severity] || "low";

      const draftFits = findDraftFits(draftClass, position);

      return {
        position,
        label: config.label,
        draftLabel: config.draftLabel,
        archetype: config.archetype,
        nhlCount: nhlPlayers.length,
        orgCount: orgPlayers.length,
        pipelineCount: pipelinePlayers.length,
        avgOverall,
        topOverall,
        avgAge,
        topProspectScore,
        needScore,
        severity,
        tone,
        draftFits,
        headline: buildNeedHeadline(position, severity, nhlCountGap, avgGap, pipelinePlayers.length),
        summary: buildNeedSummary(position, nhlPlayers.length, orgPlayers.length, avgOverall, pipelinePlayers.length),
        explanation: buildNeedExplanation(position, severity, {
          nhlCountGap,
          orgCountGap,
          avgGap,
          topGap,
          pipelineGap,
          ageRisk,
          prospectUpsideRisk,
          nhlCount: nhlPlayers.length,
          orgCount: orgPlayers.length,
          avgOverall,
          topOverall,
          avgAge,
          pipelineCount: pipelinePlayers.length,
        }),
      };
    })
    .sort((a, b) => b.needScore - a.needScore);

  const biggestNeed = positionReports[0] || null;
  const secondNeed = positionReports[1] || null;
  const bestDraftFit = biggestNeed?.draftFits?.[0] || null;

  const criticalCount = positionReports.filter((row) => row.severity === "CRITICAL").length;
  const highCount = positionReports.filter((row) => row.severity === "HIGH").length;

  const strategyLabel =
    criticalCount >= 2
      ? "Need-heavy draft"
      : criticalCount === 1
        ? "Fix the major hole"
        : highCount >= 2
          ? "Balance BPA + need"
          : "Best player available";

  const strategySub =
    biggestNeed && biggestNeed.needScore >= 70
      ? `${biggestNeed.position} should drive the first-round filter`
      : biggestNeed
        ? `${biggestNeed.position} is the softest area, but do not force it`
        : "No roster data loaded";

  const plan = buildDraftPlan(positionReports, bestDraftFit);

  return {
    totalRoster: fullOrg.length,
    nhlRosterCount: nhlRoster.length,
    prospectCount: prospects.length,
    positionReports,
    biggestNeed,
    secondNeed,
    bestDraftFit: bestDraftFit
      ? {
          ...bestDraftFit,
          sub: `${normalizePosition(getPlayerPosition(bestDraftFit))} · ${bestDraftFit.league || bestDraftFit.country || "Draft eligible"}`,
        }
      : null,
    strategyLabel,
    strategySub,
    plan,
  };
}

function buildDraftPlan(positionReports, bestDraftFit) {
  const biggest = positionReports[0];
  const second = positionReports[1];
  const third = positionReports[2];

  const plan = [];

  if (biggest) {
    plan.push({
      tone: biggest.tone,
      title: `Round 1 filter: ${POSITION_BUCKETS[biggest.position]?.archetype || biggest.position}`,
      text:
        biggest.needScore >= 70
          ? `This is not a luxury pick. ${biggest.position} is the clearest roster weakness, so only pass on it if the best-player-available gap is massive.`
          : `${biggest.position} is the top need, but the score is not severe enough to ignore a better player at another position.`,
    });
  }

  if (bestDraftFit) {
    plan.push({
      tone: "high",
      title: `Current best fit: ${getPlayerName(bestDraftFit)}`,
      text: `${getPlayerName(bestDraftFit)} matches the top positional need and should be compared against your highest-upside BPA options before making the pick.`,
    });
  } else {
    plan.push({
      tone: "medium",
      title: "Scout before forcing the pick",
      text: "No clear draft-class match was found for the top need. Scout more players or use this screen as a positional filter, not an autopick button.",
    });
  }

  if (second && third) {
    plan.push({
      tone: "low",
      title: `Middle rounds: ${second.position} / ${third.position}`,
      text: `After addressing the top need, use the middle rounds to rebuild depth at ${second.position} and ${third.position}. These are the next soft spots in the organization.`,
    });
  }

  return plan;
}

function findDraftFits(draftClass, position) {
  return [...(draftClass || EMPTY_ARRAY)]
    .filter((player) => {
      const playerPos = normalizePosition(getPlayerPosition(player));
      if (position === "D") return playerPos === "D" || playerPos === "LD" || playerPos === "RD";
      return playerPos === position;
    })
    .map((player) => ({
      ...player,
      fitScore: calculateDraftFitScore(player),
    }))
    .sort((a, b) => {
      const ar = Number(a.rank || a.overall_rank || a.draft_rank || 999);
      const br = Number(b.rank || b.overall_rank || b.draft_rank || 999);

      if (ar !== br) return ar - br;

      return b.fitScore - a.fitScore;
    })
    .slice(0, 6);
}

function calculateDraftFitScore(player) {
  const rank = Number(player.rank || player.overall_rank || player.draft_rank || 999);
  const potential = getPotentialScore(player);
  const overall = getPlayerOverall(player);
  const stock = Number(player.stock_change || player.stockChange || player.rank_change || 0);

  return Math.max(0, 110 - rank) + potential * 0.65 + overall * 0.35 + stock * 1.5;
}

function buildNeedHeadline(position, severity, nhlCountGap, avgGap, pipelineCount) {
  if (severity === "CRITICAL") {
    if (nhlCountGap > 0) return `${position} has a roster shortage`;
    if (pipelineCount <= 1) return `${position} pipeline is dangerously thin`;
    return `${position} lacks high-end quality`;
  }

  if (severity === "HIGH") {
    if (avgGap >= 4) return `${position} quality is below target`;
    if (pipelineCount <= 2) return `${position} future depth is light`;
    return `${position} needs reinforcement`;
  }

  if (severity === "MEDIUM") return `${position} is worth monitoring`;
  if (severity === "LOW") return `${position} is stable enough`;
  return `${position} is a strength`;
}

function buildNeedSummary(position, nhlCount, orgCount, avgOverall, pipelineCount) {
  return `${nhlCount} NHL / ${orgCount} org · ${formatNumber(avgOverall, 1)} avg OVR · ${pipelineCount} pipeline players.`;
}

function buildNeedExplanation(position, severity, data) {
  const label = POSITION_BUCKETS[position]?.label || position;

  if (severity === "CRITICAL") {
    return `${label} should be treated as a draft priority. The score is being driven by roster count gaps, quality gaps, or a weak prospect pipeline. This is the kind of hole that becomes painful two seasons from now if ignored.`;
  }

  if (severity === "HIGH") {
    return `${label} is not completely broken, but the organization is leaning too thin here. Drafting this position would protect the future roster and keep the depth chart from aging out.`;
  }

  if (severity === "MEDIUM") {
    return `${label} is not the biggest emergency, but there is enough softness here to use it as a tiebreaker when two prospects are close on your board.`;
  }

  if (severity === "LOW") {
    return `${label} is stable. Do not force this position unless the player is clearly the best available prospect.`;
  }

  return `${label} looks like a current organizational strength. Draft here only for elite value.`;
}

function getSeverity(score) {
  if (score >= 80) return "CRITICAL";
  if (score >= 55) return "HIGH";
  if (score >= 32) return "MEDIUM";
  if (score >= 14) return "LOW";
  return "STRENGTH";
}

function filterTeamPlayers(players, activeTeam) {
  return (players || EMPTY_ARRAY).filter((player) => {
    const teamId =
      player?.team_id ||
      player?.teamId ||
      player?.team ||
      player?.team_abbr ||
      player?.teamAbbr ||
      player?.current_team ||
      player?.currentTeam ||
      "";

    if (!teamId) return true;
    return isSameTeamIdentifier(teamId, activeTeam);
  });
}

function filterTeamProspects(players, activeTeam) {
  return (players || EMPTY_ARRAY).filter((player) => {
    const teamId =
      player?.rights_team_id ||
      player?.rightsTeamId ||
      player?.team_id ||
      player?.teamId ||
      player?.team ||
      player?.rights_team ||
      player?.rightsTeam ||
      "";

    if (!teamId) return true;
    return isSameTeamIdentifier(teamId, activeTeam);
  });
}

function isLikelyNhlRosterPlayer(player) {
  const status = String(
    player?.status ||
      player?.roster_status ||
      player?.rosterStatus ||
      player?.league ||
      player?.current_league ||
      ""
  ).toLowerCase();

  if (status.includes("prospect")) return false;
  if (status.includes("ahl")) return false;
  if (status.includes("junior")) return false;
  if (status.includes("unsigned")) return false;

  const role = String(player?.role || player?.line_role || "").toLowerCase();
  if (role.includes("prospect")) return false;

  return true;
}

function normalizeArrayMerged(...values) {
  const output = [];

  values.forEach((value) => {
    if (Array.isArray(value)) {
      output.push(...value);
      return;
    }

    if (value && typeof value === "object") {
      if (Array.isArray(value.items)) output.push(...value.items);
      if (Array.isArray(value.data)) output.push(...value.data);
      if (Array.isArray(value.results)) output.push(...value.results);
      if (Array.isArray(value.rows)) output.push(...value.rows);
      if (Array.isArray(value.players)) output.push(...value.players);
      if (Array.isArray(value.roster)) output.push(...value.roster);
      if (Array.isArray(value.prospects)) output.push(...value.prospects);
      if (Array.isArray(value.draft_class)) output.push(...value.draft_class);
      if (Array.isArray(value.draftClass)) output.push(...value.draftClass);
    }
  });

  return output.filter(Boolean);
}

function normalizeTeam(team) {
  if (!team || typeof team !== "object") {
    return {
      id: String(team || "CLB"),
      abbreviation: String(team || "CLB").slice(0, 3).toUpperCase(),
      name: String(team || "Club"),
      city: "Franchise",
    };
  }

  return {
    ...team,
    id: getTeamId(team),
    abbreviation: getTeamAbbreviation(team),
    name: getTeamDisplayName(team),
    city: getTeamCity(team),
  };
}

function findActiveTeam(rootState, allTeams, controlledTeamId) {
  const direct =
    rootState.team ||
    rootState.user_team ||
    rootState.selected_team ||
    rootState.active_team ||
    rootState.franchise_team ||
    null;

  if (direct && typeof direct === "object") return normalizeTeam(direct);

  const id =
    controlledTeamId ||
    direct ||
    rootState.user_team_id ||
    rootState.controlled_team_id ||
    rootState.selected_team_id ||
    rootState.gm_team_id ||
    rootState.team_id;

  const found = (allTeams || EMPTY_ARRAY).find((team) => isSameTeamIdentifier(id, team));
  if (found) return normalizeTeam(found);

  return normalizeTeam({ abbreviation: "CLB", name: "Club", city: "Franchise" });
}

function getTeamId(team) {
  if (!team || typeof team !== "object") return String(team || "");

  return String(
    team.id ||
      team.team_id ||
      team.teamId ||
      team.abbreviation ||
      team.abbr ||
      team.short_name ||
      team.name ||
      ""
  );
}

function getTeamAbbreviation(team) {
  if (!team) return "CLB";

  if (typeof team === "string") {
    const s = team.trim();
    return s ? s.slice(0, 3).toUpperCase() : "CLB";
  }

  return String(
    team.abbreviation ||
      team.abbr ||
      team.short_name ||
      team.shortName ||
      team.code ||
      team.id ||
      "CLB"
  )
    .slice(0, 3)
    .toUpperCase();
}

function getTeamDisplayName(team) {
  if (!team) return "Club";
  if (typeof team === "string") return team;

  return (
    team.full_name ||
    team.fullName ||
    team.name ||
    team.team_name ||
    team.nickname ||
    team.abbreviation ||
    team.abbr ||
    "Club"
  );
}

function getTeamCity(team) {
  if (!team || typeof team !== "object") return "Franchise";
  return team.city || team.location || team.market || team.region || "Franchise";
}

function isSameTeamIdentifier(identifier, team) {
  if (!identifier || !team) return false;

  const raw = String(
    typeof identifier === "object"
      ? identifier.id || identifier.team_id || identifier.abbreviation || identifier.abbr || identifier.name || ""
      : identifier
  ).toLowerCase();

  const teamId = getTeamId(team).toLowerCase();
  const abbr = getTeamAbbreviation(team).toLowerCase();
  const name = getTeamDisplayName(team).toLowerCase();

  return raw === teamId || raw === abbr || raw === name;
}

function getPlayerName(player) {
  if (!player) return "Player";
  if (typeof player === "string") return player;

  return (
    player.name ||
    player.player_name ||
    player.full_name ||
    player.fullName ||
    `${player.first_name || player.firstName || ""} ${player.last_name || player.lastName || ""}`.trim() ||
    "Player"
  );
}

function getPlayerPosition(player) {
  if (!player || typeof player !== "object") return "—";

  return (
    player.position ||
    player.pos ||
    player.primary_position ||
    player.primaryPosition ||
    player.draft_position ||
    player.draftPosition ||
    "—"
  );
}

function normalizePosition(position) {
  const raw = String(position || "").trim().toUpperCase();

  if (raw === "GOALIE" || raw === "GK") return "G";
  if (raw === "LD" || raw === "RD" || raw === "DEF" || raw === "DEFENSE" || raw === "DEFENSEMAN") return "D";
  if (raw === "LEFT WING") return "LW";
  if (raw === "RIGHT WING") return "RW";
  if (raw === "CENTER" || raw === "CENTRE") return "C";

  if (raw.includes("/")) {
    const first = raw.split("/")[0];
    return normalizePosition(first);
  }

  if (["C", "LW", "RW", "D", "G"].includes(raw)) return raw;

  return raw || "—";
}

function getPlayerOverall(player) {
  return firstNumberOrNull(
    player?.overall,
    player?.ovr,
    player?.rating,
    player?.current_overall,
    player?.currentOverall,
    player?.attributes?.overall,
    player?.profile?.overall
  );
}

function getPlayerAge(player) {
  return firstNumberOrNull(player?.age, player?.profile?.age, player?.bio?.age);
}

function getPotentialScore(player) {
  const raw = firstNumberOrNull(
    player?.potential_score,
    player?.potentialScore,
    player?.potential_overall,
    player?.potentialOverall,
    player?.ceiling_score,
    player?.ceilingScore,
    player?.overall,
    player?.ovr
  );

  if (raw !== null) return raw;

  const grade = String(player?.potential || player?.ceiling || player?.grade || "").toUpperCase();

  if (grade.includes("ELITE")) return 94;
  if (grade.includes("FRANCHISE")) return 98;
  if (grade.includes("A+")) return 96;
  if (grade.includes("A")) return 91;
  if (grade.includes("B+")) return 85;
  if (grade.includes("B")) return 80;
  if (grade.includes("C+")) return 74;
  if (grade.includes("C")) return 68;
  if (grade.includes("D")) return 60;

  return 0;
}

function firstNumberOrNull(...values) {
  for (const value of values) {
    if (value === null || value === undefined || value === "") continue;
    const number = Number(value);
    if (Number.isFinite(number)) return number;
  }

  return null;
}

function average(values) {
  const clean = values.filter(isFiniteNumber);
  if (!clean.length) return 0;
  return clean.reduce((sum, value) => sum + value, 0) / clean.length;
}

function isFiniteNumber(value) {
  return Number.isFinite(Number(value));
}

function formatNumber(value, decimals = 0) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "—";
  return number.toFixed(decimals);
}

function TeamNeedsStyles() {
  return (
    <style>{`
      .teamneeds-root {
        --bg: #04101a;
        --bg-2: #061522;
        --panel: rgba(9, 25, 38, 0.94);
        --panel-2: rgba(12, 35, 52, 0.94);
        --panel-3: rgba(15, 46, 66, 0.78);
        --line: rgba(156, 218, 236, 0.14);
        --line-2: rgba(115, 229, 241, 0.25);
        --line-strong: rgba(73, 231, 240, 0.5);
        --text: #e9f7fb;
        --muted: #8096a8;
        --muted-2: #607789;
        --cyan: #13d8e7;
        --cyan-soft: rgba(19, 216, 231, 0.13);
        --gold: #e9a83c;
        --gold-soft: rgba(233, 168, 60, 0.14);
        --green: #52df94;
        --green-soft: rgba(82, 223, 148, 0.13);
        --red: #ff606d;
        --red-soft: rgba(255, 96, 109, 0.13);
        --blue: #8ab4ff;
        --blue-soft: rgba(138, 180, 255, 0.13);
        --purple: #c992ff;
        --purple-soft: rgba(201, 146, 255, 0.14);
        --shadow: 0 24px 70px rgba(0, 0, 0, 0.42);

        min-height: 100vh;
        width: 100%;
        background:
          radial-gradient(circle at 24% 0%, rgba(19, 216, 231, 0.12), transparent 30%),
          radial-gradient(circle at 92% 18%, rgba(233, 168, 60, 0.08), transparent 26%),
          linear-gradient(180deg, #06131f 0%, #020a11 100%);
        color: var(--text);
        display: grid;
        grid-template-columns: 94px minmax(0, 1fr);
        overflow: hidden;
        font-family:
          Inter,
          ui-sans-serif,
          system-ui,
          -apple-system,
          BlinkMacSystemFont,
          "Segoe UI",
          sans-serif;
      }

      .teamneeds-root *,
      .teamneeds-root *::before,
      .teamneeds-root *::after {
        box-sizing: border-box;
      }

      .teamneeds-root button {
        font-family: inherit;
      }

      .teamneeds-sidebar {
        min-height: 100vh;
        background:
          linear-gradient(180deg, rgba(5, 16, 26, 0.98), rgba(3, 10, 17, 0.98)),
          radial-gradient(circle at 100% 14%, rgba(19, 216, 231, 0.14), transparent 34%);
        border-right: 1px solid var(--line);
        display: flex;
        flex-direction: column;
        position: relative;
        z-index: 4;
      }

      .teamneeds-brand-button {
        height: 112px;
        border: 0;
        background: transparent;
        color: var(--text);
        display: grid;
        place-items: center;
        border-bottom: 1px solid var(--line);
        cursor: pointer;
      }

      .teamneeds-brand-button span {
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        border: 2px solid rgba(223, 245, 250, 0.52);
        border-radius: 12px;
        color: var(--cyan);
        box-shadow: 0 0 24px rgba(19, 216, 231, 0.18);
      }

      .teamneeds-side-nav {
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 18px 0;
      }

      .teamneeds-side-button {
        width: 100%;
        min-height: 66px;
        border: 0;
        background: transparent;
        color: var(--muted);
        display: grid;
        place-items: center;
        gap: 4px;
        cursor: pointer;
        position: relative;
        transition: 0.2s ease;
      }

      .teamneeds-side-button:hover {
        color: var(--text);
        background: rgba(255, 255, 255, 0.035);
      }

      .teamneeds-side-button.is-active {
        color: var(--cyan);
        background:
          linear-gradient(90deg, rgba(19, 216, 231, 0.17), rgba(19, 216, 231, 0.03)),
          radial-gradient(circle at 100% 50%, rgba(19, 216, 231, 0.24), transparent 52%);
      }

      .teamneeds-side-button.is-active::before {
        content: "";
        position: absolute;
        left: 0;
        top: 12px;
        bottom: 12px;
        width: 3px;
        border-radius: 999px;
        background: var(--cyan);
        box-shadow: 0 0 22px rgba(19, 216, 231, 0.8);
      }

      .teamneeds-side-button span {
        font-size: 22px;
      }

      .teamneeds-side-button small {
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 0.02em;
      }

      .teamneeds-main {
        min-width: 0;
        height: 100vh;
        overflow: auto;
        padding: 24px 26px 26px;
      }

      .teamneeds-main::-webkit-scrollbar {
        width: 10px;
      }

      .teamneeds-main::-webkit-scrollbar-thumb {
        background: rgba(110, 173, 191, 0.25);
        border-radius: 999px;
      }

      .teamneeds-topbar {
        min-height: 102px;
        display: grid;
        grid-template-columns: minmax(250px, 1fr) minmax(360px, 1.35fr) minmax(300px, 0.9fr);
        align-items: center;
        gap: 22px;
      }

      .teamneeds-team-identity {
        display: flex;
        align-items: center;
        gap: 18px;
        min-width: 0;
      }

      .teamneeds-team-logo {
        width: 82px;
        height: 82px;
        border-radius: 24px;
        display: grid;
        place-items: center;
        background:
          radial-gradient(circle at 30% 0%, rgba(19, 216, 231, 0.35), transparent 34%),
          linear-gradient(180deg, rgba(18, 42, 61, 0.88), rgba(6, 20, 31, 0.98));
        border: 1px solid var(--line-2);
        box-shadow: var(--shadow);
        flex: 0 0 auto;
      }

      .teamneeds-team-logo span {
        font-size: 24px;
        font-weight: 1000;
        letter-spacing: 0.12em;
        color: var(--cyan);
      }

      .teamneeds-team-identity p,
      .teamneeds-draft-board-title p,
      .teamneeds-section-title p,
      .teamneeds-card-header p {
        margin: 0 0 5px;
        color: var(--cyan);
        text-transform: uppercase;
        letter-spacing: 0.22em;
        font-size: 11px;
        font-weight: 1000;
      }

      .teamneeds-team-identity h1 {
        margin: 0;
        font-size: clamp(31px, 3vw, 48px);
        line-height: 0.92;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }

      .teamneeds-team-identity span {
        display: block;
        margin-top: 7px;
        color: var(--muted);
        font-size: 12px;
        font-weight: 800;
      }

      .teamneeds-draft-board-title {
        text-align: center;
      }

      .teamneeds-draft-board-title h2 {
        margin: 0;
        font-size: clamp(31px, 3vw, 50px);
        line-height: 0.92;
        letter-spacing: 0.14em;
        text-transform: uppercase;
      }

      .teamneeds-actions {
        display: flex;
        justify-content: flex-end;
        gap: 12px;
        flex-wrap: wrap;
      }

      .teamneeds-actions button,
      .teamneeds-card-header button {
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(12, 31, 47, 0.72);
        color: var(--text);
        padding: 11px 16px;
        font-size: 11px;
        font-weight: 1000;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        cursor: pointer;
        transition: 0.2s ease;
      }

      .teamneeds-actions button:hover,
      .teamneeds-card-header button:hover {
        border-color: var(--line-strong);
        background: rgba(19, 216, 231, 0.12);
        transform: translateY(-1px);
      }

      .teamneeds-stat-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0;
        border: 1px solid var(--line);
        background: rgba(8, 23, 35, 0.86);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow);
        margin-top: 18px;
      }

      .teamneeds-stat-pill {
        min-height: 92px;
        padding: 17px 18px;
        display: flex;
        align-items: center;
        gap: 14px;
        border-right: 1px solid rgba(156, 218, 236, 0.08);
        background:
          linear-gradient(180deg, rgba(18, 42, 61, 0.45), rgba(6, 20, 31, 0.34)),
          radial-gradient(circle at 100% 0%, rgba(19, 216, 231, 0.05), transparent 52%);
      }

      .teamneeds-stat-pill:last-child {
        border-right: 0;
      }

      .teamneeds-stat-pill > div {
        width: 44px;
        height: 44px;
        flex: 0 0 auto;
        display: grid;
        place-items: center;
        border-radius: 14px;
        background: rgba(148, 185, 205, 0.12);
        border: 1px solid rgba(148, 185, 205, 0.12);
        font-size: 18px;
      }

      .teamneeds-stat-pill span,
      .teamneeds-stat-pill small {
        display: block;
      }

      .teamneeds-stat-pill span {
        color: var(--muted);
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 1000;
      }

      .teamneeds-stat-pill strong {
        display: block;
        margin-top: 3px;
        font-size: 24px;
        line-height: 1;
        color: var(--text);
      }

      .teamneeds-stat-pill small {
        margin-top: 5px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 800;
      }

      .teamneeds-stat-pill.tone-cyan > div { color: var(--cyan); background: var(--cyan-soft); }
      .teamneeds-stat-pill.tone-gold > div { color: var(--gold); background: var(--gold-soft); }
      .teamneeds-stat-pill.tone-green > div { color: var(--green); background: var(--green-soft); }
      .teamneeds-stat-pill.tone-blue > div { color: var(--blue); background: var(--blue-soft); }
      .teamneeds-stat-pill.tone-danger > div { color: var(--red); background: var(--red-soft); }

      .teamneeds-content-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.4fr) minmax(360px, 0.75fr);
        gap: 22px;
        margin-top: 22px;
      }

      .teamneeds-board-panel,
      .teamneeds-card {
        background:
          linear-gradient(180deg, rgba(13, 31, 46, 0.94), rgba(5, 17, 28, 0.94)),
          radial-gradient(circle at 50% 0%, rgba(19, 216, 231, 0.08), transparent 40%);
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: var(--shadow);
        overflow: hidden;
      }

      .teamneeds-section-title,
      .teamneeds-card-header {
        min-height: 76px;
        padding: 18px 20px;
        border-bottom: 1px solid var(--line);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }

      .teamneeds-section-title h3,
      .teamneeds-card-header h3 {
        margin: 0;
        font-size: 19px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .teamneeds-section-title span,
      .teamneeds-card-header span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }

      .teamneeds-card-header.compact {
        min-height: 68px;
      }

      .teamneeds-priority-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 14px;
        padding: 18px;
      }

      .teamneeds-priority-card {
        min-height: 330px;
        border: 1px solid rgba(156, 218, 236, 0.12);
        border-radius: 18px;
        background:
          radial-gradient(circle at 50% 0%, rgba(255, 255, 255, 0.05), transparent 36%),
          rgba(7, 22, 35, 0.88);
        color: var(--text);
        padding: 16px;
        text-align: left;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        gap: 16px;
        transition: 0.2s ease;
      }

      .teamneeds-priority-card:hover,
      .teamneeds-priority-card.is-active {
        transform: translateY(-2px);
        border-color: var(--line-strong);
        box-shadow: 0 20px 46px rgba(0, 0, 0, 0.25);
      }

      .teamneeds-priority-icon {
        width: 52px;
        height: 52px;
        border-radius: 18px;
        display: grid;
        place-items: center;
        font-size: 22px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.08);
      }

      .teamneeds-priority-main {
        display: flex;
        flex-direction: column;
        flex: 1;
      }

      .teamneeds-priority-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }

      .teamneeds-priority-topline strong {
        font-size: 28px;
        letter-spacing: 0.08em;
      }

      .teamneeds-priority-topline span {
        font-size: 9px;
        font-weight: 1000;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: var(--muted);
      }

      .teamneeds-priority-card h4 {
        margin: 14px 0 0;
        font-size: 18px;
        line-height: 1.1;
      }

      .teamneeds-priority-card p {
        margin: 10px 0 0;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.45;
        font-weight: 700;
      }

      .teamneeds-meter {
        height: 8px;
        margin-top: auto;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        overflow: hidden;
      }

      .teamneeds-meter i {
        display: block;
        height: 100%;
        border-radius: inherit;
        background: var(--cyan);
      }

      .teamneeds-priority-card footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 10px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 1000;
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }

      .teamneeds-priority-card footer b {
        color: var(--text);
        font-size: 18px;
      }

      .teamneeds-priority-card.tone-critical {
        border-color: rgba(255, 96, 109, 0.34);
        background:
          radial-gradient(circle at 50% 0%, rgba(255, 96, 109, 0.14), transparent 40%),
          rgba(7, 22, 35, 0.88);
      }

      .teamneeds-priority-card.tone-critical .teamneeds-priority-icon,
      .teamneeds-spotlight.tone-critical .teamneeds-big-icon {
        color: var(--red);
        background: var(--red-soft);
      }

      .teamneeds-priority-card.tone-critical .teamneeds-meter i {
        background: var(--red);
      }

      .teamneeds-priority-card.tone-high .teamneeds-priority-icon,
      .teamneeds-spotlight.tone-high .teamneeds-big-icon {
        color: var(--gold);
        background: var(--gold-soft);
      }

      .teamneeds-priority-card.tone-high .teamneeds-meter i {
        background: var(--gold);
      }

      .teamneeds-priority-card.tone-medium .teamneeds-priority-icon,
      .teamneeds-spotlight.tone-medium .teamneeds-big-icon {
        color: var(--blue);
        background: var(--blue-soft);
      }

      .teamneeds-priority-card.tone-medium .teamneeds-meter i {
        background: var(--blue);
      }

      .teamneeds-priority-card.tone-low .teamneeds-priority-icon,
      .teamneeds-spotlight.tone-low .teamneeds-big-icon,
      .teamneeds-priority-card.tone-strength .teamneeds-priority-icon,
      .teamneeds-spotlight.tone-strength .teamneeds-big-icon {
        color: var(--green);
        background: var(--green-soft);
      }

      .teamneeds-priority-card.tone-low .teamneeds-meter i,
      .teamneeds-priority-card.tone-strength .teamneeds-meter i {
        background: var(--green);
      }

      .teamneeds-right-rail {
        display: grid;
        gap: 22px;
        align-content: start;
      }

      .teamneeds-spotlight-body {
        padding: 22px;
      }

      .teamneeds-big-icon {
        width: 72px;
        height: 72px;
        border-radius: 24px;
        display: grid;
        place-items: center;
        font-size: 32px;
        margin-bottom: 18px;
      }

      .teamneeds-spotlight-body h4 {
        margin: 0;
        font-size: 22px;
        line-height: 1.12;
      }

      .teamneeds-spotlight-body p {
        margin: 12px 0 0;
        color: var(--muted);
        line-height: 1.55;
        font-size: 13px;
        font-weight: 750;
      }

      .teamneeds-detail-grid {
        margin-top: 20px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }

      .teamneeds-detail-grid article {
        min-height: 70px;
        padding: 12px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.07);
      }

      .teamneeds-detail-grid span {
        display: block;
        color: var(--muted);
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 1000;
      }

      .teamneeds-detail-grid strong {
        display: block;
        margin-top: 7px;
        font-size: 22px;
      }

      .teamneeds-target-list {
        padding: 14px;
        display: grid;
        gap: 10px;
      }

      .teamneeds-target-row {
        min-height: 58px;
        display: grid;
        grid-template-columns: 38px minmax(0, 1fr) 46px;
        align-items: center;
        gap: 10px;
        padding: 10px;
        border-radius: 14px;
        border: 1px solid rgba(156, 218, 236, 0.09);
        background: rgba(255, 255, 255, 0.04);
      }

      .teamneeds-target-row > span {
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        border-radius: 12px;
        background: rgba(19, 216, 231, 0.1);
        color: var(--cyan);
        font-weight: 1000;
      }

      .teamneeds-target-row strong,
      .teamneeds-target-row small {
        display: block;
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .teamneeds-target-row strong {
        font-size: 13px;
      }

      .teamneeds-target-row small {
        margin-top: 3px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 750;
      }

      .teamneeds-target-row em {
        font-style: normal;
        text-align: right;
        font-size: 12px;
        font-weight: 1000;
      }

      .teamneeds-target-row em.up {
        color: var(--green);
      }

      .teamneeds-target-row em.down {
        color: var(--red);
      }

      .teamneeds-bottom-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.1fr) minmax(360px, 0.9fr);
        gap: 22px;
        margin-top: 22px;
      }

      .teamneeds-table {
        padding: 0 14px 14px;
      }

      .teamneeds-table-head,
      .teamneeds-table-row {
        display: grid;
        grid-template-columns: minmax(120px, 1fr) 58px 58px 78px 58px 58px 72px 88px;
        align-items: center;
        gap: 8px;
      }

      .teamneeds-table-head {
        height: 36px;
        color: rgba(233, 247, 251, 0.64);
        text-transform: uppercase;
        font-size: 10px;
        font-weight: 1000;
        letter-spacing: 0.08em;
        border-bottom: 1px solid rgba(156, 218, 236, 0.12);
      }

      .teamneeds-table-row {
        min-height: 54px;
        color: rgba(233, 247, 251, 0.83);
        font-size: 12px;
        font-weight: 850;
        border-bottom: 1px solid rgba(156, 218, 236, 0.07);
      }

      .teamneeds-table-row:last-child {
        border-bottom: 0;
      }

      .teamneeds-table-row > span:not(:first-child),
      .teamneeds-table-head > span:not(:first-child) {
        text-align: right;
      }

      .teamneeds-table-row b,
      .teamneeds-table-row small {
        display: block;
      }

      .teamneeds-table-row b {
        font-size: 15px;
      }

      .teamneeds-table-row small {
        margin-top: 3px;
        color: var(--muted);
        font-size: 10px;
      }

      .teamneeds-table-row.tone-critical > span:last-child {
        color: var(--red);
      }

      .teamneeds-table-row.tone-high > span:last-child {
        color: var(--gold);
      }

      .teamneeds-table-row.tone-medium > span:last-child {
        color: var(--blue);
      }

      .teamneeds-table-row.tone-low > span:last-child,
      .teamneeds-table-row.tone-strength > span:last-child {
        color: var(--green);
      }

      .teamneeds-plan {
        padding: 18px;
        display: grid;
        gap: 12px;
      }

      .teamneeds-plan-card {
        display: grid;
        grid-template-columns: 42px minmax(0, 1fr);
        gap: 14px;
        padding: 16px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(156, 218, 236, 0.09);
      }

      .teamneeds-plan-card > span {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        font-weight: 1000;
        background: rgba(19, 216, 231, 0.1);
        color: var(--cyan);
      }

      .teamneeds-plan-card strong {
        display: block;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }

      .teamneeds-plan-card p {
        margin: 7px 0 0;
        color: var(--muted);
        line-height: 1.5;
        font-size: 13px;
        font-weight: 750;
      }

      .teamneeds-plan-card.tone-critical > span {
        color: var(--red);
        background: var(--red-soft);
      }

      .teamneeds-plan-card.tone-high > span {
        color: var(--gold);
        background: var(--gold-soft);
      }

      .teamneeds-plan-card.tone-medium > span {
        color: var(--blue);
        background: var(--blue-soft);
      }

      .teamneeds-plan-card.tone-low > span,
      .teamneeds-plan-card.tone-strength > span {
        color: var(--green);
        background: var(--green-soft);
      }

      .teamneeds-empty {
        margin: 0;
        padding: 18px;
        color: var(--muted);
        line-height: 1.5;
        font-size: 13px;
        font-weight: 800;
      }

      @media (max-width: 1320px) {
        .teamneeds-topbar,
        .teamneeds-content-grid,
        .teamneeds-bottom-grid {
          grid-template-columns: 1fr;
        }

        .teamneeds-draft-board-title {
          text-align: left;
        }

        .teamneeds-actions {
          justify-content: flex-start;
        }

        .teamneeds-priority-grid {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
      }

      @media (max-width: 900px) {
        .teamneeds-root {
          grid-template-columns: 1fr;
        }

        .teamneeds-sidebar {
          display: none;
        }

        .teamneeds-main {
          padding: 16px;
        }

        .teamneeds-stat-strip,
        .teamneeds-priority-grid,
        .teamneeds-detail-grid {
          grid-template-columns: 1fr;
        }

        .teamneeds-stat-pill {
          border-right: 0;
          border-bottom: 1px solid rgba(156, 218, 236, 0.08);
        }

        .teamneeds-stat-pill:last-child {
          border-bottom: 0;
        }

        .teamneeds-table {
          overflow-x: auto;
        }

        .teamneeds-table-head,
        .teamneeds-table-row {
          min-width: 820px;
        }
      }

      @media (max-width: 620px) {
        .teamneeds-team-identity {
          align-items: flex-start;
        }

        .teamneeds-team-logo {
          width: 62px;
          height: 62px;
          border-radius: 18px;
        }

        .teamneeds-team-logo span {
          font-size: 18px;
        }

        .teamneeds-team-identity h1,
        .teamneeds-draft-board-title h2 {
          font-size: 30px;
        }

        .teamneeds-priority-card {
          min-height: 260px;
        }
      }
    `}</style>
  );
}

export default TeamNeeds;