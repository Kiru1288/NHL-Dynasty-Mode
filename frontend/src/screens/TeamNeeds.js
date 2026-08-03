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
    <div className="teamneeds-root register-ops" data-register="ops">
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

        <section className="teamneeds-ops-context" aria-label="Draft planning context">
          <ContextCell
            code="ROS"
            label="Roster"
            value={report.totalRoster || "—"}
            sub={`${report.nhlRosterCount} NHL · ${report.prospectCount} pipeline`}
          />
          <ContextCell
            code="PRI"
            label="Priority"
            value={report.biggestNeed?.position || "—"}
            sub={report.biggestNeed ? formatNeedContext(report.biggestNeed) : "No roster data loaded"}
            tone={report.biggestNeed?.tone}
          />
          <ContextCell
            code="STR"
            label="Strategy"
            value={report.strategyLabel}
            sub={report.strategySub}
          />
          <ContextCell
            code="FIT"
            label="Best fit"
            value={report.bestDraftFit ? getPlayerName(report.bestDraftFit) : "—"}
            sub={report.bestDraftFit?.sub || "Load draft class to match needs"}
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

            <div className="teamneeds-depth-board">
              <div className="teamneeds-depth-head">
                <span>#</span>
                <span>Pos</span>
                <span>Organizational depth</span>
                <span>Context</span>
                <span>Score</span>
              </div>

              {report.positionReports.map((need, index) => {
                const config = POSITION_BUCKETS[need.position] || {};
                return (
                  <button
                    key={need.position}
                    type="button"
                    className={`teamneeds-depth-row tone-${need.tone} ${
                      activeNeed?.position === need.position ? "is-active" : ""
                    }`}
                    onClick={() => setSelectedNeed(need.position)}
                  >
                    <span className="teamneeds-depth-rank">{index + 1}</span>
                    <span className="teamneeds-depth-pos">
                      <strong>{need.position}</strong>
                      <small>{config.label}</small>
                    </span>

                    <div className="teamneeds-depth-layers">
                      <DepthLayer
                        label="NHL"
                        current={need.nhlCount}
                        target={config.minNhl}
                        tone={need.tone}
                      />
                      <DepthLayer
                        label="Org"
                        current={need.orgCount}
                        target={config.minOrg}
                        tone={need.tone}
                      />
                      <DepthLayer
                        label="Pipe"
                        current={need.pipelineCount}
                        target={2}
                        tone={need.tone}
                      />
                    </div>

                    <div className="teamneeds-depth-context">
                      <strong>{need.headline}</strong>
                      <p>{formatNeedContext(need)}</p>
                    </div>

                    <span className="teamneeds-depth-score">{Math.round(need.needScore)}</span>
                  </button>
                );
              })}
            </div>
          </section>

          <aside className="teamneeds-right-rail">
            <section className={`teamneeds-card teamneeds-spotlight tone-${activeNeed?.tone || "medium"}`}>
              <header className="teamneeds-card-header">
                <div>
                  <p>Selected need</p>
                  <h3>{activeNeed?.draftLabel || "Draft strategy"}</h3>
                </div>
                <span className="teamneeds-context-badge">
                  {activeNeed ? formatNeedContext(activeNeed) : "—"}
                </span>
              </header>

              <div className="teamneeds-spotlight-body">
                <h4>{activeNeed?.archetype || "Best player available with positional awareness"}</h4>

                <p>{activeNeed?.explanation || "The roster does not show a severe weakness here, but the draft board should still be monitored."}</p>

                {activeNeed ? (
                  <div className="teamneeds-depth-layers teamneeds-depth-layers--detail">
                    <DepthLayer
                      label="NHL"
                      current={activeNeed.nhlCount}
                      target={POSITION_BUCKETS[activeNeed.position]?.minNhl}
                      tone={activeNeed.tone}
                    />
                    <DepthLayer
                      label="Org"
                      current={activeNeed.orgCount}
                      target={POSITION_BUCKETS[activeNeed.position]?.minOrg}
                      tone={activeNeed.tone}
                    />
                    <DepthLayer
                      label="Pipe"
                      current={activeNeed.pipelineCount}
                      target={2}
                      tone={activeNeed.tone}
                    />
                  </div>
                ) : null}

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
                  <span>{formatNeedContext(row)}</span>
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

function ContextCell({ code, label, value, sub, tone }) {
  return (
    <article className={`teamneeds-context-cell${tone ? ` tone-${tone}` : ""}`}>
      <span className="teamneeds-context-cell__icon" aria-hidden="true">
        {code || "OPS"}
      </span>
      <div>
        <span className="teamneeds-context-cell__label">{label}</span>
        <strong className="teamneeds-context-cell__value">{value ?? "—"}</strong>
        <small className="teamneeds-context-cell__sub">{sub || "—"}</small>
      </div>
    </article>
  );
}

function DepthLayer({ label, current, target, tone = "medium" }) {
  const safeCurrent = Number(current) || 0;
  const safeTarget = Number(target) || 1;
  const pct = Math.min(100, Math.round((safeCurrent / safeTarget) * 100));
  const gap = Math.max(0, safeTarget - safeCurrent);

  return (
    <div className={`teamneeds-depth-layer tone-${tone}`}>
      <div className="teamneeds-depth-layer__head">
        <span>{label}</span>
        <strong>
          {safeCurrent}
          <em>/{safeTarget}</em>
        </strong>
      </div>
      <div className="teamneeds-depth-layer__track" aria-hidden="true">
        <i style={{ width: `${pct}%` }} />
      </div>
      {gap > 0 ? <small>{gap} below target</small> : <small>At target</small>}
    </div>
  );
}

function formatNeedContext(need) {
  if (!need) return "—";
  const config = POSITION_BUCKETS[need.position] || {};
  const nhlGap = Math.max(0, (config.minNhl || 0) - (need.nhlCount || 0));
  const orgGap = Math.max(0, (config.minOrg || 0) - (need.orgCount || 0));
  const parts = [];

  if (nhlGap > 0) parts.push(`${nhlGap} below NHL min`);
  if (orgGap > 0) parts.push(`${orgGap} below org min`);
  if ((need.pipelineCount || 0) <= 1) parts.push("thin pipeline");
  if (need.avgOverall > 0 && need.avgOverall < (config.idealAvg || 79)) {
    parts.push(`${formatNumber(config.idealAvg - need.avgOverall, 1)} OVR below target`);
  }
  if (!parts.length) {
    if (need.severity === "STRENGTH") return "Organizational strength";
    return `${need.nhlCount || 0} NHL · ${formatNumber(need.avgOverall, 1)} avg OVR`;
  }
  return parts.slice(0, 2).join(" · ");
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
        --bg: var(--ops-navy);
        --bg-2: var(--ops-black);
        --panel: var(--ops-panel);
        --panel-2: var(--ops-panel-2);
        --line: var(--ops-grid);
        --line-2: var(--ops-grid-2);
        --line-strong: var(--ops-grid-strong);
        --text: var(--ops-text);
        --muted: var(--ops-text-secondary);
        --cyan: var(--ops-cyan);
        --cyan-soft: var(--ops-cyan-soft);
        --gold: var(--ops-gold);
        --gold-soft: var(--ops-gold-soft);
        --green: var(--ops-success);
        --green-soft: var(--ops-success-soft);
        --red: var(--ops-injury);
        --red-soft: var(--ops-injury-soft);
        --blue: var(--ops-info);
        --blue-soft: var(--ops-info-soft);
        --shadow: var(--depth-overlay);

        min-height: 100vh;
        width: 100%;
        background:
          radial-gradient(circle at 18% 0%, rgba(19, 216, 231, 0.08), transparent 28%),
          linear-gradient(180deg, var(--ops-black) 0%, var(--ops-navy-deep) 100%);
        color: var(--text);
        display: grid;
        grid-template-columns: 94px minmax(0, 1fr);
        overflow: hidden;
        font-family: var(--font-ops-ui);
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
        transition:
          color 140ms ease,
          background-color 140ms ease,
          border-color 140ms ease;
      }

      .teamneeds-side-button:hover {
        color: var(--text);
        background: rgba(255, 255, 255, 0.035);
      }

      /* Command rail matches the network standard: hard rail, notched plate. */
      .teamneeds-side-button.is-active {
        color: var(--cyan);
        background: linear-gradient(90deg, rgba(19, 216, 231, 0.16), rgba(19, 216, 231, 0.02));
        clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%);
      }

      .teamneeds-side-button.is-active::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        border-radius: 0;
        background: var(--cyan);
        box-shadow: none;
      }

      .teamneeds-side-button span {
        font-size: 22px;
      }

      .teamneeds-side-button small {
        font-size: 11px;
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
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(250px, 1fr) minmax(220px, 1fr) auto;
        align-items: center;
        gap: 18px;
      }

      .teamneeds-team-identity {
        display: flex;
        align-items: center;
        gap: 18px;
        min-width: 0;
      }

      .teamneeds-team-logo {
        width: 54px;
        height: 54px;
        border-radius: var(--radius-hud, 4px);
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
        font-size: clamp(20px, 1.7vw, 26px);
        line-height: 1;
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

      /* Screen identity lives on the left; the report name is a document
         subject line, not a second display headline competing with it. */
      .teamneeds-draft-board-title {
        text-align: left;
        padding-left: 16px;
        border-left: 2px solid rgba(19, 216, 231, 0.35);
      }

      .teamneeds-draft-board-title h2 {
        margin: 0;
        font-size: 13px;
        line-height: 1.2;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: rgba(214, 234, 244, 0.9);
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
        border-radius: var(--radius-hud, 4px);
        background: rgba(12, 31, 47, 0.72);
        color: var(--text);
        padding: 8px 14px;
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
      }

      .teamneeds-ops-context {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        border: 1px solid var(--line);
        border-radius: var(--radius-card);
        overflow: hidden;
        margin-top: var(--space-4);
        background: rgba(6, 21, 34, 0.72);
      }

      .teamneeds-context-cell {
        min-height: 72px;
        padding: var(--space-3);
        display: flex;
        align-items: flex-start;
        gap: var(--space-2);
        border-right: 1px solid rgba(156, 218, 236, 0.08);
      }

      .teamneeds-context-cell:last-child {
        border-right: 0;
      }

      .teamneeds-context-cell__icon {
        flex: 0 0 auto;
        min-width: 28px;
        padding: 3px 5px;
        margin-top: 1px;
        border: 1px solid var(--line);
        border-radius: var(--radius-ops, 2px);
        color: var(--cyan);
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 0.08em;
        line-height: 1.2;
        text-align: center;
      }

      .teamneeds-context-cell__label,
      .teamneeds-context-cell__sub {
        display: block;
      }

      .teamneeds-context-cell__label {
        color: var(--muted);
        font-size: var(--type-phase-label-size);
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }

      .teamneeds-context-cell__value {
        display: block;
        margin-top: 2px;
        font-size: var(--type-compact-size);
        font-weight: 800;
        line-height: 1.2;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .teamneeds-context-cell__sub {
        margin-top: 4px;
        color: var(--muted);
        font-size: var(--type-table-meta-size);
        font-weight: 700;
        line-height: 1.35;
      }

      .teamneeds-context-cell.tone-critical .teamneeds-context-cell__icon { color: var(--red); }
      .teamneeds-context-cell.tone-high .teamneeds-context-cell__icon { color: var(--gold); }
      .teamneeds-context-cell.tone-medium .teamneeds-context-cell__icon { color: var(--blue); }
      .teamneeds-context-cell.tone-low .teamneeds-context-cell__icon,
      .teamneeds-context-cell.tone-strength .teamneeds-context-cell__icon { color: var(--green); }

      .teamneeds-content-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.4fr) minmax(360px, 0.75fr);
        gap: 22px;
        margin-top: 22px;
      }

      .teamneeds-board-panel,
      .teamneeds-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: var(--radius-card);
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

      .teamneeds-depth-board {
        padding: var(--space-2) var(--space-3) var(--space-3);
      }

      .teamneeds-depth-head,
      .teamneeds-depth-row {
        display: grid;
        grid-template-columns: 36px 72px minmax(220px, 1.2fr) minmax(180px, 1fr) 52px;
        align-items: center;
        gap: var(--space-3);
      }

      .teamneeds-depth-head {
        min-height: 32px;
        padding: 0 var(--space-2);
        color: var(--muted);
        font-size: var(--type-phase-label-size);
        font-weight: 900;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-bottom: 1px solid var(--line);
      }

      .teamneeds-depth-row {
        width: 100%;
        min-height: 74px;
        margin-top: var(--space-1);
        padding: var(--space-2);
        border: 1px solid rgba(156, 218, 236, 0.08);
        border-radius: var(--radius-control);
        background: rgba(6, 21, 34, 0.45);
        color: var(--text);
        text-align: left;
        cursor: pointer;
        transition: background var(--motion-micro), border-color var(--motion-micro);
      }

      .teamneeds-depth-row:hover,
      .teamneeds-depth-row.is-active {
        border-color: var(--line-strong);
        background: var(--cyan-soft);
      }

      .teamneeds-depth-rank {
        font-family: var(--font-mono-data);
        font-size: var(--type-table-meta-size);
        font-weight: 800;
        color: var(--muted);
      }

      .teamneeds-depth-pos strong,
      .teamneeds-depth-pos small {
        display: block;
      }

      .teamneeds-depth-pos strong {
        font-size: var(--type-compact-size);
        letter-spacing: 0.06em;
      }

      .teamneeds-depth-pos small {
        margin-top: 2px;
        color: var(--muted);
        font-size: var(--type-table-meta-size);
      }

      .teamneeds-depth-layers {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: var(--space-2);
      }

      .teamneeds-depth-layers--detail {
        margin: var(--space-3) 0;
      }

      .teamneeds-depth-layer__head {
        display: flex;
        justify-content: space-between;
        gap: var(--space-1);
        font-size: var(--type-table-meta-size);
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
      }

      .teamneeds-depth-layer__head strong {
        color: var(--text);
        font-family: var(--font-mono-data);
      }

      .teamneeds-depth-layer__head em {
        font-style: normal;
        color: var(--muted);
        font-weight: 700;
      }

      /* Department signature: the organizational depth stack. Each layer is
         divided into roster slots so a shortage reads as missing bodies,
         not as a percentage bar. */
      .teamneeds-depth-layer__track {
        position: relative;
        height: 6px;
        margin-top: 4px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 0;
        overflow: hidden;
      }

      .teamneeds-depth-layer__track::after {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: repeating-linear-gradient(
          90deg,
          transparent 0 calc(20% - 1px),
          rgba(4, 16, 26, 0.9) calc(20% - 1px) 20%
        );
      }

      .teamneeds-depth-layer__track i {
        display: block;
        height: 100%;
        background: var(--cyan);
        border-radius: 0;
      }

      .teamneeds-depth-layer small {
        display: block;
        margin-top: 3px;
        color: var(--muted);
        font-size: 0.6875rem;
        font-weight: 700;
      }

      .teamneeds-depth-layer.tone-critical .teamneeds-depth-layer__track i { background: var(--red); }
      .teamneeds-depth-layer.tone-high .teamneeds-depth-layer__track i { background: var(--gold); }
      .teamneeds-depth-layer.tone-medium .teamneeds-depth-layer__track i { background: var(--blue); }
      .teamneeds-depth-layer.tone-low .teamneeds-depth-layer__track i,
      .teamneeds-depth-layer.tone-strength .teamneeds-depth-layer__track i { background: var(--green); }

      .teamneeds-depth-context strong,
      .teamneeds-depth-context p {
        display: block;
        margin: 0;
      }

      .teamneeds-depth-context strong {
        font-size: var(--type-compact-size);
        line-height: 1.25;
      }

      .teamneeds-depth-context p {
        margin-top: 3px;
        color: var(--muted);
        font-size: var(--type-table-meta-size);
        line-height: 1.35;
      }

      .teamneeds-depth-score {
        text-align: right;
        font-family: var(--font-mono-data);
        font-size: var(--type-compact-size);
        font-weight: 800;
      }

      .teamneeds-context-badge {
        max-width: 220px;
        text-align: right;
        color: var(--muted);
        font-size: var(--type-table-meta-size);
        font-weight: 800;
        line-height: 1.35;
      }

      .teamneeds-right-rail {
        display: grid;
        gap: 22px;
        align-content: start;
      }

      .teamneeds-spotlight-body {
        padding: 22px;
      }

      .teamneeds-spotlight-body h4 {
        margin: 0;
        font-size: var(--type-compact-size);
        line-height: 1.35;
        font-weight: 800;
      }

      .teamneeds-spotlight-body p {
        margin: 12px 0 0;
        color: var(--muted);
        line-height: 1.55;
        font-size: 13px;
        font-weight: 750;
      }

      /* Position facts are a ruled register, not six equal metric cards. */
      .teamneeds-detail-grid {
        margin-top: 12px;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        column-gap: 18px;
        border-top: 1px solid rgba(255, 255, 255, 0.09);
      }

      .teamneeds-detail-grid article {
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: baseline;
        gap: 10px;
        padding: 5px 0;
        border-radius: 0;
        background: transparent;
        border: 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
      }

      .teamneeds-detail-grid span {
        display: block;
        color: var(--muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 1000;
      }

      .teamneeds-detail-grid strong {
        display: block;
        margin-top: 0;
        font-size: 14px;
        font-variant-numeric: tabular-nums;
      }

      .teamneeds-target-list {
        padding: 6px 12px 10px;
        display: grid;
        gap: 0;
      }

      .teamneeds-target-row {
        min-height: 40px;
        display: grid;
        grid-template-columns: 38px minmax(0, 1fr) 46px;
        align-items: center;
        gap: 10px;
        padding: 6px 8px;
        border-radius: 0;
        border: 0;
        border-bottom: 1px solid rgba(156, 218, 236, 0.09);
        background: transparent;
      }

      /* Priority index reads as a stencilled position plate. */
      .teamneeds-target-row > span {
        width: 32px;
        height: 24px;
        display: grid;
        place-items: center;
        border-radius: var(--radius-ops, 2px);
        border: 1px solid rgba(19, 216, 231, 0.3);
        background: rgba(19, 216, 231, 0.08);
        color: var(--cyan);
        font-weight: 1000;
        font-variant-numeric: tabular-nums;
        letter-spacing: 0.04em;
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
        font-size: 11px;
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
        font-size: 11px;
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
        border-radius: 6px;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(156, 218, 236, 0.09);
      }

      .teamneeds-plan-card > span {
        width: 42px;
        height: 42px;
        border-radius: 10px;
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
        padding: 14px 16px;
        border: 1px dashed var(--line);
        border-radius: var(--radius-control, 6px);
        background: rgba(6, 21, 34, 0.55);
        color: var(--muted);
        line-height: 1.5;
        font-size: 13px;
        font-weight: 800;
      }

      .teamneeds-empty::before {
        content: "DEPTH BOARD · STANDBY";
        display: block;
        margin-bottom: 6px;
        color: var(--cyan);
        font-size: 11px;
        font-weight: 900;
        letter-spacing: 0.14em;
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
        .teamneeds-ops-context,
        .teamneeds-depth-layers,
        .teamneeds-detail-grid {
          grid-template-columns: 1fr;
        }

        .teamneeds-context-cell {
          border-right: 0;
          border-bottom: 1px solid rgba(156, 218, 236, 0.08);
        }

        .teamneeds-context-cell:last-child {
          border-bottom: 0;
        }

        .teamneeds-depth-head,
        .teamneeds-depth-row {
          grid-template-columns: 1fr;
          gap: var(--space-2);
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
          border-radius: 6px;
        }

        .teamneeds-team-logo span {
          font-size: 18px;
        }

        .teamneeds-team-identity h1,
        .teamneeds-draft-board-title h2 {
          font-size: 30px;
        }
      }
    `}</style>
  );
}

export default TeamNeeds;