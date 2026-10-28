import React, { useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { resolveFranchiseTeamLogo } from "../utils/teamLogos";

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
          <SideButton icon="▦" label="Office" onClick={() => handleNavigate(SCREENS.HUB)} />
          <SideButton icon="◫" label="Calendar" onClick={() => handleNavigate(SCREENS.CALENDAR)} />
          <SideButton icon="◉" label="Roster" onClick={() => handleNavigate(SCREENS.ROSTER)} />
          <SideButton active icon="◈" label="Needs" onClick={() => handleNavigate(SCREENS.TEAM_NEEDS)} />
          <SideButton icon="◎" label="Scouting" onClick={() => handleNavigate(SCREENS.SCOUTING)} />
          <SideButton icon="◌" label="Lottery" onClick={() => handleNavigate(SCREENS.DRAFT_LOTTERY)} />
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
            icon="◉"
            label="Roster"
            value={report.totalRoster || "—"}
            sub={`${report.nhlRosterCount} NHL · ${report.prospectCount} pipeline`}
          />
          <ContextCell
            icon="◎"
            label="Priority"
            value={report.biggestNeed?.position || "—"}
            sub={report.biggestNeed ? formatNeedContext(report.biggestNeed) : "No roster data loaded"}
            tone={report.biggestNeed?.tone}
          />
          <ContextCell
            icon="☰"
            label="Strategy"
            value={report.strategyLabel}
            sub={report.strategySub}
          />
          <ContextCell
            icon="⌖"
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

function ContextCell({ icon, label, value, sub, tone }) {
  return (
    <article className={`teamneeds-context-cell${tone ? ` tone-${tone}` : ""}`}>
      <span className="teamneeds-context-cell__icon" aria-hidden="true">
        {icon || "◈"}
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
  const name = getTeamDisplayName(team);
  const logo = resolveFranchiseTeamLogo(team, name);

  return (
    <div className="teamneeds-team-logo" aria-label={`${name} logo`}>
      {logo ? <img src={logo} alt="" /> : <span>{getTeamAbbreviation(team)}</span>}
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
    null
  );
}

export default TeamNeeds;