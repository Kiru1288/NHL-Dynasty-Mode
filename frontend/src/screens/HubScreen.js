import React, { useCallback, useEffect, useMemo, useRef } from "react";
import { useGameUI } from "../game/GameUIContext";
import { HUB_MENU, franchiseFeedText } from "../game/constants";
import { GameFooter } from "../components/game/GameFooter";

const menuIndexById = (id, fallback = 0) => {
  const idx = HUB_MENU.findIndex((m) => String(m.id).toLowerCase() === String(id).toLowerCase());
  return idx >= 0 ? idx : fallback;
};

const ROSTER_HUB_INDEX = menuIndexById("roster", 0);
const CALENDAR_HUB_INDEX = menuIndexById("calendar", 1);
const SCOUTING_HUB_INDEX = menuIndexById("scouting", 2);
const TRADE_HUB_INDEX = menuIndexById("trades", menuIndexById("trade", 3));
const DRAFT_HUB_INDEX = menuIndexById("draft", menuIndexById("draftClass", 4));
const STATS_HUB_INDEX = menuIndexById("stats", 5);
const SETTINGS_HUB_INDEX = menuIndexById("settings", 6);

function safeNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function fmtMoney(value) {
  if (value == null || value === "—") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  if (Math.abs(n) >= 1000000) return `$${(n / 1000000).toFixed(2)}M`;
  return `$${n.toFixed(2)}M`;
}

function fmtRecord(record) {
  if (!record) return "0-0-0";
  return `${record.w ?? 0}-${record.l ?? 0}-${record.otl ?? 0}`;
}

function fmtScoreLine(g) {
  if (!g) return "—";
  const ot = g.overtime ? " OT" : "";
  return `${g.home_goals ?? 0}-${g.away_goals ?? 0}${ot}`;
}

function classifyNotificationLine(line) {
  const t = line && typeof line === "object" ? String(line.type || "").toLowerCase() : "";
  if (t === "injury" || t === "player_story") return "alert";

  const u = franchiseFeedText(line).toUpperCase();
  if (u.includes("TRADE") || u.includes("ACQUIRE")) return "trade";
  if (u.includes("INJURY") || u.includes("WJC") || u.includes("SUSPEND")) return "alert";
  if (u.includes("PLAYOFF") || u.includes("CUP") || u.includes("CHAMPION")) return "major";
  return "default";
}

function TeamLogo({ team, size = "lg" }) {
  const logo =
    team?.logo ||
    team?.logo_url ||
    team?.logoUrl ||
    team?.team_logo ||
    team?.image ||
    team?.crest ||
    "";

  const name = team?.name || team?.team_name || "TEAM";
  const initials = String(name)
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0])
    .join("")
    .toUpperCase();

  if (logo) {
    return (
      <div className={`exec-logo exec-logo--${size}`}>
        <img src={logo} alt={name} />
      </div>
    );
  }

  return (
    <div className={`exec-logo exec-logo--${size} exec-logo--fallback`} aria-label={name}>
      {initials || "NHL"}
    </div>
  );
}

function MiniTeamMark({ game, side = "home" }) {
  const name = side === "away" ? game?.away_name : game?.home_name;
  const logo = side === "away" ? game?.away_logo || game?.away_logo_url : game?.home_logo || game?.home_logo_url;

  if (logo) {
    return (
      <span className="exec-mini-mark">
        <img src={logo} alt={name || side} />
      </span>
    );
  }

  return <span className="exec-mini-mark exec-mini-mark--fallback">{String(name || "?").slice(0, 1)}</span>;
}

function getCapSpace(team, franchiseState) {
  const explicit =
    team?.cap_space ??
    team?.capSpace ??
    franchiseState?.cap_space ??
    franchiseState?.team_cap_space ??
    franchiseState?.cap?.space;

  if (explicit != null) return explicit;

  const limit = safeNumber(team?.cap_limit ?? team?.salary_cap ?? franchiseState?.salary_cap, NaN);
  const hit = safeNumber(team?.cap_hit ?? team?.payroll ?? franchiseState?.cap_hit, NaN);
  if (Number.isFinite(limit) && Number.isFinite(hit)) return limit - hit;

  return null;
}

function getCapHit(team, franchiseState) {
  return team?.cap_hit ?? team?.capHit ?? franchiseState?.cap_hit ?? franchiseState?.cap?.hit ?? null;
}

function getMorale(franchiseState, team) {
  const raw =
    franchiseState?.team_morale ??
    team?.morale ??
    franchiseState?.morale ??
    franchiseState?.chemistry ??
    franchiseState?.team_chemistry;

  if (raw == null) return { label: "Confident", value: 72 };

  if (typeof raw === "string") return { label: raw, value: 72 };

  const n = safeNumber(raw, 72);
  let label = "Confident";
  if (n >= 85) label = "Excellent";
  else if (n >= 65) label = "Confident";
  else if (n >= 45) label = "Uneasy";
  else label = "Frustrated";

  return { label, value: Math.max(0, Math.min(100, n)) };
}

function getFanInterest(franchiseState, team) {
  const raw = franchiseState?.fan_interest ?? team?.fan_interest ?? franchiseState?.market_interest;
  const n = safeNumber(raw, 82);

  let label = "High";
  if (n >= 85) label = "Elite";
  else if (n >= 70) label = "High";
  else if (n >= 45) label = "Medium";
  else label = "Low";

  return { label, value: Math.max(0, Math.min(100, n)) };
}

function getStandingsLine(franchiseState, team) {
  const standings = franchiseState?.standings || [];
  const myId = String(team?.id ?? franchiseState?.user_team_id ?? "");
  const row = standings.find((r) => String(r.team_id ?? r.id) === myId);

  if (row?.division_rank && row?.division) {
    return `${row.division_rank}${ordinalSuffix(row.division_rank)} ${row.division}`;
  }

  if (row?.conference_rank && row?.conference) {
    return `${row.conference_rank}${ordinalSuffix(row.conference_rank)} ${row.conference}`;
  }

  if (team?.standings_position) return team.standings_position;
  if (franchiseState?.standings_summary) return franchiseState.standings_summary;

  return "Standings";
}

function ordinalSuffix(n) {
  const num = Number(n);
  if (!Number.isFinite(num)) return "";
  const mod100 = num % 100;
  if (mod100 >= 11 && mod100 <= 13) return "th";
  const mod10 = num % 10;
  if (mod10 === 1) return "st";
  if (mod10 === 2) return "nd";
  if (mod10 === 3) return "rd";
  return "th";
}

function phaseLabel(franchiseState) {
  const nhlToday = franchiseState?.nhl_today || {};
  const ph = franchiseState?.phase;

  if (ph === "complete") return "Season Complete";
  if (ph === "regular" && nhlToday?.ui_phase) return String(nhlToday.ui_phase);
  if (ph === "regular") return "Regular Season";
  if (!ph) return "—";

  return String(ph)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (m) => m.toUpperCase());
}

function getNextLeagueDay(franchiseState) {
  return (
    franchiseState?.calendar_summary ||
    franchiseState?.next_league_day ||
    franchiseState?.nhl_today?.date_label ||
    franchiseState?.nhl_today?.iso ||
    "—"
  );
}

function getUpcomingGames(franchiseState) {
  const blocks = franchiseState?.schedule_upcoming || [];
  const games = [];

  for (const block of blocks) {
    for (const game of block.games || []) {
      games.push({
        ...game,
        date: block.display_date || block.date_label || block.iso || block.calendar_index || "Upcoming",
        user_plays: block.user_plays,
      });
      if (games.length >= 3) return games;
    }
  }

  return games;
}

function getTeamLeaders(franchiseState, team) {
  const stats = franchiseState?.stats_central || {};
  const myId = String(team?.id ?? franchiseState?.user_team_id ?? "");

  const players =
    stats.team_leaders ||
    stats.user_team_leaders ||
    stats.roster_leaders ||
    stats.league_leaders ||
    franchiseState?.team_leaders ||
    [];

  const filtered = Array.isArray(players)
    ? players.filter((p) => {
        const pTeam = String(p.team_id ?? p.team ?? p.teamId ?? "");
        return !myId || !pTeam || pTeam === myId || pTeam === String(team?.name);
      })
    : [];

  const byGoals = [...filtered].sort((a, b) => safeNumber(b.goals ?? b.g, 0) - safeNumber(a.goals ?? a.g, 0))[0];
  const byAssists = [...filtered].sort((a, b) => safeNumber(b.assists ?? b.a, 0) - safeNumber(a.assists ?? a.a, 0))[0];
  const bySave =
    [...filtered]
      .filter((p) => p.save_pct != null || p.sv_pct != null || p.position === "G")
      .sort((a, b) => safeNumber(b.save_pct ?? b.sv_pct, 0) - safeNumber(a.save_pct ?? a.sv_pct, 0))[0] || null;

  return [
    {
      label: "Goals",
      player: byGoals?.name || byGoals?.player_name || "—",
      value: byGoals ? safeNumber(byGoals.goals ?? byGoals.g, 0) : "—",
    },
    {
      label: "Assists",
      player: byAssists?.name || byAssists?.player_name || "—",
      value: byAssists ? safeNumber(byAssists.assists ?? byAssists.a, 0) : "—",
    },
    {
      label: "Save %",
      player: bySave?.name || bySave?.player_name || "—",
      value: bySave ? Number(bySave.save_pct ?? bySave.sv_pct ?? 0).toFixed(3).replace(/^0/, "") : "—",
    },
  ];
}

function getLeaguePulse(franchiseState) {
  const notifications = franchiseState?.notifications || [];
  const timeline = franchiseState?.timeline || [];
  const all = [...notifications, ...timeline].slice(-20).reverse();

  const trade = all.find((x) => franchiseFeedText(x).toUpperCase().includes("TRADE"));
  const injury = all.find((x) => franchiseFeedText(x).toUpperCase().includes("INJURY"));
  const story = all.find((x) => franchiseFeedText(x).length > 8);

  const text = franchiseFeedText(trade || injury || story || "");
  if (text) return text;

  return "League activity will appear here as the season develops.";
}

function DashboardButton({ active, icon, label, sub, onClick }) {
  return (
    <button type="button" className={`exec-menu-card ${active ? "is-active" : ""}`} onClick={onClick}>
      <span className="exec-menu-card__icon">{icon}</span>
      <span className="exec-menu-card__label">{label}</span>
      {sub ? <span className="exec-menu-card__sub">{sub}</span> : null}
    </button>
  );
}

export function HubScreen() {
  const {
    franchiseState,
    hubMenuIndex,
    setHubMenuIndex,
    openHubMenu,
    error,
    onAdvanceDay,
    onAdvanceFranchise,
    advancing,
    onResolveDecision,
    onResolveStorylineChoice,
    refreshFranchise,
    gmName,
    setGmName,
  } = useGameUI();

  const team = franchiseState?.team || {};
  const rec = team?.record;
  const capSpace = getCapSpace(team, franchiseState);
  const capHit = getCapHit(team, franchiseState);
  const morale = getMorale(franchiseState, team);
  const fanInterest = getFanInterest(franchiseState, team);
  const standingsLine = getStandingsLine(franchiseState, team);
  const upcomingGames = useMemo(() => getUpcomingGames(franchiseState), [franchiseState]);
  const teamLeaders = useMemo(() => getTeamLeaders(franchiseState, team), [franchiseState, team]);
  const pulse = useMemo(() => getLeaguePulse(franchiseState), [franchiseState]);

  const pending = franchiseState?.pending_decisions || [];
  const storylineChoices = franchiseState?.storyline_choices || [];
  const notificationLines = useMemo(() => (franchiseState?.notifications || []).slice(-8).reverse(), [franchiseState]);

  const sortedPending = useMemo(() => {
    const copy = [...pending];
    copy.sort((a, b) => {
      const pa = String(a?.priority || "").toUpperCase() === "CRITICAL" ? 0 : 1;
      const pb = String(b?.priority || "").toUpperCase() === "CRITICAL" ? 0 : 1;
      return pa - pb;
    });
    return copy;
  }, [pending]);

  const activeRef = useRef(hubMenuIndex);
  activeRef.current = hubMenuIndex;

  const stations = useMemo(
    () => [
      { idx: ROSTER_HUB_INDEX, label: "Roster", sub: "Lines / contracts", icon: "👥" },
      { idx: CALENDAR_HUB_INDEX, label: "Calendar", sub: "Schedule / results", icon: "🗓️", featured: true },
      { idx: SCOUTING_HUB_INDEX, label: "Scouting / Intel", sub: "League targets", icon: "🔭" },
      { idx: TRADE_HUB_INDEX, label: "Trade Floor", sub: "Offers / market", icon: "↔" },
      { idx: DRAFT_HUB_INDEX, label: "Draft Class", sub: "Prospects", icon: "🏆" },
      { idx: SETTINGS_HUB_INDEX, label: "GM Office", sub: "Staff / systems", icon: "💺" },
      { idx: STATS_HUB_INDEX, label: "Systems", sub: "Stats central", icon: "⚙" },
    ],
    []
  );

  const openStation = useCallback(
    (idx) => {
      setHubMenuIndex(idx);
      openHubMenu(idx);
    },
    [openHubMenu, setHubMenuIndex]
  );

  useEffect(() => {
    function onKey(e) {
      if (e.target?.matches?.("input, textarea, select, button")) return;

      const currentStationIndex = stations.findIndex((s) => s.idx === activeRef.current);

      if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        const next = stations[Math.max(0, currentStationIndex - 1)] || stations[0];
        setHubMenuIndex(next.idx);
      }

      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        const next = stations[Math.min(stations.length - 1, currentStationIndex + 1)] || stations[0];
        setHubMenuIndex(next.idx);
      }

      if (e.key === "Enter") {
        e.preventDefault();
        openHubMenu(activeRef.current);
      }
    }

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [openHubMenu, setHubMenuIndex, stations]);

  const currentFocus =
    franchiseState?.current_focus ||
    franchiseState?.focus ||
    team?.focus ||
    "Improve special teams performance";

  const nextObjective =
    franchiseState?.next_objective ||
    franchiseState?.objective ||
    team?.objective ||
    "Increase team chemistry";

  const canAdvance = Boolean(franchiseState?.flags?.can_advance);

  return (
    <div className="game-screen exec-root">
      <HubExecStyles />

      <div className="exec-bg" aria-hidden>
        <div className="exec-bg__orb exec-bg__orb--one" />
        <div className="exec-bg__orb exec-bg__orb--two" />
        <div className="exec-bg__rink" />
        <div className="exec-bg__grid" />
      </div>

      <section className="exec-shell">
        <header className="exec-topbar">
          <div className="exec-brand">
            <TeamLogo team={team} size="xl" />
            <div className="exec-brand__copy">
              <h1>{team?.name || "Franchise Club"}</h1>
              <p>Hockey Exec Game Hub</p>
            </div>
          </div>

          <div className="exec-top-stat">
            <span className="exec-stat-icon">▣</span>
            <div>
              <span>Season / Date</span>
              <strong>YR {franchiseState?.season_year ?? "2025"} | {getNextLeagueDay(franchiseState)}</strong>
              <small>{phaseLabel(franchiseState)}</small>
            </div>
          </div>

          <div className="exec-top-stat">
            <span className="exec-stat-icon">♕</span>
            <div>
              <span>Record</span>
              <strong>{fmtRecord(rec)}</strong>
              <small>{team?.points ?? franchiseState?.points ?? "0"} PTS | {standingsLine}</small>
            </div>
          </div>

          <div className="exec-top-stat">
            <span className="exec-stat-icon">$</span>
            <div>
              <span>Cap Status</span>
              <strong>{fmtMoney(capHit ?? capSpace)}</strong>
              <small>{fmtMoney(capSpace)} Available</small>
            </div>
          </div>

          <div className="exec-top-stat">
            <span className="exec-stat-icon">☺</span>
            <div>
              <span>Team Morale</span>
              <strong>{morale.label}</strong>
              <div className="exec-dots" aria-label={`Morale ${morale.value}`}>
                {Array.from({ length: 7 }).map((_, i) => (
                  <b key={i} className={i < Math.round((morale.value / 100) * 7) ? "is-on" : ""} />
                ))}
              </div>
            </div>
          </div>

          <div className="exec-top-stat">
            <span className="exec-stat-icon">♨</span>
            <div>
              <span>Fan Interest</span>
              <strong>{fanInterest.label}</strong>
              <small>{fanInterest.value} / 100</small>
            </div>
          </div>

          <div className="exec-profile">
            <div className="exec-avatar" aria-hidden>
              <span />
            </div>
            <div>
              <span>GM Profile</span>
              <input
                value={gmName || ""}
                onChange={(e) => setGmName(e.target.value)}
                placeholder="GM Name"
                aria-label="GM name"
              />
              <small>General Manager</small>
            </div>
          </div>
        </header>

        <main className="exec-main">
          <aside className="exec-left">
            <section className="exec-takeover-card">
              <div>
                <span className="exec-kicker">Franchise Takeover</span>
                <h2>You Control<br />The Board</h2>
                <p>Build a contender through smart decisions, strong development, and a winning culture.</p>
              </div>
              <div className="exec-takeover-card__logo">
                <TeamLogo team={team} size="lg" />
              </div>
              <div className="exec-focus-line">
                <b>Focus:</b>
                <span>{currentFocus}</span>
              </div>
            </section>

            <section className="exec-panel exec-leaders">
              <div className="exec-panel__head">
                <span>Team Leaders</span>
              </div>

              <div className="exec-leader-list">
                {teamLeaders.map((leader) => (
                  <div className="exec-leader-row" key={leader.label}>
                    <div className="exec-headshot">
                      <span>{String(leader.player || "?").slice(0, 1)}</span>
                    </div>
                    <div>
                      <strong>{leader.player}</strong>
                      <small>{leader.label}</small>
                    </div>
                    <b>{leader.value}</b>
                  </div>
                ))}
              </div>

              <button type="button" className="exec-mini-button" onClick={() => openStation(STATS_HUB_INDEX)}>
                Full Team Stats <span>›</span>
              </button>
            </section>
          </aside>

          <section className="exec-center">
            <div className="exec-menu-grid">
              {stations.map((station) => (
                <DashboardButton
                  key={station.label}
                  active={hubMenuIndex === station.idx || station.featured}
                  icon={station.icon}
                  label={station.label}
                  sub={station.sub}
                  onClick={() => openStation(station.idx)}
                />
              ))}
            </div>
          </section>

          <aside className="exec-right">
            <section className="exec-panel exec-snapshot">
              <div className="exec-panel__head">
                <span>Team Snapshot</span>
              </div>

              <div className="exec-snapshot-list">
                <div>
                  <span>Record</span>
                  <strong>{fmtRecord(rec)} ({team?.points ?? franchiseState?.points ?? 0} PTS)</strong>
                </div>
                <div>
                  <span>Cap Space</span>
                  <strong className="blue">{fmtMoney(capSpace)}</strong>
                </div>
                <div>
                  <span>Next League Day</span>
                  <strong>{getNextLeagueDay(franchiseState)}</strong>
                </div>
                <div>
                  <span>Season Phase</span>
                  <strong>{phaseLabel(franchiseState)}</strong>
                </div>
              </div>

              <button type="button" className="exec-mini-button" onClick={() => openStation(STATS_HUB_INDEX)}>
                Standings: {standingsLine} <span>›</span>
              </button>
            </section>

            <section className="exec-panel exec-schedule">
              <div className="exec-panel__head">
                <span>Upcoming Schedule</span>
              </div>

              <div className="exec-schedule-list">
                {upcomingGames.length === 0 ? (
                  <div className="exec-empty">No upcoming games queued yet.</div>
                ) : (
                  upcomingGames.map((game, i) => {
                    const isHome = String(game.home_id) === String(team?.id || franchiseState?.user_team_id);
                    const opponent = isHome ? game.away_name : game.home_name;

                    return (
                      <div className="exec-schedule-row" key={`${game.date}-${game.home_id}-${game.away_id}-${i}`}>
                        <MiniTeamMark game={game} side={isHome ? "away" : "home"} />
                        <div>
                          <small>{game.date}</small>
                          <strong>{isHome ? "vs" : "@"} {opponent || game.away_name || game.home_name || "Opponent"}</strong>
                        </div>
                        <span>{isHome ? "⌂" : "✈"}</span>
                      </div>
                    );
                  })
                )}
              </div>

              <button type="button" className="exec-mini-button" onClick={() => openStation(CALENDAR_HUB_INDEX)}>
                Full Calendar <span>›</span>
              </button>
            </section>
          </aside>
        </main>

        {error ? <div className="exec-error">{error}</div> : null}

        {sortedPending.length > 0 || storylineChoices.length > 0 ? (
          <section className="exec-decision-dock">
            {sortedPending.slice(0, 3).map((d) => (
              <div className="exec-decision-card" key={d.id}>
                <span>{String(d.priority || "Decision").toUpperCase()}</span>
                <strong>{d.title}</strong>
                <p>{d.description}</p>
                <div>
                  {(d.options || []).map((o) => (
                    <button key={o.id} type="button" onClick={() => onResolveDecision(d.id, o.id)}>
                      {o.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}

            {storylineChoices.slice(0, 2).map((s) => (
              <div className="exec-decision-card" key={s.storyline_id || s.decision_id}>
                <span>STORYLINE</span>
                <strong>{s.title}</strong>
                <p>{s.description}</p>
                <div>
                  {(s.action_options || []).slice(0, 3).map((o) => (
                    <button
                      key={o.id}
                      type="button"
                      onClick={() => onResolveStorylineChoice?.(s.storyline_id, o.id)}
                    >
                      {o.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </section>
        ) : null}

        {notificationLines.length > 0 ? (
          <section className="exec-wire-strip">
            {notificationLines.slice(0, 4).map((line, i) => (
              <div
                key={line && typeof line === "object" && line.id != null ? String(line.id) : `${i}-${franchiseFeedText(line)}`}
                className={`exec-wire-item exec-wire-item--${classifyNotificationLine(line)}`}
              >
                {franchiseFeedText(line)}
              </div>
            ))}
          </section>
        ) : null}

        <footer className="exec-action-bar">
          <div className="exec-sim-buttons">
            <button
              type="button"
              className="exec-advance"
              disabled={!canAdvance || advancing}
              onClick={onAdvanceDay}
            >
              {advancing ? "Advancing…" : "Advance Day"} <span>»</span>
            </button>

            <button
              type="button"
              disabled={franchiseState?.phase === "complete" || advancing}
              onClick={() => onAdvanceFranchise({ mode: "days", count: 7, auto_resolve: true })}
            >
              7 Days
            </button>

            <button
              type="button"
              disabled={franchiseState?.phase === "complete" || advancing}
              onClick={() => onAdvanceFranchise({ mode: "days", count: 15, auto_resolve: true })}
            >
              15 Days
            </button>

            <button
              type="button"
              disabled={franchiseState?.phase === "complete" || advancing}
              onClick={() => onAdvanceFranchise({ mode: "season", count: 1, auto_resolve: true })}
            >
              Sim Season
            </button>

            <button type="button" disabled={advancing} onClick={refreshFranchise}>
              Refresh ↻
            </button>
          </div>

          <div className="exec-league-pulse">
            <div className="exec-pulse-icon">⌁</div>
            <div>
              <span>League Pulse</span>
              <p>{pulse}</p>
            </div>
            <button type="button" onClick={() => openStation(STATS_HUB_INDEX)}>
              View League <span>›</span>
            </button>
          </div>
        </footer>

        <div className="exec-bottom-status">
          <div>
            <span>Current Focus</span>
            <strong>{currentFocus}</strong>
          </div>
          <div>
            <span>Next Objective</span>
            <strong>{nextObjective}</strong>
          </div>
          <div className="exec-shortcut">Shortcut: Press ? For Help</div>
        </div>
      </section>

      <GameFooter />
    </div>
  );
}

function HubExecStyles() {
  return (
    <style>{`
      .exec-root {
        --bg: #020812;
        --panel: rgba(7, 20, 34, 0.82);
        --panel2: rgba(12, 32, 51, 0.78);
        --panel3: rgba(9, 25, 42, 0.94);
        --line: rgba(96, 170, 232, 0.26);
        --line2: rgba(130, 197, 255, 0.4);
        --text: #f2f7ff;
        --muted: #8fa5ba;
        --muted2: #667a8d;
        --blue: #36a8ff;
        --blue2: #0f6edc;
        --orange: #ff8b22;
        --orange2: #ffb347;
        --danger: #ff5f6d;
        min-height: 100vh;
        width: 100%;
        position: relative;
        overflow: hidden;
        background:
          radial-gradient(circle at 50% 35%, rgba(0, 125, 255, 0.14), transparent 30%),
          radial-gradient(circle at 20% 10%, rgba(0, 132, 255, 0.12), transparent 26%),
          linear-gradient(180deg, #04101d 0%, #020812 52%, #02060d 100%);
        color: var(--text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      .exec-bg,
      .exec-bg__grid,
      .exec-bg__rink,
      .exec-bg__orb {
        position: absolute;
        inset: 0;
        pointer-events: none;
      }

      .exec-bg__grid {
        opacity: 0.18;
        background-image:
          linear-gradient(rgba(70, 165, 255, 0.12) 1px, transparent 1px),
          linear-gradient(90deg, rgba(70, 165, 255, 0.12) 1px, transparent 1px);
        background-size: 42px 42px;
        mask-image: radial-gradient(circle at center, black, transparent 75%);
      }

      .exec-bg__rink {
        width: 920px;
        height: 470px;
        left: 50%;
        top: 53%;
        transform: translate(-50%, -50%);
        border: 1px solid rgba(72, 148, 225, 0.14);
        border-radius: 50%;
        box-shadow:
          inset 0 0 80px rgba(35, 123, 220, 0.08),
          0 0 120px rgba(35, 123, 220, 0.08);
      }

      .exec-bg__rink::before,
      .exec-bg__rink::after {
        content: "";
        position: absolute;
        inset: 52px 120px;
        border: 1px solid rgba(72, 148, 225, 0.08);
        border-radius: 50%;
      }

      .exec-bg__rink::after {
        inset: 112px 235px;
      }

      .exec-bg__orb--one {
        width: 340px;
        height: 340px;
        left: 12%;
        top: 12%;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0, 132, 255, 0.18), transparent 68%);
        filter: blur(6px);
      }

      .exec-bg__orb--two {
        width: 300px;
        height: 300px;
        right: 10%;
        bottom: 16%;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255, 128, 24, 0.1), transparent 70%);
        filter: blur(8px);
      }

      .exec-shell {
        position: relative;
        z-index: 1;
        min-height: 100vh;
        padding: 18px 34px 24px;
        display: grid;
        grid-template-rows: auto 1fr auto auto;
        gap: 22px;
      }

      .exec-topbar {
        min-height: 126px;
        display: grid;
        grid-template-columns: 1.6fr 1.05fr 0.9fr 0.95fr 0.95fr 0.95fr 1fr;
        align-items: center;
        gap: 0;
        border: 1px solid rgba(69, 154, 240, 0.35);
        border-radius: 18px;
        background:
          linear-gradient(90deg, rgba(6, 17, 30, 0.98), rgba(4, 14, 25, 0.92)),
          radial-gradient(circle at 5% 0%, rgba(56, 165, 255, 0.18), transparent 34%);
        box-shadow:
          0 24px 70px rgba(0, 0, 0, 0.38),
          inset 0 1px 0 rgba(255, 255, 255, 0.04);
        overflow: hidden;
      }

      .exec-brand,
      .exec-top-stat,
      .exec-profile {
        min-height: 96px;
        padding: 18px 24px;
        display: flex;
        align-items: center;
        gap: 18px;
        border-right: 1px solid rgba(100, 150, 200, 0.18);
      }

      .exec-brand__copy h1 {
        margin: 0;
        font-size: 24px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        line-height: 1.05;
      }

      .exec-brand__copy p {
        margin: 9px 0 0;
        color: var(--blue);
        text-transform: uppercase;
        letter-spacing: 0.34em;
        font-weight: 800;
        font-size: 12px;
      }

      .exec-logo {
        flex: 0 0 auto;
        display: grid;
        place-items: center;
        border-radius: 50%;
        background:
          radial-gradient(circle at 50% 35%, rgba(255, 255, 255, 0.16), transparent 22%),
          radial-gradient(circle, rgba(41, 152, 255, 0.22), rgba(5, 17, 31, 0.9) 62%);
        border: 1px solid rgba(94, 172, 255, 0.36);
        box-shadow:
          inset 0 0 28px rgba(55, 166, 255, 0.22),
          0 0 28px rgba(0, 133, 255, 0.18);
        overflow: hidden;
      }

      .exec-logo img {
        width: 82%;
        height: 82%;
        object-fit: contain;
        filter: drop-shadow(0 8px 18px rgba(0, 0, 0, 0.42));
      }

      .exec-logo--xl {
        width: 102px;
        height: 102px;
      }

      .exec-logo--lg {
        width: 112px;
        height: 112px;
      }

      .exec-logo--fallback {
        font-weight: 950;
        font-size: 24px;
        color: white;
        letter-spacing: 0.08em;
      }

      .exec-top-stat span,
      .exec-profile span {
        display: block;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 11px;
        font-weight: 900;
        margin-bottom: 8px;
      }

      .exec-top-stat strong,
      .exec-profile input {
        display: block;
        color: var(--text);
        font-size: 15px;
        font-weight: 900;
        letter-spacing: 0.04em;
      }

      .exec-top-stat small,
      .exec-profile small {
        display: block;
        color: var(--muted);
        margin-top: 5px;
        font-size: 12px;
      }

      .exec-stat-icon {
        width: 34px;
        height: 34px;
        border-radius: 12px;
        display: grid !important;
        place-items: center;
        margin: 0 !important;
        color: white !important;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.18);
        letter-spacing: 0 !important;
      }

      .exec-dots {
        display: flex;
        gap: 7px;
        margin-top: 6px;
      }

      .exec-dots b {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: rgba(85, 126, 164, 0.42);
      }

      .exec-dots b.is-on {
        background: var(--blue);
        box-shadow: 0 0 12px rgba(54, 168, 255, 0.8);
      }

      .exec-profile {
        border-right: 0;
      }

      .exec-profile input {
        width: 100%;
        border: 0;
        outline: 0;
        padding: 0;
        background: transparent;
        font: inherit;
      }

      .exec-avatar {
        width: 74px;
        height: 74px;
        border-radius: 50%;
        background:
          radial-gradient(circle at 50% 32%, #f2a35f 0 18%, transparent 19%),
          radial-gradient(circle at 50% 70%, #172033 0 34%, transparent 35%),
          linear-gradient(145deg, rgba(32, 83, 125, 0.55), rgba(4, 14, 24, 0.95));
        border: 1px solid rgba(89, 169, 255, 0.35);
        position: relative;
        box-shadow: inset 0 0 28px rgba(48, 142, 255, 0.2);
      }

      .exec-avatar span {
        position: absolute;
        width: 30px;
        height: 30px;
        left: 22px;
        top: 16px;
        border-radius: 50%;
        background: #f4a05f;
        margin: 0;
      }

      .exec-main {
        display: grid;
        grid-template-columns: 500px minmax(460px, 1fr) 450px;
        gap: 34px;
        align-items: center;
      }

      .exec-left,
      .exec-right {
        display: grid;
        gap: 28px;
      }

      .exec-takeover-card,
      .exec-panel {
        border: 1px solid var(--line);
        border-radius: 12px;
        background:
          linear-gradient(180deg, rgba(8, 25, 43, 0.88), rgba(5, 15, 27, 0.82)),
          radial-gradient(circle at 80% 16%, rgba(46, 148, 255, 0.14), transparent 36%);
        box-shadow:
          0 22px 60px rgba(0, 0, 0, 0.28),
          inset 0 1px 0 rgba(255, 255, 255, 0.04);
      }

      .exec-takeover-card {
        min-height: 258px;
        padding: 30px 28px;
        position: relative;
        overflow: hidden;
      }

      .exec-takeover-card::after {
        content: "";
        position: absolute;
        width: 180px;
        height: 1px;
        right: 42px;
        top: 145px;
        transform: rotate(-48deg);
        background: linear-gradient(90deg, transparent, rgba(255, 139, 34, 0.9), transparent);
        box-shadow: 0 0 16px rgba(255, 139, 34, 0.5);
      }

      .exec-kicker,
      .exec-panel__head span,
      .exec-league-pulse span,
      .exec-bottom-status span {
        color: var(--blue);
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 12px;
        font-weight: 950;
      }

      .exec-takeover-card h2 {
        margin: 20px 0 16px;
        font-size: 30px;
        line-height: 1.08;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        text-shadow: 0 0 22px rgba(255, 255, 255, 0.16);
      }

      .exec-takeover-card p {
        width: 58%;
        color: #b7c5d2;
        line-height: 1.55;
        margin: 0;
        font-size: 14px;
      }

      .exec-takeover-card__logo {
        position: absolute;
        right: 44px;
        top: 56px;
      }

      .exec-focus-line {
        position: absolute;
        left: 28px;
        right: 28px;
        bottom: 25px;
        padding-top: 18px;
        border-top: 1px solid rgba(131, 184, 235, 0.18);
        display: flex;
        gap: 18px;
        color: #bdcbd8;
        font-size: 14px;
      }

      .exec-focus-line b {
        color: var(--blue);
      }

      .exec-panel {
        padding: 18px 18px;
      }

      .exec-panel__head {
        margin: 0 0 14px;
      }

      .exec-leader-list,
      .exec-snapshot-list,
      .exec-schedule-list {
        border: 1px solid rgba(103, 157, 209, 0.16);
        border-radius: 9px;
        overflow: hidden;
        background: rgba(4, 14, 25, 0.42);
      }

      .exec-leader-row {
        height: 68px;
        display: grid;
        grid-template-columns: 54px 1fr auto;
        gap: 14px;
        align-items: center;
        padding: 0 18px;
        border-bottom: 1px solid rgba(103, 157, 209, 0.13);
      }

      .exec-leader-row:last-child {
        border-bottom: 0;
      }

      .exec-headshot {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: grid;
        place-items: center;
        background:
          radial-gradient(circle at 50% 25%, #f1b084 0 20%, transparent 21%),
          linear-gradient(180deg, rgba(40, 82, 118, 0.9), rgba(13, 27, 43, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.08);
        overflow: hidden;
      }

      .exec-headshot span {
        color: rgba(255, 255, 255, 0.75);
        font-weight: 950;
        transform: translateY(10px);
      }

      .exec-leader-row strong {
        display: block;
        font-size: 14px;
      }

      .exec-leader-row small {
        display: block;
        color: var(--muted);
        margin-top: 4px;
      }

      .exec-leader-row b {
        color: var(--blue);
        font-size: 18px;
      }

      .exec-mini-button {
        width: 100%;
        height: 42px;
        margin-top: 12px;
        border: 1px solid rgba(77, 155, 239, 0.34);
        border-radius: 7px;
        background: linear-gradient(180deg, rgba(14, 47, 81, 0.82), rgba(7, 24, 43, 0.88));
        color: #dfeeff;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-weight: 900;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 18px;
      }

      .exec-mini-button:hover {
        border-color: rgba(255, 139, 34, 0.75);
        box-shadow: 0 0 18px rgba(255, 139, 34, 0.16);
      }

      .exec-center {
        min-height: 430px;
        display: grid;
        place-items: center;
      }

      .exec-menu-grid {
        width: min(720px, 100%);
        display: grid;
        grid-template-columns: repeat(3, minmax(160px, 1fr));
        gap: 12px;
        justify-items: stretch;
      }

      .exec-menu-card {
        min-height: 142px;
        border: 1px solid rgba(62, 145, 225, 0.42);
        border-radius: 13px;
        background:
          radial-gradient(circle at 50% 0%, rgba(49, 155, 255, 0.18), transparent 46%),
          linear-gradient(180deg, rgba(11, 38, 66, 0.9), rgba(6, 20, 36, 0.92));
        color: white;
        cursor: pointer;
        display: grid;
        place-items: center;
        gap: 8px;
        padding: 22px;
        position: relative;
        overflow: hidden;
        box-shadow:
          0 18px 42px rgba(0, 0, 0, 0.26),
          inset 0 1px 0 rgba(255, 255, 255, 0.06);
        transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
      }

      .exec-menu-card:nth-child(7) {
        grid-column: 2;
      }

      .exec-menu-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 139, 34, 0.75);
        box-shadow:
          0 20px 48px rgba(0, 0, 0, 0.34),
          0 0 24px rgba(255, 139, 34, 0.18),
          inset 0 1px 0 rgba(255, 255, 255, 0.08);
      }

      .exec-menu-card.is-active {
        border-color: var(--orange);
        background:
          radial-gradient(circle at 50% 22%, rgba(255, 139, 34, 0.38), transparent 54%),
          linear-gradient(180deg, rgba(90, 45, 16, 0.94), rgba(18, 27, 44, 0.96));
        box-shadow:
          0 0 0 1px rgba(255, 188, 73, 0.36),
          0 0 32px rgba(255, 139, 34, 0.34),
          inset 0 0 48px rgba(255, 139, 34, 0.16);
      }

      .exec-menu-card__icon {
        font-size: 39px;
        line-height: 1;
        filter: drop-shadow(0 8px 16px rgba(0, 0, 0, 0.32));
      }

      .exec-menu-card__label {
        font-size: 18px;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }

      .exec-menu-card__sub {
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
      }

      .exec-snapshot-list > div {
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 0 18px;
        border-bottom: 1px solid rgba(103, 157, 209, 0.13);
      }

      .exec-snapshot-list > div:last-child {
        border-bottom: 0;
      }

      .exec-snapshot-list span {
        color: var(--muted);
        font-size: 14px;
      }

      .exec-snapshot-list strong {
        font-size: 14px;
      }

      .exec-snapshot-list .blue {
        color: var(--blue);
      }

      .exec-schedule-row {
        min-height: 68px;
        display: grid;
        grid-template-columns: 42px 1fr 24px;
        align-items: center;
        gap: 12px;
        padding: 0 14px;
        border-bottom: 1px solid rgba(103, 157, 209, 0.13);
      }

      .exec-schedule-row:last-child {
        border-bottom: 0;
      }

      .exec-schedule-row small {
        display: block;
        color: var(--muted);
        margin-bottom: 5px;
      }

      .exec-schedule-row strong {
        font-size: 14px;
      }

      .exec-schedule-row > span:last-child {
        color: #c8d6e6;
        opacity: 0.8;
        font-size: 18px;
      }

      .exec-mini-mark {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        background: rgba(37, 90, 135, 0.34);
        border: 1px solid rgba(98, 170, 245, 0.24);
        overflow: hidden;
      }

      .exec-mini-mark img {
        width: 84%;
        height: 84%;
        object-fit: contain;
      }

      .exec-mini-mark--fallback {
        color: var(--blue);
        font-weight: 950;
      }

      .exec-empty {
        padding: 18px;
        color: var(--muted);
      }

      .exec-error {
        border: 1px solid rgba(255, 95, 109, 0.42);
        background: rgba(80, 10, 20, 0.72);
        color: #ffd6da;
        border-radius: 10px;
        padding: 12px 16px;
        font-weight: 800;
      }

      .exec-decision-dock {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 12px;
      }

      .exec-decision-card {
        border: 1px solid rgba(255, 139, 34, 0.34);
        border-radius: 10px;
        padding: 14px;
        background:
          radial-gradient(circle at 0% 0%, rgba(255, 139, 34, 0.18), transparent 38%),
          rgba(10, 23, 38, 0.86);
      }

      .exec-decision-card span {
        color: var(--orange2);
        font-size: 11px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 950;
      }

      .exec-decision-card strong {
        display: block;
        margin-top: 8px;
      }

      .exec-decision-card p {
        color: var(--muted);
        margin: 8px 0 12px;
        font-size: 13px;
        line-height: 1.45;
      }

      .exec-decision-card div {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }

      .exec-decision-card button {
        border: 1px solid rgba(80, 157, 240, 0.42);
        border-radius: 7px;
        background: rgba(18, 55, 90, 0.78);
        color: white;
        padding: 8px 10px;
        cursor: pointer;
        font-weight: 800;
      }

      .exec-wire-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
      }

      .exec-wire-item {
        min-height: 48px;
        border: 1px solid rgba(90, 154, 218, 0.22);
        border-radius: 9px;
        background: rgba(7, 22, 36, 0.82);
        color: #cbd8e8;
        padding: 12px;
        font-size: 12px;
        line-height: 1.35;
        overflow: hidden;
      }

      .exec-wire-item--trade {
        border-color: rgba(54, 168, 255, 0.35);
      }

      .exec-wire-item--alert {
        border-color: rgba(255, 139, 34, 0.42);
      }

      .exec-wire-item--major {
        border-color: rgba(255, 210, 96, 0.45);
      }

      .exec-action-bar {
        min-height: 108px;
        border: 1px solid rgba(69, 154, 240, 0.28);
        border-radius: 12px;
        background: rgba(6, 18, 31, 0.86);
        display: grid;
        grid-template-columns: 1fr 0.75fr;
        align-items: center;
        gap: 28px;
        padding: 18px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.26);
      }

      .exec-sim-buttons {
        display: grid;
        grid-template-columns: 320px repeat(4, minmax(120px, 1fr));
        gap: 14px;
      }

      .exec-sim-buttons button,
      .exec-league-pulse button {
        min-height: 54px;
        border-radius: 8px;
        border: 1px solid rgba(83, 155, 232, 0.42);
        background: linear-gradient(180deg, rgba(17, 52, 88, 0.92), rgba(7, 25, 44, 0.96));
        color: white;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        cursor: pointer;
      }

      .exec-sim-buttons button:disabled {
        opacity: 0.45;
        cursor: not-allowed;
      }

      .exec-sim-buttons .exec-advance {
        border-color: rgba(255, 179, 71, 0.82);
        background:
          radial-gradient(circle at 50% 0%, rgba(255, 184, 65, 0.32), transparent 54%),
          linear-gradient(180deg, #e46d10, #7c2907);
        box-shadow:
          0 0 0 1px rgba(255, 205, 104, 0.32),
          0 0 28px rgba(255, 139, 34, 0.34);
        font-size: 18px;
      }

      .exec-sim-buttons .exec-advance span {
        font-size: 28px;
        vertical-align: -2px;
        margin-left: 12px;
      }

      .exec-league-pulse {
        display: grid;
        grid-template-columns: 52px 1fr 150px;
        gap: 16px;
        align-items: center;
        border-left: 1px solid rgba(119, 164, 205, 0.18);
        padding-left: 28px;
      }

      .exec-pulse-icon {
        color: var(--orange);
        font-size: 44px;
        text-shadow: 0 0 18px rgba(255, 139, 34, 0.45);
      }

      .exec-league-pulse p {
        margin: 6px 0 0;
        color: var(--muted);
        line-height: 1.4;
        font-size: 13px;
      }

      .exec-bottom-status {
        height: 44px;
        border: 1px solid rgba(79, 143, 208, 0.22);
        border-radius: 8px;
        background: rgba(5, 16, 28, 0.74);
        display: grid;
        grid-template-columns: auto auto 1fr;
        align-items: center;
        gap: 44px;
        padding: 0 24px;
      }

      .exec-bottom-status > div {
        display: flex;
        align-items: center;
        gap: 18px;
      }

      .exec-bottom-status strong {
        font-size: 13px;
        color: #c7d4e2;
      }

      .exec-shortcut {
        justify-self: end;
        color: var(--muted2);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 12px;
      }

      @media (max-width: 1500px) {
        .exec-shell {
          padding: 14px;
        }

        .exec-topbar {
          grid-template-columns: 1.6fr 1fr 1fr 1fr;
        }

        .exec-top-stat:nth-of-type(n + 4) {
          display: none;
        }

        .exec-main {
          grid-template-columns: 360px 1fr 360px;
          gap: 18px;
        }

        .exec-sim-buttons {
          grid-template-columns: 240px repeat(4, 1fr);
        }
      }

      @media (max-width: 1120px) {
        .exec-root {
          overflow: auto;
        }

        .exec-shell {
          min-height: auto;
        }

        .exec-topbar,
        .exec-main,
        .exec-action-bar,
        .exec-bottom-status {
          grid-template-columns: 1fr;
        }

        .exec-top-stat,
        .exec-brand,
        .exec-profile {
          border-right: 0;
          border-bottom: 1px solid rgba(100, 150, 200, 0.14);
        }

        .exec-menu-grid {
          grid-template-columns: 1fr 1fr;
        }

        .exec-menu-card:nth-child(7) {
          grid-column: auto;
        }

        .exec-wire-strip {
          grid-template-columns: 1fr;
        }

        .exec-sim-buttons {
          grid-template-columns: 1fr 1fr;
        }

        .exec-league-pulse {
          border-left: 0;
          padding-left: 0;
          grid-template-columns: 44px 1fr;
        }

        .exec-league-pulse button {
          grid-column: 1 / -1;
        }
      }
    `}</style>
  );
}