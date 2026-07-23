import React, { useMemo, useState } from "react";
import { useGameUI } from "../../game/GameUIContext";
import { SCREENS } from "../../game/constants";
import { getTeamLogoSrc } from "../../utils/teamLogos";
import { pickFranchiseData } from "../eventUtils";
import {
  extractDevelopmentPlayers,
  extractLeagueNhlDevelopment,
  summarizeDevelopmentReport,
  groupPlayersByOrg,
  sortDevelopmentPlayers,
  ovrDeltaClass,
  formatSeasonStats,
  ORG_GROUPS,
} from "./prospectDevelopmentHelpers";
import "../../styles/nhlcalShell.css";
import "./ProspectDevelopment.css";

const TABS = [
  { id: "org", label: "Your Organization" },
  { id: "league", label: "League NHL" },
];

function SideNavButton({ active, icon, label, onClick }) {
  return (
    <button
      type="button"
      className={`nhlcal-side-button${active ? " is-active" : ""}`}
      onClick={onClick}
    >
      <span className="nhlcal-side-icon">{icon}</span>
      <span className="nhlcal-side-label">{label}</span>
    </button>
  );
}

function TeamLogo({ team, size = "large" }) {
  const src = getTeamLogoSrc(team);
  if (!src) {
    const label = String(team?.abbrev || team?.abbr || team?.name || "TM").slice(0, 3).toUpperCase();
    return (
      <span className={`nhlcal-team-logo size-${size} nhldev-logo-fallback`} aria-hidden>
        {label}
      </span>
    );
  }
  return (
    <span className={`nhlcal-team-logo size-${size}`}>
      <img src={src} alt="" loading="lazy" />
    </span>
  );
}

function PlayerAvatar({ name, size = 52 }) {
  const initials = String(name || "?")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() || "")
    .join("") || "?";
  return (
    <span className="nhldev-avatar" style={{ width: size, height: size, fontSize: size * 0.34 }} aria-hidden>
      {initials}
    </span>
  );
}

function CompactRow({ player, selected, onSelect }) {
  const deltaCls = ovrDeltaClass(player.ovrDelta);
  return (
    <button
      type="button"
      className={`nhldev-prospect-row${selected ? " is-selected" : ""}`}
      onClick={() => onSelect(player)}
    >
      <PlayerAvatar name={player.name} size={56} />
      <div className="nhldev-row-main">
        <div className="nhldev-row-head">
          <strong>{player.name}</strong>
          <span className="nhldev-row-ovr">{player.ovr} OVR</span>
        </div>
        <div className="nhldev-row-meta">
          <span>{player.position}</span>
          <span>Age {player.age}</span>
          {player.potential != null ? <span>POT {player.potential}</span> : null}
          <span>{player.league !== "—" ? player.league : player.team}</span>
        </div>
        <div className="nhldev-row-line">
          <span className="nhldev-ovr-shift">
            {player.previousOvr} → {player.ovr}
            <em className={`nhldev-delta-pill ${deltaCls}`}>{player.ovrDeltaLabel}</em>
          </span>
        </div>
        {player.primaryReason ? (
          <div className="nhldev-row-reason">
            <span className="nhldev-reason-text">{player.primaryReason}</span>
          </div>
        ) : null}
      </div>
    </button>
  );
}

function DetailPanel({ player, franchiseTeam }) {
  if (!player) {
    return (
      <section className="nhlcal-card nhldev-detail">
        <header className="nhlcal-card-header">
          <div>
            <p>Development</p>
            <h3>Select Player</h3>
          </div>
        </header>
        <p className="nhlcal-small-empty">Select a player to see overall and attribute growth.</p>
      </section>
    );
  }

  const deltaCls = ovrDeltaClass(player.ovrDelta);
  const statsLine = formatSeasonStats(player);
  const attrs = (player.allAttributes?.length ? player.allAttributes : player.topAttributes) || [];
  const teamLabel = (() => {
    if (player.league === "NHL" || String(player.orgGroup || "").startsWith("NHL")) {
      return franchiseTeam || player.team || "—";
    }
    return player.team || franchiseTeam || "—";
  })();

  return (
    <section className="nhlcal-card nhldev-detail">
      <header className="nhlcal-card-header nhldev-detail-head">
        <PlayerAvatar name={player.name} size={72} />
        <div>
          <p>{player.league} · {teamLabel}</p>
          <h3>{player.name}</h3>
        </div>
        <span className="nhlcal-header-pill">{player.position}</span>
      </header>

      <div className="nhldev-detail-hero">
        <div className="nhldev-detail-ovr-block">
          <span className="nhldev-detail-ovr-label">Overall change</span>
          <strong className="nhldev-detail-ovr-value">
            {player.previousOvr}
            <em>→</em>
            {player.ovr}
          </strong>
          <span className={`nhldev-delta-pill large ${deltaCls}`}>{player.ovrDeltaLabel}</span>
        </div>
        <div className="nhldev-detail-stat-grid">
          <div>
            <span>Previous OVR</span>
            <strong>{player.previousOvr}</strong>
          </div>
          <div>
            <span>Current OVR</span>
            <strong>{player.ovr}</strong>
          </div>
          <div>
            <span>Age</span>
            <strong>{player.age}</strong>
          </div>
          <div>
            <span>Potential</span>
            <strong>{player.potential != null ? player.potential : "—"}</strong>
          </div>
        </div>
      </div>

      <div className="nhldev-detail-block">
        <h4>Attribute growth</h4>
        {attrs.length ? (
          <ul className="nhldev-attr-list">
            {attrs.map((a) => (
              <li key={a.key} className={a.delta > 0 ? "is-up" : a.delta < 0 ? "is-down" : ""}>
                <span>{a.label}</span>
                <strong>
                  {a.before != null && a.after != null
                    ? `${Math.round(a.before)} → ${Math.round(a.after)} `
                    : ""}
                  ({a.display})
                </strong>
              </li>
            ))}
          </ul>
        ) : (
          <p className="nhlcal-small-empty">
            {Math.abs(player.ovrDelta) < 1
              ? "No attribute movement recorded this cycle."
              : "Overall moved, but individual attribute deltas were not logged."}
          </p>
        )}
      </div>

      {player.primaryReason ? (
        <div className="nhldev-detail-block">
          <h4>Notes</h4>
          <p className="nhldev-bullet-line">{player.primaryReason}</p>
        </div>
      ) : null}

      {statsLine ? (
        <div className="nhldev-detail-block">
          <h4>Season</h4>
          <p className="nhldev-stats-line">{statsLine}</p>
        </div>
      ) : null}
    </section>
  );
}

function seasonLabel(franchiseState) {
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${y}–${Number(y) + 1}` : "Franchise Mode";
}

export default function ProspectDevelopmentMenu({
  franchiseState = {},
  eventData = {},
  onContinue,
  onBack,
}) {
  const gameUI = useGameUI();
  const navigate = gameUI?.navigate || (() => {});
  const franchise = pickFranchiseData(franchiseState, eventData) || franchiseState;
  const team = franchise?.user_team || franchise?.team || franchiseState?.user_team || {};
  const franchiseLabel = team?.name || team?.full_name || team?.nickname || "Franchise";

  const [tab, setTab] = useState("org");
  const [selectedId, setSelectedId] = useState(null);
  const [expandedGroup, setExpandedGroup] = useState("NHL / AHL");

  const orgPlayers = useMemo(
    () => extractDevelopmentPlayers(franchiseState, eventData),
    [franchiseState, eventData]
  );
  const leaguePlayers = useMemo(
    () => extractLeagueNhlDevelopment(franchiseState, eventData),
    [franchiseState, eventData]
  );
  const allPlayers = tab === "league" ? leaguePlayers : orgPlayers;
  const summary = useMemo(
    () => summarizeDevelopmentReport(
      eventData?.development_report ?? franchiseState?.development_report ?? {},
      orgPlayers
    ),
    [eventData, franchiseState, orgPlayers]
  );
  const orgGroups = useMemo(() => groupPlayersByOrg(orgPlayers), [orgPlayers]);
  const empty = !allPlayers.length;

  const selected = useMemo(() => {
    const list = allPlayers;
    if (!list.length) return null;
    return list.find((p) => p.id === selectedId) || list[0];
  }, [allPlayers, selectedId]);

  const handleSelect = (p) => setSelectedId(p.id);
  const handleBack = () => {
    if (onBack) {
      onBack();
      return;
    }
    navigate(SCREENS.CALENDAR);
  };

  return (
    <div className="nhlcal-root nhldev-root">
      <aside className="nhlcal-sidebar">
        <button
          type="button"
          className="nhlcal-brand-button"
          onClick={() => navigate(SCREENS.OFFICE)}
          title="Office"
        >
          <span className="nhlcal-shield-icon">⌂</span>
        </button>
        <nav className="nhlcal-side-nav" aria-label="Franchise navigation">
          <SideNavButton icon="▦" label="Office" onClick={() => navigate(SCREENS.OFFICE)} />
          <SideNavButton icon="◫" label="Calendar" onClick={() => navigate(SCREENS.CALENDAR)} />
          <SideNavButton active icon="▤" label="Dev" onClick={() => {}} />
          <SideNavButton icon="◉" label="Roster" onClick={() => navigate(SCREENS.ROSTER)} />
        </nav>
      </aside>

      <main className="nhlcal-main nhldev-main">
        <header className="nhlcal-topbar nhldev-topbar nhldev-topbar--logo">
          <section className="nhldev-brand-only">
            <TeamLogo team={team} size="xlarge" />
          </section>

          <section className="nhlcal-action-cluster">
            <button type="button" className="nhlcal-quick-link" onClick={handleBack}>
              {onBack ? "Hub World" : "Calendar"}
            </button>
            <div className="nhlcal-date-chip">
              <span className="nhlcal-date-icon">◫</span>
              <div>
                <strong>{seasonLabel(franchiseState)}</strong>
                <span>Offseason</span>
              </div>
            </div>
            {onContinue ? (
              <button type="button" className="nhlcal-advance-button" onClick={onContinue}>
                Continue to Lottery
              </button>
            ) : null}
          </section>
        </header>

        {!empty && tab === "org" ? (
          <p className="nhldev-summary-line">
            {summary.improved} improved · {summary.regressed} regressed · {summary.total} players
          </p>
        ) : null}

        <div className="nhldev-tab-bar">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              className={`nhlcal-quick-link${tab === t.id ? " is-active" : ""}`}
              onClick={() => {
                setTab(t.id);
                setSelectedId(null);
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {empty ? (
          <section className="nhlcal-empty-state nhldev-empty">
            <h3>No development data yet.</h3>
            <p>Advance from salary cap or reload your franchise save.</p>
            {onContinue ? (
              <button type="button" className="nhlcal-advance-button" onClick={onContinue}>
                Continue to Lottery
              </button>
            ) : null}
          </section>
        ) : (
          <section className="nhlcal-content-grid nhldev-grid">
            <section className="nhlcal-calendar-panel nhldev-panel">
              {tab === "org" ? (
                <div className="nhldev-org-groups nhldev-scroll">
                  {ORG_GROUPS.map((groupName) => {
                    const list = sortDevelopmentPlayers(orgGroups[groupName] || [], "ovr_delta");
                    if (!list.length) return null;
                    const open = expandedGroup === groupName;
                    return (
                      <section key={groupName} className="nhldev-org-group">
                        <button
                          type="button"
                          className="nhldev-group-head"
                          onClick={() => setExpandedGroup(open ? "" : groupName)}
                        >
                          <span>{groupName}</span>
                          <span className="nhldev-group-count">{list.length}</span>
                        </button>
                        {open ? (
                          <div className="nhldev-prospect-list">
                            {list.map((p) => (
                              <CompactRow
                                key={p.id}
                                player={p}
                                selected={selected?.id === p.id}
                                onSelect={handleSelect}
                              />
                            ))}
                          </div>
                        ) : null}
                      </section>
                    );
                  })}
                </div>
              ) : (
                <div className="nhldev-prospect-list nhldev-scroll">
                  <p className="nhldev-league-note">
                    Season-end overall growth across the NHL — find who jumped around the league.
                  </p>
                  {leaguePlayers.map((p) => (
                    <CompactRow
                      key={p.id}
                      player={p}
                      selected={selected?.id === p.id}
                      onSelect={handleSelect}
                    />
                  ))}
                </div>
              )}
            </section>

            <DetailPanel player={selected} franchiseTeam={franchiseLabel} />
          </section>
        )}
      </main>
    </div>
  );
}
