import React, { useLayoutEffect, useMemo, useRef, useState } from "react";
import { useGameUI } from "../../game/GameUIContext";
import { SCREENS } from "../../game/constants";
import { getTeamLogoSrc } from "../../utils/teamLogos";
import PlayerHeadshot from "../../components/PlayerHeadshot";
import TeamLogoBadge from "../../components/ui/TeamLogoBadge";
import { pickFranchiseData } from "../eventUtils";
import {
  extractDevelopmentPlayers,
  extractLeagueNhlDevelopment,
  summarizeDevelopmentReport,
  groupPlayersByOrg,
  ovrDeltaClass,
  reconciledDevelopmentTrend,
  computeDevelopmentOverview,
  seasonStatCells,
  ORG_GROUPS,
} from "./prospectDevelopmentHelpers";

const VIEWS = [
  { id: "org", label: "Your organization" },
  { id: "league", label: "Around the NHL" },
];

const POSITIONS = [
  { id: "all", label: "All" },
  { id: "F", label: "Forwards" },
  { id: "D", label: "Defence" },
  { id: "G", label: "Goalies" },
];

const COLUMNS = [
  { id: "name", label: "Player", cls: "is-player" },
  { id: "team", label: "Team", cls: "is-team" },
  { id: "age", label: "Age", cls: "is-age" },
  { id: "ovr_delta", label: "Overall change", cls: "is-change" },
  { id: "potential", label: "POT", cls: "is-pot" },
  { id: "attrs", label: "Top gains", cls: "is-attrs", sortable: false },
];

function num(v, fb = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fb;
}

function signed(n, digits = 0) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "0";
  const s = digits ? v.toFixed(digits) : String(Math.round(v));
  if (Number(s) === 0) return digits ? (0).toFixed(digits) : "0";
  return v > 0 ? `+${s}` : s;
}

function posGroup(pos) {
  const p = String(pos || "").toUpperCase();
  if (p === "G") return "G";
  if (["D", "LD", "RD"].includes(p)) return "D";
  return "F";
}

function headshotPlayer(p) {
  return { ...(p.raw || {}), id: p.id, player_id: p.id, name: p.name, position: p.position, age: p.age };
}

function teamLogoFor(p, fallbackTeam) {
  const lg = String(p.league || "").toUpperCase();
  const isOrgPro = lg === "NHL" || lg === "AHL" || lg === "ECHL" || p.pool === "nhl" || p.pool === "ahl";
  // Junior / European clubs have no NHL logo; a name lookup there would map
  // "Ottawa 67's" onto the Senators via its first three letters.
  if (!isOrgPro && !p.teamAbbrev) return null;
  if (p.teamAbbrev || p.teamName) {
    const src = getTeamLogoSrc({ team_abbrev: p.teamAbbrev, name: p.teamName, team_name: p.teamName });
    if (src) return src;
  }
  if (isOrgPro && fallbackTeam) return getTeamLogoSrc(fallbackTeam);
  return null;
}

function Shot({ player, size = 56 }) {
  return (
    <span className="dev2-shot" style={{ "--dev2-shot": `${size}px` }} aria-hidden>
      <PlayerHeadshot
        player={headshotPlayer(player)}
        size="lg"
        preferPhoto
        showFlag={false}
        style={{ "--size": `${size}px` }}
      />
    </span>
  );
}

function Logo({ player, fallbackTeam, size = 36 }) {
  const src = teamLogoFor(player, fallbackTeam);
  if (!src) return <span className="dev2-leaguecode">{player.league !== "—" ? player.league : ""}</span>;
  return <TeamLogoBadge teamLogo={src} teamName={player.team} size={size} variant="badge" className="dev2-logo" />;
}

function DeltaBar({ delta, max }) {
  const d = num(delta);
  const pct = max > 0 ? Math.min(100, (Math.abs(d) / max) * 100) : 0;
  return (
    <span className={`dev2-dbar ${d > 0 ? "is-up" : d < 0 ? "is-down" : "is-flat"}`} aria-hidden>
      <span className="dev2-dbar-fill" style={{ width: `${Math.max(d ? 6 : 0, pct)}%` }} />
    </span>
  );
}

function trendTone(p) {
  const t = reconciledDevelopmentTrend(p);
  if (t === "Breakout" || t === "Improved") return "up";
  if (t === "Regressed") return "down";
  return "flat";
}

function PlayerRow({ player, selected, onSelect, maxDelta, fallbackTeam }) {
  const d = num(player.ovrDelta);
  const gains = (player.topAttributes || []).filter((a) => a.delta > 0).slice(0, 3);
  const drops = (player.topAttributes || []).filter((a) => a.delta < 0).slice(0, 2);
  const shown = gains.length ? gains : drops;
  return (
    <button
      type="button"
      className={`dev2-row${selected ? " is-selected" : ""}`}
      onClick={() => onSelect(player)}
      aria-pressed={selected}
    >
      <span className="dev2-cell is-player">
        <Shot player={player} size={60} />
        <span className="dev2-who">
          <span className="dev2-name">{player.name}</span>
          <span className="dev2-meta">
            {player.position} · {player.league !== "—" ? player.league : "Unassigned"}
            {player.team && player.team !== "—" ? ` · ${player.team}` : ""}
          </span>
        </span>
      </span>
      <span className="dev2-cell is-team">
        <Logo player={player} fallbackTeam={fallbackTeam} size={40} />
      </span>
      <span className="dev2-cell is-age">{player.age}</span>
      <span className="dev2-cell is-change">
        <span className="dev2-ovrline">
          <span className="dev2-ovr-prev">{player.previousOvr}</span>
          <span className="dev2-arrow">→</span>
          <span className="dev2-ovr-now">{player.ovr}</span>
          <span className={`dev2-delta is-${ovrDeltaClass(d)}`}>{signed(d)}</span>
        </span>
        <DeltaBar delta={d} max={maxDelta} />
      </span>
      <span className="dev2-cell is-pot">{player.potential ?? "—"}</span>
      <span className="dev2-cell is-attrs">
        {shown.length ? (
          shown.map((a) => (
            <span key={a.key} className={`dev2-gain ${a.delta > 0 ? "is-up" : "is-down"}`}>
              {a.label} <b>{a.display}</b>
            </span>
          ))
        ) : (
          <span className="dev2-quiet">No rating moved</span>
        )}
      </span>
    </button>
  );
}

function MoverLine({ player, onSelect }) {
  const d = num(player.ovrDelta);
  return (
    <button type="button" className="dev2-mover" onClick={() => onSelect(player)}>
      <Shot player={player} size={44} />
      <span className="dev2-mover-who">
        <span className="dev2-mover-name">{player.name}</span>
        <span className="dev2-mover-meta">
          {player.position} · {player.age} · {player.previousOvr} → {player.ovr}
        </span>
      </span>
      <span className={`dev2-mover-delta is-${ovrDeltaClass(d)}`}>{signed(d)}</span>
    </button>
  );
}

function SummaryBand({ overview, onSelect, scopeLabel }) {
  const maxAbs = Math.max(1, ...overview.ageGroups.map((g) => Math.abs(g.avg)));
  return (
    <section className="dev2-summary" aria-label="Development summary">
      <div className="dev2-sum-totals">
        <div className="dev2-sum-scope">{scopeLabel}</div>
        <div className="dev2-bignums">
          <div>
            <span className="dev2-bignum is-up">{overview.improved}</span>
            <span className="dev2-biglabel">improved</span>
          </div>
          <div>
            <span className="dev2-bignum is-down">{overview.regressed}</span>
            <span className="dev2-biglabel">regressed</span>
          </div>
          <div>
            <span className="dev2-bignum">{overview.unchanged}</span>
            <span className="dev2-biglabel">unchanged</span>
          </div>
        </div>
        <p className="dev2-sum-line">
          {overview.total} players · average {signed(overview.avg, 1)} OVR · net {signed(overview.netGain)} overall points
        </p>
      </div>

      <div className="dev2-sum-col">
        <div className="dev2-sum-head">Biggest risers</div>
        {overview.risers.length ? (
          overview.risers.map((p) => <MoverLine key={p.id} player={p} onSelect={onSelect} />)
        ) : (
          <p className="dev2-quiet">Nobody gained overall.</p>
        )}
      </div>

      <div className="dev2-sum-col">
        <div className="dev2-sum-head">Biggest fallers</div>
        {overview.fallers.length ? (
          overview.fallers.map((p) => <MoverLine key={p.id} player={p} onSelect={onSelect} />)
        ) : (
          <p className="dev2-quiet">Nobody lost overall this year.</p>
        )}
      </div>

      <div className="dev2-sum-col">
        <div className="dev2-sum-head">Average change by age</div>
        <ul className="dev2-ages">
          {overview.ageGroups.map((g) => (
            <li key={g.id} className={g.count ? "" : "is-empty"}>
              <span className="dev2-age-label">{g.label}</span>
              <span className="dev2-age-track">
                <span
                  className={`dev2-age-fill ${g.avg > 0 ? "is-up" : g.avg < 0 ? "is-down" : ""}`}
                  style={{ width: `${g.count ? Math.max(4, (Math.abs(g.avg) / maxAbs) * 100) : 0}%` }}
                />
              </span>
              <span className="dev2-age-val">{g.count ? signed(g.avg, 1) : "—"}</span>
              <span className="dev2-age-n">{g.count}</span>
            </li>
          ))}
        </ul>
        <div className="dev2-split">
          <div>
            <span className="dev2-split-num">{signed(overview.prospects.avg, 1)}</span>
            <span className="dev2-split-label">
              Prospects, 23 and under ({overview.prospects.improved} of {overview.prospects.count} up)
            </span>
          </div>
          <div>
            <span className="dev2-split-num">{signed(overview.vets.avg, 1)}</span>
            <span className="dev2-split-label">
              Veterans, 24 and over ({overview.vets.improved} of {overview.vets.count} up)
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}

function DetailView({ player, fallbackTeam, franchiseLabel }) {
  if (!player) {
    return (
      <aside className="dev2-detail">
        <p className="dev2-quiet">Select a player to see why his rating moved.</p>
      </aside>
    );
  }
  const d = num(player.ovrDelta);
  const attrs = (player.allAttributes?.length ? player.allAttributes : player.topAttributes) || [];
  const stats = seasonStatCells(player);
  const drivers = player.growthDrivers || [];
  const notes = [player.primaryReason, ...(player.secondaryReasons || [])].filter(Boolean).join(" · ");
  const isOrgPro = player.league === "NHL" || player.league === "AHL" || String(player.orgGroup || "").startsWith("NHL");
  const teamLabel = isOrgPro && !player.teamName ? franchiseLabel : player.team;
  const room = player.potential != null ? Math.max(0, player.potential - player.ovr) : null;
  const history = (player.developmentHistory || []).filter((h) => h && h.season != null).slice(-5);
  const outlook = player.outlook;
  const statsLeague = player.seasonStats?.league;

  return (
    <aside className="dev2-detail" aria-label={`${player.name} development detail`}>
      <header className="dev2-d-head">
        <Shot player={player} size={128} />
        <div className="dev2-d-id">
          <div className="dev2-d-name" role="heading" aria-level={2}>{player.name}</div>
          <div className="dev2-d-team">
            <Logo player={player} fallbackTeam={fallbackTeam} size={44} />
            <span>{teamLabel && teamLabel !== "—" ? teamLabel : "Unassigned"}</span>
          </div>
          <div className="dev2-d-meta">
            {player.position} · Age {player.age} · {player.league !== "—" ? player.league : "No league"}
            {player.developmentPhase ? ` · ${player.developmentPhase}` : ""}
          </div>
        </div>
      </header>

      <div className="dev2-d-ovr">
        <div className="dev2-d-ovrnums">
          <span className="dev2-d-prev">{player.previousOvr}</span>
          <span className="dev2-d-arrow">→</span>
          <span className="dev2-d-now">{player.ovr}</span>
          <span className={`dev2-d-delta is-${ovrDeltaClass(d)}`}>{signed(d)}</span>
        </div>
        <dl className="dev2-d-facts">
          <div><dt>Potential</dt><dd>{player.potential ?? "—"}</dd></div>
          <div><dt>Room left</dt><dd>{room ?? "—"}</dd></div>
          <div><dt>Trend</dt><dd className={`is-${trendTone(player)}`}>{reconciledDevelopmentTrend(player)}</dd></div>
          {player.readinessTier && player.readinessTier !== "—" && player.orgGroup !== "NHL / AHL" ? (
            <div><dt>Readiness</dt><dd>{player.readinessTier}</dd></div>
          ) : null}
        </dl>
      </div>

      {notes ? <p className="dev2-d-notes" title={notes}>{notes}</p> : null}

      <section className="dev2-d-block">
        <div className="dev2-d-h">Why he moved</div>
        {drivers.length ? (
          <ul className="dev2-drivers">
            {drivers.map((dr) => (
              <li key={dr.key} className={`is-${dr.tone || "neutral"}`}>
                <span className="dev2-drv-dot" aria-hidden />
                <span className="dev2-drv-label">{dr.label}</span>
                <span className="dev2-drv-value">{dr.value}</span>
                <span className="dev2-drv-effect">{dr.effect}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="dev2-quiet">
            Growth factors arrive with the next report load (restart the game server to enable them).
          </p>
        )}
        {outlook?.summary ? <p className="dev2-outlook">Next season: {outlook.summary}</p> : null}
      </section>

      <section className="dev2-d-block">
        <div className="dev2-d-h">
          Season stats{statsLeague ? <span className="dev2-d-hsub"> · {statsLeague}</span> : null}
        </div>
        {stats.length ? (
          <dl className="dev2-stats">
            {stats.map((c) => (
              <div key={c.label}>
                <dt>{c.label}</dt>
                <dd>{c.value}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="dev2-quiet">No games logged this season.</p>
        )}
      </section>

      <section className="dev2-d-block">
        <div className="dev2-d-h">Attribute changes</div>
        {attrs.length ? (
          <ul className="dev2-attrs">
            {attrs.map((a) => (
              <li key={a.key} className={a.delta > 0 ? "is-up" : "is-down"}>
                <span className="dev2-attr-label">{a.label}</span>
                <span className="dev2-attr-track" aria-hidden>
                  {a.before != null && a.after != null ? (
                    <>
                      <span className="dev2-attr-base" style={{ width: `${Math.min(a.before, a.after)}%` }} />
                      <span
                        className="dev2-attr-move"
                        style={{ left: `${Math.min(a.before, a.after)}%`, width: `${Math.abs(a.after - a.before)}%` }}
                      />
                    </>
                  ) : null}
                </span>
                <span className="dev2-attr-vals">
                  {a.before != null && a.after != null ? `${a.before} → ${a.after}` : ""}
                </span>
                <span className="dev2-attr-delta">{a.display}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="dev2-quiet">
            {Math.abs(d) < 1 ? "No rating moved this cycle." : "Overall moved, but rating changes were not logged."}
          </p>
        )}
      </section>

      {history.length > 1 ? (
        <section className="dev2-d-block">
          <div className="dev2-d-h">Development history</div>
          <ol className="dev2-history">
            {history.map((h) => (
              <li key={`${h.season}-${h.new_ovr ?? h.ovr_after ?? ""}`}>
                <span className="dev2-h-season">{h.season}</span>
                <span className="dev2-h-ovr">{h.new_ovr ?? h.ovr_after ?? "—"}</span>
                <span className={`dev2-h-delta is-${ovrDeltaClass(h.ovr_delta)}`}>
                  {h.ovr_delta != null ? signed(h.ovr_delta) : ""}
                </span>
              </li>
            ))}
          </ol>
        </section>
      ) : null}
    </aside>
  );
}

function sortRows(list, sortId, dir) {
  const out = [...list];
  const mul = dir === "asc" ? 1 : -1;
  out.sort((a, b) => {
    let r = 0;
    switch (sortId) {
      case "name": r = String(a.name).localeCompare(String(b.name)); break;
      case "team": r = String(a.team).localeCompare(String(b.team)); break;
      case "age": r = num(a.age, 99) - num(b.age, 99); break;
      case "potential": r = num(a.potential, -1) - num(b.potential, -1); break;
      case "ovr": r = num(a.ovr) - num(b.ovr); break;
      case "ovr_delta":
      default: r = num(a.ovrDelta) - num(b.ovrDelta) || num(a.ovr) - num(b.ovr);
    }
    return r * mul;
  });
  return out;
}

function seasonLabel(franchiseState) {
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${y}–${String(Number(y) + 1).slice(-2)} offseason` : "Offseason";
}

/** How many fixed-height rows fit in a box, re-measured on resize (the screen never scrolls). */
function useFitCount(rowPx) {
  const ref = useRef(null);
  const [n, setN] = useState(8);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return undefined;
    const measure = () => setN(Math.max(1, Math.floor(el.clientHeight / rowPx)));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [rowPx]);
  return [ref, n];
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
  const devReport = eventData?.development_report ?? franchiseState?.development_report ?? {};
  const franchiseLabel =
    team?.name || team?.full_name || team?.nickname || devReport?.user_team_name || "Your club";
  const fallbackTeam = useMemo(
    () => ({ ...team, team_abbrev: team?.abbrev || team?.abbr || devReport?.user_team_abbrev }),
    [team, devReport]
  );

  const [view, setView] = useState("org");
  const [group, setGroup] = useState("all");
  const [pos, setPos] = useState("all");
  const [sort, setSort] = useState({ id: "ovr_delta", dir: "desc" });
  const [selectedId, setSelectedId] = useState(null);

  const orgPlayers = useMemo(
    () => extractDevelopmentPlayers(franchiseState, eventData),
    [franchiseState, eventData]
  );
  const leaguePlayers = useMemo(
    () => extractLeagueNhlDevelopment(franchiseState, eventData),
    [franchiseState, eventData]
  );
  const summary = useMemo(() => summarizeDevelopmentReport(devReport, orgPlayers), [devReport, orgPlayers]);
  const orgGroups = useMemo(() => groupPlayersByOrg(orgPlayers), [orgPlayers]);

  const scoped = view === "league" ? leaguePlayers : orgPlayers;
  const overview = useMemo(() => computeDevelopmentOverview(scoped), [scoped]);

  const rows = useMemo(() => {
    let list = scoped;
    if (view === "org" && group !== "all") list = orgGroups[group] || [];
    if (pos !== "all") list = list.filter((p) => posGroup(p.position) === pos);
    return sortRows(list, sort.id, sort.dir);
  }, [scoped, view, group, orgGroups, pos, sort]);

  const maxDelta = useMemo(() => Math.max(1, ...scoped.map((p) => Math.abs(num(p.ovrDelta)))), [scoped]);
  const empty = !orgPlayers.length && !leaguePlayers.length;

  const selected = useMemo(() => {
    const pool = [...rows, ...scoped];
    if (!pool.length) return null;
    return pool.find((p) => p.id === selectedId) || rows[0] || scoped[0];
  }, [rows, scoped, selectedId]);

  const handleSelect = (p) => setSelectedId(p.id);
  const [rowsRef, perPage] = useFitCount(76);
  const [pageRaw, setPage] = useState(0);
  const pageCount = Math.max(1, Math.ceil(rows.length / perPage));
  const page = Math.min(pageRaw, pageCount - 1);
  const pageRows = rows.slice(page * perPage, page * perPage + perPage);
  const filterKey = `${view}|${group}|${pos}|${sort.id}|${sort.dir}`;
  const lastKey = useRef(filterKey);
  if (lastKey.current !== filterKey) {
    lastKey.current = filterKey;
    if (pageRaw !== 0) setPage(0);
  }
  const handleBack = () => {
    if (onBack) {
      onBack();
      return;
    }
    navigate(SCREENS.CALENDAR);
  };
  const toggleSort = (id) =>
    setSort((s) => (s.id === id ? { id, dir: s.dir === "desc" ? "asc" : "desc" } : { id, dir: id === "name" || id === "team" || id === "age" ? "asc" : "desc" }));

  const userLogo = getTeamLogoSrc(fallbackTeam);
  const scopeLabel =
    view === "league" ? "League-wide NHL movers" : `${franchiseLabel} organization`;

  return (
    <div className="dev2-root">
      <nav className="dev2-rail" aria-label="Franchise navigation">
        <button type="button" className="dev2-railitem" onClick={() => navigate(SCREENS.HUB)}>
          <span aria-hidden>▦</span>Office
        </button>
        <button type="button" className="dev2-railitem" onClick={() => navigate(SCREENS.CALENDAR)}>
          <span aria-hidden>◫</span>Calendar
        </button>
        <span className="dev2-railitem is-active" aria-current="page">
          <span aria-hidden>▲</span>Development
        </span>
        <button type="button" className="dev2-railitem" onClick={() => navigate(SCREENS.ROSTER)}>
          <span aria-hidden>◉</span>Roster
        </button>
      </nav>

      <main className="dev2-main">
        <header className="dev2-top">
          <div className="dev2-brand">
            {userLogo ? (
              <TeamLogoBadge teamLogo={userLogo} teamName={franchiseLabel} size={84} variant="badge" className="dev2-brandlogo" />
            ) : null}
            <div>
              <div className="dev2-heading" role="heading" aria-level={1}>Player development</div>
              <div className="dev2-sub">
                {seasonLabel(franchiseState)} · year-end growth review
                {!empty ? ` · ${summary.improved} improved, ${summary.regressed} regressed across ${summary.total} in the organization` : ""}
              </div>
            </div>
          </div>
          <div className="dev2-actions">
            <button type="button" className="dev2-textlink" onClick={handleBack}>
              {onBack ? "Hub World" : "Calendar"}
            </button>
            {onContinue ? (
              <button type="button" className="dev2-primary" onClick={onContinue}>
                Continue to lottery
              </button>
            ) : null}
          </div>
        </header>

        {empty ? (
          <section className="dev2-empty">
            <div className="dev2-empty-h">No development data yet</div>
            <p>Advance from the salary cap stage or reload your franchise save.</p>
            {onContinue ? (
              <button type="button" className="dev2-primary" onClick={onContinue}>
                Continue to lottery
              </button>
            ) : null}
          </section>
        ) : (
          <>
            <div className="dev2-views" role="tablist" aria-label="Scope">
              {VIEWS.map((v) => (
                <button
                  key={v.id}
                  type="button"
                  role="tab"
                  aria-selected={view === v.id}
                  className={`dev2-view${view === v.id ? " is-active" : ""}`}
                  onClick={() => {
                    setView(v.id);
                    setGroup("all");
                    setSelectedId(null);
                  }}
                >
                  {v.label}
                  <span className="dev2-view-n">{v.id === "league" ? leaguePlayers.length : orgPlayers.length}</span>
                </button>
              ))}
            </div>

            {scoped.length ? (
              <SummaryBand overview={overview} onSelect={handleSelect} scopeLabel={scopeLabel} />
            ) : null}

            <section className="dev2-body">
              <div className="dev2-listwrap">
                <div className="dev2-filters">
                  {view === "org" ? (
                    <div className="dev2-filterset" aria-label="Level">
                      {["all", ...ORG_GROUPS].map((g) => {
                        const n = g === "all" ? orgPlayers.length : (orgGroups[g] || []).length;
                        if (g !== "all" && !n) return null;
                        return (
                          <button
                            key={g}
                            type="button"
                            className={`dev2-filter${group === g ? " is-active" : ""}`}
                            onClick={() => setGroup(g)}
                          >
                            {g === "all" ? "Whole organization" : g} <span>{n}</span>
                          </button>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="dev2-filterset">
                      <span className="dev2-filternote">
                        Every NHL player whose overall moved a full point, all 32 clubs
                      </span>
                    </div>
                  )}
                  <div className="dev2-filterset" aria-label="Position">
                    {POSITIONS.map((p) => (
                      <button
                        key={p.id}
                        type="button"
                        className={`dev2-filter${pos === p.id ? " is-active" : ""}`}
                        onClick={() => setPos(p.id)}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="dev2-thead" role="row">
                  {COLUMNS.map((c) =>
                    c.sortable === false ? (
                      <span key={c.id} className={`dev2-th ${c.cls}`}>{c.label}</span>
                    ) : (
                      <button
                        key={c.id}
                        type="button"
                        className={`dev2-th ${c.cls}${sort.id === c.id ? " is-sorted" : ""}`}
                        onClick={() => toggleSort(c.id)}
                      >
                        {c.label}
                        {sort.id === c.id ? <span aria-hidden>{sort.dir === "desc" ? " ▼" : " ▲"}</span> : null}
                      </button>
                    )
                  )}
                </div>

                <div className="dev2-rows" ref={rowsRef}>
                  {rows.length ? (
                    pageRows.map((p) => (
                      <PlayerRow
                        key={p.id}
                        player={p}
                        selected={selected?.id === p.id}
                        onSelect={handleSelect}
                        maxDelta={maxDelta}
                        fallbackTeam={fallbackTeam}
                      />
                    ))
                  ) : (
                    <p className="dev2-quiet dev2-pad">No players match these filters.</p>
                  )}
                </div>
                {pageCount > 1 ? (
                  <div className="dev2-pager" role="group" aria-label="Pages">
                    <span>
                      {page * perPage + 1}–{Math.min(rows.length, (page + 1) * perPage)} of {rows.length} players
                    </span>
                    <button type="button" className="dev2-pager-step" disabled={page <= 0} onClick={() => setPage(page - 1)} aria-label="Previous page">
                      ‹
                    </button>
                    <span>
                      Page <b>{page + 1}</b> of {pageCount}
                    </span>
                    <button type="button" className="dev2-pager-step" disabled={page >= pageCount - 1} onClick={() => setPage(page + 1)} aria-label="Next page">
                      ›
                    </button>
                  </div>
                ) : null}
              </div>

              <DetailView player={selected} fallbackTeam={fallbackTeam} franchiseLabel={franchiseLabel} />
            </section>
          </>
        )}
      </main>
    </div>
  );
}
