import React, { useLayoutEffect, useMemo, useRef, useState } from "react";
import PlayerHeadshot from "../../components/PlayerHeadshot";
import TeamLogoBadge from "../../components/ui/TeamLogoBadge";
import { ensurePlayerHeadshotFields } from "../../utils/playerHeadshots";
import { resolveFranchiseTeamLogo } from "../../utils/teamLogos";
import {
  contractLine,
  draftLine,
  finalSeasonLine,
  fmtGaa,
  fmtInt,
  fmtMillions,
  fmtSvPct,
  getCareerTotals,
  getRetireeAge,
  getRetireeName,
  getRetireeOverall,
  getRetireePosition,
  getRetireeTeam,
  isGoalieRetiree,
  isMinorLeagueRetiree,
  isUserTeamRetiree,
  normalizeRetireesPayload,
  retireeToHeadshotPlayer,
  sortRetirees,
  tributeScore,
  trophySummary,
} from "./retirementHelpers";

function seasonLabel(franchiseState) {
  const y = franchiseState?.season_year || franchiseState?.seasonYear;
  return y ? `${y}–${String(Number(y) + 1).slice(-2)}` : "";
}

function teamLogoFor(row) {
  const name = String(getRetireeTeam(row) || "").replace(/\s*\((AHL|ECHL)\)\s*$/i, "");
  if (!name || /^(UFA|free agent|overseas)$/i.test(name)) return "";
  return resolveFranchiseTeamLogo(
    { team_abbrev: row?.team_abbrev || undefined, name, team_name: name, full_name: name },
    name
  );
}

function headshotPlayer(row) {
  return ensurePlayerHeadshotFields(retireeToHeadshotPlayer(row));
}

function metaLine(row) {
  const c = getCareerTotals(row);
  const age = getRetireeAge(row);
  const seasons = c.seasons || null;
  return [
    getRetireePosition(row),
    age != null ? `age ${age}` : null,
    seasons ? `${seasons} NHL season${seasons === 1 ? "" : "s"}` : null,
    c.first && c.last ? `${c.first.slice(0, 4)}–${c.last.slice(0, 4)}` : null,
    row?.captaincy_status === "captain" ? "captain" : row?.captaincy_status === "alternate" ? "alternate captain" : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

function careerFigures(row) {
  const c = getCareerTotals(row);
  if (isGoalieRetiree(row)) {
    return [
      { k: "GP", v: fmtInt(c.gp) },
      { k: "W", v: fmtInt(c.w) },
      { k: "SV%", v: fmtSvPct(c.sv_pct) },
      { k: "GAA", v: fmtGaa(c.gaa) },
      { k: "SO", v: fmtInt(c.so) },
    ];
  }
  return [
    { k: "GP", v: fmtInt(c.gp) },
    { k: "G", v: fmtInt(c.g) },
    { k: "A", v: fmtInt(c.a) },
    { k: "PTS", v: fmtInt(c.pts) },
  ];
}

function careerLabel(row) {
  return getCareerTotals(row).source === "nhl" ? "NHL career" : "Recorded in this save";
}

function bestSeasonLine(row, goalie) {
  const b = row?.best_season;
  if (!b || typeof b !== "object") return null;
  const where = b.team ? ` with ${b.team}` : "";
  if (goalie) {
    return `${b.season}${where}: ${fmtInt(b.w)} wins${b.sv_pct ? `, ${fmtSvPct(b.sv_pct)}` : ""}`;
  }
  return `${b.season}${where}: ${fmtInt(b.g)} G, ${fmtInt(b.a)} A, ${fmtInt(b.pts)} PTS`;
}

function cupsAndTrophies(row) {
  const trophies = trophySummary(row);
  const cupsFromList = trophies.find((t) => /stanley cup/i.test(t.name));
  const cups = Math.max(Number(row?.cups_won) || 0, cupsFromList ? cupsFromList.n : 0);
  const rest = trophies.filter((t) => !/stanley cup/i.test(t.name));
  return { cups, rest };
}

function Fact({ label, children }) {
  if (children == null || children === "" || children === false) return null;
  return (
    <div className="ret2-fact">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function HeroTribute({ row, rank, isMine }) {
  const name = getRetireeName(row);
  const team = getRetireeTeam(row);
  const goalie = isGoalieRetiree(row);
  const overall = getRetireeOverall(row);
  const c = getCareerTotals(row);
  const { cups, rest } = cupsAndTrophies(row);
  const logo = teamLogoFor(row);
  const peak = Number(row?.peak_overall);
  const showPeak = Number.isFinite(peak) && overall != null && peak > overall;

  return (
    <article className={`ret2-hero${isMine ? " is-mine" : ""}${rank === 0 ? " is-lead" : ""}`} aria-label={`${name} tribute`}>
      <div className="ret2-hero-logo-bg" aria-hidden="true">
        {logo ? <img src={logo} alt="" draggable={false} /> : null}
      </div>
      <div className="ret2-hero-face">
        <PlayerHeadshot
          player={headshotPlayer(row)}
          size="lg"
          preferPhoto
          showFlag={false}
          style={{ "--size": rank === 0 ? "236px" : "176px" }}
        />
      </div>
      <div className="ret2-hero-body">
        <div className="ret2-hero-top">
          <TeamLogoBadge teamLogo={logo} teamName={team} size={rank === 0 ? 84 : 64} variant="watermark" className="ret2-logo" />
          <div className="ret2-hero-id">
            <div className="ret2-hero-name" role="heading" aria-level={2}>
              {name}
            </div>
            <div className="ret2-hero-meta">
              <span className="ret2-team-name">{team}</span> · {metaLine(row)}
            </div>
          </div>
          <div className="ret2-hero-ovr" aria-label={`Final overall ${overall ?? "unknown"}`}>
            <strong>{overall ?? "—"}</strong>
            <span>{showPeak ? `final OVR · peak ${peak}` : "final OVR"}</span>
          </div>
        </div>

        <p className="ret2-hero-reason">
          {row?.confirmed === false ? "Considering retirement — your decision is pending. " : ""}
          {row?.retirement_reason || "Retired"}
          {row?.retirement_detail ? <span className="ret2-hero-detail"> — {row.retirement_detail}</span> : null}
        </p>

        <div className="ret2-hero-stats">
          <span className="ret2-stats-label">{careerLabel(row)}</span>
          <div className="ret2-hero-figs">
            {careerFigures(row).map((f) => (
              <div key={f.k} className="ret2-fig">
                <strong>{f.v}</strong>
                <span>{f.k}</span>
              </div>
            ))}
            {cups ? (
              <div className="ret2-fig is-cup">
                <strong>{cups}</strong>
                <span>{cups === 1 ? "Stanley Cup" : "Stanley Cups"}</span>
              </div>
            ) : null}
          </div>
        </div>

        <dl className="ret2-facts">
          <Fact label="Best season">{bestSeasonLine(row, goalie)}</Fact>
          <Fact label="Final season">{finalSeasonLine(row, goalie)}</Fact>
          <Fact label="Clubs">{c.teams.length ? c.teams.join(" → ") : null}</Fact>
          <Fact label="Drafted">{draftLine(row)}</Fact>
          <Fact label="Last contract">{contractLine(row)}</Fact>
          <Fact label="Trophies">{rest.length ? rest.map((t) => (t.n > 1 ? `${t.name} ×${t.n}` : t.name)).join(", ") : null}</Fact>
        </dl>
      </div>
    </article>
  );
}

function ListRow({ row, isMine }) {
  const name = getRetireeName(row);
  const team = getRetireeTeam(row);
  const goalie = isGoalieRetiree(row);
  const overall = getRetireeOverall(row);
  const logo = teamLogoFor(row);
  const { cups } = cupsAndTrophies(row);
  const final = finalSeasonLine(row, goalie);
  const draft = draftLine(row);
  const contract = contractLine(row);

  return (
    <article className={`ret2-row${isMine ? " is-mine" : ""}`}>
      <div className="ret2-row-face">
        <PlayerHeadshot player={headshotPlayer(row)} size="lg" preferPhoto showFlag={false} style={{ "--size": "92px" }} />
      </div>
      <TeamLogoBadge teamLogo={logo} teamName={team} size={58} variant="watermark" className="ret2-logo" />
      <div className="ret2-row-id">
        <div className="ret2-row-name">{name}</div>
        <div className="ret2-row-meta">
          <span className="ret2-team-name">{team}</span> · {metaLine(row)}
        </div>
        <div className="ret2-row-reason">
          {row?.confirmed === false ? <span className="ret2-pending">Decision pending · </span> : null}
          {row?.retirement_reason || "Retired"}
          {row?.retirement_detail ? <span className="ret2-row-detail"> — {row.retirement_detail}</span> : null}
        </div>
      </div>
      <div className="ret2-row-ovr">
        <strong>{overall ?? "—"}</strong>
        <span>OVR</span>
      </div>
      <div className="ret2-row-career">
        <span className="ret2-stats-label">
          {careerLabel(row)}
          {cups ? <span className="ret2-cup-note"> · {cups}× Cup</span> : null}
        </span>
        <div className="ret2-row-figs">
          {careerFigures(row).map((f) => (
            <div key={f.k} className="ret2-fig">
              <strong>{f.v}</strong>
              <span>{f.k}</span>
            </div>
          ))}
        </div>
      </div>
      <dl className="ret2-row-facts">
        <Fact label="Final season">{final}</Fact>
        <Fact label="Drafted">{draft}</Fact>
        <Fact label="Contract">{contract}</Fact>
      </dl>
    </article>
  );
}

function MinorRow({ row, isMine }) {
  const team = getRetireeTeam(row);
  const age = getRetireeAge(row);
  const c = getCareerTotals(row);
  return (
    <li className={`ret2-minor${isMine ? " is-mine" : ""}`}>
      <TeamLogoBadge teamLogo={teamLogoFor(row)} teamName={team} size={34} variant="watermark" className="ret2-logo" />
      <span className="ret2-minor-name">{getRetireeName(row)}</span>
      <span className="ret2-minor-meta">
        {[getRetireePosition(row), age != null ? `age ${age}` : null, team, c.source === "nhl" && c.gp ? `${fmtInt(c.gp)} NHL GP` : null]
          .filter(Boolean)
          .join(" · ")}
      </span>
      <span className="ret2-minor-ovr">{getRetireeOverall(row) ?? "—"}</span>
    </li>
  );
}

const SORTS = [
  { id: "ovr", label: "Overall" },
  { id: "career", label: "Career" },
  { id: "games", label: "Games" },
  { id: "age", label: "Age" },
];

function sortRows(rows, sortId) {
  if (sortId === "ovr") return sortRetirees(rows);
  const key = {
    career: (r) => tributeScore(r),
    games: (r) => getCareerTotals(r).gp || 0,
    age: (r) => getRetireeAge(r) || 0,
  }[sortId];
  return [...rows].sort((a, b) => key(b) - key(a) || getRetireeName(a).localeCompare(getRetireeName(b)));
}


/** How many fixed-height rows fit in a box, re-measured on resize (no scrolling anywhere). */
function useFitCount(rowPx, cols = 1) {
  const ref = useRef(null);
  const [n, setN] = useState(6 * cols);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return undefined;
    const measure = () => setN(Math.max(1, Math.floor(el.clientHeight / rowPx)) * cols);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [rowPx, cols]);
  return [ref, n];
}

function Pager({ page, pages, onPage }) {
  if (pages <= 1) return null;
  return (
    <div className="ret2-pager" role="group" aria-label="Pages">
      <button type="button" className="ret2-pager-step" disabled={page <= 0} onClick={() => onPage(page - 1)} aria-label="Previous page">
        ‹
      </button>
      <span>
        Page <b>{page + 1}</b> of {pages}
      </span>
      <button type="button" className="ret2-pager-step" disabled={page >= pages - 1} onClick={() => onPage(page + 1)} aria-label="Next page">
        ›
      </button>
    </div>
  );
}

export default function RetirementsBoard({ franchiseState = {}, retirees = [], onContinue, onBack }) {
  const userTeamId = franchiseState?.user_team_id ?? franchiseState?.userTeamId ?? null;
  const all = useMemo(() => normalizeRetireesPayload(retirees), [retirees]);
  const [filter, setFilter] = useState("all");
  const [sortId, setSortId] = useState("ovr");

  const nhl = useMemo(() => all.filter((r) => !isMinorLeagueRetiree(r)), [all]);
  const minors = useMemo(() => sortRetirees(all.filter((r) => isMinorLeagueRetiree(r))), [all]);
  const mine = useMemo(() => all.filter((r) => isUserTeamRetiree(r, userTeamId)), [all, userTeamId]);

  const pool = filter === "mine" ? mine.filter((r) => !isMinorLeagueRetiree(r)) : filter === "minors" ? [] : nhl;
  const minorPool = filter === "mine" ? minors.filter((r) => isUserTeamRetiree(r, userTeamId)) : filter === "nhl" ? [] : minors;

  // Hero tributes: the biggest careers in the current view (stars only, max 3).
  const heroes = useMemo(() => {
    const ranked = [...pool].sort((a, b) => tributeScore(b) - tributeScore(a));
    const strong = ranked.filter((r) => (getRetireeOverall(r) ?? 0) >= 80 || (getCareerTotals(r).gp || 0) >= 600);
    return (strong.length ? strong : ranked).slice(0, filter === "mine" ? Math.min(3, ranked.length) : 3);
  }, [pool, filter]);
  const heroSet = new Set(heroes);
  const rest = sortRows(pool.filter((r) => !heroSet.has(r)), sortId);

  const totalGp = nhl.reduce((s, r) => s + (getCareerTotals(r).gp || 0), 0);
  const careerKnown = nhl.some((r) => getCareerTotals(r).source === "nhl");
  const totalCups = nhl.reduce((s, r) => s + cupsAndTrophies(r).cups, 0);
  const capFreed = fmtMillions(retirees && !Array.isArray(retirees) ? retirees?.summary?.cap_freed_user : null);

  const [listModeRaw, setListMode] = useState("nhl");
  const listMode = listModeRaw === "nhl" && !rest.length ? "minors" : listModeRaw === "minors" && !minorPool.length ? "nhl" : listModeRaw;
  const [boardRef, perPage] = useFitCount(listMode === "nhl" ? 82 : 46, listMode === "nhl" ? 1 : 3);
  const [page, setPage] = useState(0);
  const listRows = listMode === "nhl" ? rest : minorPool;
  const pageCount = Math.max(1, Math.ceil(listRows.length / perPage));
  const safePage = Math.min(page, pageCount - 1);
  const pageRows = listRows.slice(safePage * perPage, safePage * perPage + perPage);
  // On the last NHL page, the room left under the rows shows minor-league retirees instead of empty space.
  const leftoverRows = listMode === "nhl" ? perPage - pageRows.length : 0;
  const fillMinors =
    listMode === "nhl" && leftoverRows >= 2 && minorPool.length
      ? minorPool.slice(0, Math.max(0, Math.floor((leftoverRows * 82 - 40) / 46)) * 3)
      : [];

  const tabs = [
    { id: "all", label: "All", n: all.length },
    { id: "mine", label: "Your club", n: mine.length },
    { id: "nhl", label: "NHL", n: nhl.length },
    { id: "minors", label: "Minors & free agents", n: minors.length },
  ];

  return (
    <section className="ret2-root">
      <header className="ret2-head">
        <button type="button" className="ret2-back" onClick={onBack}>
          ← Back
        </button>
        <div className="ret2-headline">
          <div className="ret2-heading" role="heading" aria-level={1}>
            Retirements
          </div>
          <div className="ret2-sub">
            Career archive{seasonLabel(franchiseState) ? ` · ${seasonLabel(franchiseState)} offseason` : ""} · saying goodbye to the
            players who called it a career
          </div>
        </div>
        <div className="ret2-summary">
          <div>
            <strong>{all.length}</strong>
            <span>retiring</span>
          </div>
          <div className={mine.length ? "is-mine" : ""}>
            <strong>{mine.length}</strong>
            <span>from your club</span>
          </div>
          {careerKnown ? (
            <div>
              <strong>{fmtInt(totalGp)}</strong>
              <span>NHL games between them</span>
            </div>
          ) : null}
          {totalCups ? (
            <div>
              <strong>{totalCups}</strong>
              <span>Stanley Cups</span>
            </div>
          ) : null}
          {capFreed ? (
            <div className="is-mine">
              <strong>{capFreed}</strong>
              <span>cap freed for you</span>
            </div>
          ) : null}
        </div>
      </header>

      <nav className="ret2-tools" aria-label="Filter and sort retirees">
        <div className="ret2-tabs" role="tablist">
          {tabs.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={filter === t.id}
              className={`ret2-tab${filter === t.id ? " is-on" : ""}`}
              onClick={() => {
                setFilter(t.id);
                setPage(0);
              }}
              disabled={!t.n && t.id !== "all"}
            >
              {t.label} <span className="ret2-tab-n">{t.n}</span>
            </button>
          ))}
        </div>
        <div className="ret2-sorts">
          <span className="ret2-sort-label">Sort by</span>
          {SORTS.map((s) => (
            <button
              key={s.id}
              type="button"
              className={`ret2-tab${sortId === s.id ? " is-on" : ""}`}
              aria-pressed={sortId === s.id}
              onClick={() => {
                setSortId(s.id);
                setPage(0);
              }}
            >
              {s.label}
            </button>
          ))}
        </div>
      </nav>

      <main className={`ret2-stage${heroes.length ? "" : " no-heroes"}`} aria-label="Retired players">
        {!all.length ? <p className="ret2-empty">No retirements this year.</p> : null}

        {heroes.length ? (
          <section className={`ret2-heroes n${heroes.length}`} aria-label="Featured retirees">
            {heroes.map((row, i) => (
              <HeroTribute key={`${row.player_id || row.name}-hero`} row={row} rank={i} isMine={isUserTeamRetiree(row, userTeamId)} />
            ))}
          </section>
        ) : null}

        {rest.length || minorPool.length ? (
          <section className="ret2-board" aria-label="All retirees">
            <div className="ret2-board-head">
              <div className="ret2-board-tabs" role="tablist">
                {rest.length ? (
                  <button type="button" role="tab" aria-selected={listMode === "nhl"} className={`ret2-tab${listMode === "nhl" ? " is-on" : ""}`} onClick={() => {
                      setListMode("nhl");
                      setPage(0);
                    }}>
                    Also retiring <span className="ret2-tab-n">{rest.length}</span>
                  </button>
                ) : null}
                {minorPool.length ? (
                  <button type="button" role="tab" aria-selected={listMode === "minors"} className={`ret2-tab${listMode === "minors" ? " is-on" : ""}`} onClick={() => {
                      setListMode("minors");
                      setPage(0);
                    }}>
                    From the minors and free agency <span className="ret2-tab-n">{minorPool.length}</span>
                  </button>
                ) : null}
              </div>
              <Pager page={safePage} pages={pageCount} onPage={setPage} />
            </div>
            <div className={`ret2-board-body is-${listMode}`} ref={boardRef}>
              {listMode === "nhl"
                ? [
                    ...pageRows.map((row, index) => (
                      <ListRow key={`${row.player_id || row.name}-${index}`} row={row} isMine={isUserTeamRetiree(row, userTeamId)} />
                    )),
                    fillMinors.length ? (
                      <div key="fill-head" className="ret2-fill-head">
                        From the minors and free agency
                        <button type="button" className="ret2-fill-more" onClick={() => setListMode("minors")}>
                          See all {minorPool.length}
                        </button>
                      </div>
                    ) : null,
                    fillMinors.length ? (
                      <ul key="fill" className="ret2-minor-grid">
                        {fillMinors.map((row, index) => (
                          <MinorRow key={`${row.player_id || row.name}-f${index}`} row={row} isMine={isUserTeamRetiree(row, userTeamId)} />
                        ))}
                      </ul>
                    ) : null,
                  ]
                : (
                  <ul className="ret2-minor-grid">
                    {pageRows.map((row, index) => (
                      <MinorRow key={`${row.player_id || row.name}-m${index}`} row={row} isMine={isUserTeamRetiree(row, userTeamId)} />
                    ))}
                  </ul>
                )}
            </div>
          </section>
        ) : null}

        {all.length && !heroes.length && !rest.length && !minorPool.length ? (
          <p className="ret2-empty">Nobody in this view.</p>
        ) : null}
      </main>

      <footer className="ret2-foot">
        <span className="ret2-foot-note">Next: Board of Governors</span>
        <button type="button" className="ret2-go" onClick={onContinue}>
          Continue to Board of Governors
        </button>
      </footer>
    </section>
  );
}
