/**
 * Legacy Hall: the save's archive. Every section loads one slice from
 * /api/franchise/legacy/* when opened; nothing here is on the franchise state payload.
 * Empty sections say why (no seasons finished yet, data not tracked) instead of filling in.
 *
 * Layout (2026-11 overhaul): Double Dash dashboard. A title bar with the franchise picker, a cup-select
 * rail of sections on the left, and each section as a grid of cream panels: big number tiles first,
 * then the chart or leaderboard, then the detail lists.
 */
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { loadTeamIdentities } from "../components/teamIdentity/teamIdentityStore";
import { apiError, legacyApi, seasonLabel, takeLegacyFocus } from "../components/legacy/legacyApi";

const SECTIONS = [
  { id: "franchise", label: "Franchise", sub: "Banners & records", glyph: "★" },
  { id: "players", label: "Players", sub: "Leaders & careers", glyph: "♛" },
  { id: "league", label: "League", sub: "Champions by season", glyph: "🏆" },
  { id: "drafts", label: "Drafts", sub: "How the picks turned out", glyph: "✎" },
  { id: "trades", label: "Trades", sub: "Who won the deal", glyph: "⇄" },
  { id: "eras", label: "Eras", sub: "The franchise story", glyph: "◷" },
  { id: "rivalries", label: "Rivalries", sub: "Bad blood", glyph: "⚔" },
  { id: "archive", label: "Archive", sub: "Every event on file", glyph: "▤" },
];
const STATS = [["pts", "Points"], ["g", "Goals"], ["a", "Assists"], ["gp", "Games"], ["plus_minus", "+/-"], ["w", "Goalie wins"], ["so", "Shutouts"]];
const EVENT_TYPES = ["", "championship", "career_milestone", "franchise_record", "captaincy_start", "captaincy_end", "rebrand", "retirement", "hall_of_fame", "number_retired", "first_playoffs_in_save"];
const pretty = (s) => String(s || "").replace(/_/g, " ");
const isG = (pos) => String(pos || "").toUpperCase().startsWith("G");

function useLoad(fn, deps) {
  const [state, setState] = useState({ data: null, err: "", loading: true });
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, err: "" }));
    fn()
      .then((data) => alive && setState({ data, err: "", loading: false }))
      .catch((e) => alive && setState({ data: null, err: apiError(e), loading: false }));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

/** Click + keyboard for rows that open something. */
const act = (fn) =>
  fn
    ? {
        role: "button",
        tabIndex: 0,
        onClick: fn,
        onKeyDown: (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fn();
          }
        },
      }
    : {};

function Status({ state, empty }) {
  if (state.loading) return <div className="lgh-empty is-loading">Pulling the files…</div>;
  if (state.err) return <div className="lgh-empty is-err">{state.err}</div>;
  if (empty) return <div className="lgh-empty">{empty}</div>;
  return null;
}

function Panel({ title, tone = "", right = null, className = "", children }) {
  return (
    <section className={`lgh-panel ${tone ? `is-${tone}` : ""} ${className}`}>
      {title ? (
        <header className="lgh-panel__head">
          <h3>{title}</h3>
          {right}
        </header>
      ) : null}
      <div className="lgh-panel__body">{children}</div>
    </section>
  );
}

function Tile({ label, value, sub, tone = "" }) {
  return (
    <div className={`lgh-tile ${tone ? `is-${tone}` : ""}`}>
      <span className="lgh-tile__label">{label}</span>
      <b className="lgh-tile__value">{value}</b>
      {sub ? <span className="lgh-tile__sub">{sub}</span> : null}
    </div>
  );
}

function Pills({ value, options, onChange, label }) {
  return (
    <div className="lgh-pills" role="radiogroup" aria-label={label}>
      {options.map(([k, l]) => (
        <button key={k || "all"} type="button" role="radio" aria-checked={value === k} className={`lgh-pill ${value === k ? "is-on" : ""}`} onClick={() => onChange(k)}>
          {l}
        </button>
      ))}
    </div>
  );
}

function Pager({ page, total, per, onPage, newer = "Prev", older = "Next" }) {
  if (!total || total <= per) return null;
  return (
    <div className="lgh-pager">
      <button type="button" className="lgh-pill" disabled={page === 0} onClick={() => onPage(page - 1)}>
        ◀ {newer}
      </button>
      <span>
        {page * per + 1}–{Math.min(total, (page + 1) * per)} of {total}
      </span>
      <button type="button" className="lgh-pill" disabled={(page + 1) * per >= total} onClick={() => onPage(page + 1)}>
        {older} ▶
      </button>
    </div>
  );
}

const Note = ({ children }) => (children ? <p className="lgh-note">{children}</p> : null);

export default function LegacyHall() {
  const { setScreen, franchiseState } = useGameUI();
  const userTid = String(franchiseState?.user_team_id ?? franchiseState?.user_team?.team_id ?? "");
  const [section, setSection] = useState("franchise");
  const [teamId, setTeamId] = useState("");
  const [teams, setTeams] = useState([]);
  const [player, setPlayer] = useState(null);

  useEffect(() => {
    loadTeamIdentities().then((byKey) => {
      const seen = new Map();
      byKey?.forEach((t) => t?.team_id != null && seen.set(String(t.team_id), t));
      setTeams([...seen.values()].sort((a, b) => String(a.team_name).localeCompare(String(b.team_name))));
    });
    const f = takeLegacyFocus();
    if (f?.type === "player") {
      setSection("players");
      setPlayer(f.player_id);
    }
  }, []);
  const tid = teamId || userTid;
  const teamName = useCallback(
    (id) => {
      const t = teams.find((x) => String(x.team_id) === String(id));
      return t ? t.team_name || t.abbreviation : id ? `Team ${id}` : "–";
    },
    [teams]
  );
  const openPlayer = (pid) => {
    setPlayer(pid);
    setSection("players");
  };
  const cur = SECTIONS.find((s) => s.id === section) || SECTIONS[0];

  return (
    <div className="lgh">
      <header className="lgh-top">
        <button type="button" className="lgh-back" onClick={() => setScreen(SCREENS.HUB)}>
          ◀ Back
        </button>
        <div className="lgh-title">
          <span className="lgh-kicker">The permanent record of this save</span>
          <h1>Legacy Hall</h1>
        </div>
        <label className="lgh-teampick">
          <span>Franchise</span>
          <select value={tid} onChange={(e) => setTeamId(e.target.value)}>
            {teams.map((t) => (
              <option key={t.team_id} value={String(t.team_id)}>
                {t.team_name}
                {String(t.team_id) === userTid ? " (you)" : ""}
              </option>
            ))}
          </select>
        </label>
      </header>
      <div className="lgh-frame">
        <nav className="lgh-rail" aria-label="Legacy Hall sections">
          {SECTIONS.map((s) => (
            <button key={s.id} type="button" className={`lgh-cup ${section === s.id ? "is-on" : ""}`} aria-current={section === s.id ? "page" : undefined} onClick={() => setSection(s.id)}>
              <span className="lgh-cup__glyph" aria-hidden="true">
                {s.glyph}
              </span>
              <span className="lgh-cup__txt">
                <b>{s.label}</b>
                <small>{s.sub}</small>
              </span>
            </button>
          ))}
        </nav>
        <main className="lgh-main">
          <div className="lgh-sechead">
            <h2>{cur.label}</h2>
            <span>{tid ? teamName(tid) : ""}</span>
          </div>
          {section === "franchise" && tid && (
            <FranchiseHall
              tid={tid}
              fs={franchiseState}
              isUser={tid === userTid}
              openPlayer={openPlayer}
              onOpen={(link) => {
                if (link?.type === "player") openPlayer(link.player_id);
                else if (link?.type === "trade") setSection("trades");
                else if (link?.type === "rivalry") setSection("rivalries");
                else if (link?.type === "season") setSection("league");
              }}
            />
          )}
          {section === "players" && <PlayerLegacy tid={tid} player={player} setPlayer={setPlayer} teamName={teamName} isUser={tid === userTid} />}
          {section === "league" && <LeagueHistory teamName={teamName} openPlayer={openPlayer} />}
          {section === "drafts" && <DraftHistory tid={tid} teamName={teamName} openPlayer={openPlayer} />}
          {section === "trades" && <TradeHistory tid={tid} teamName={teamName} openPlayer={openPlayer} />}
          {section === "eras" && tid && <Eras tid={tid} />}
          {section === "rivalries" && <Rivalries tid={tid} />}
          {section === "archive" && <Archive tid={tid} openPlayer={openPlayer} />}
        </main>
      </div>
    </div>
  );
}

/* ───────────────────────── Franchise */

function Discovery({ tid, onOpen }) {
  const st = useLoad(() => legacyApi.trivia(tid), [tid]);
  const facts = st.data?.facts || [];
  if (st.loading || !facts.length) return null;
  return (
    <div className="lgh-trivia" aria-label="From the archives">
      <span className="lgh-trivia__tag">From the archives</span>
      {facts.slice(0, 3).map((f) => (
        <button key={f.id} type="button" className="lgh-trivia__card" onClick={() => onOpen(f.link)} disabled={!f.link} title={f.source ? `Source: ${f.source}` : undefined}>
          <small>{pretty(f.kind)}</small>
          <span>{f.text}</span>
        </button>
      ))}
    </div>
  );
}

function FranchiseHall({ tid, fs, isUser, openPlayer, onOpen }) {
  const st = useLoad(() => legacyApi.franchise(tid), [tid]);
  const lead = useLoad(() => legacyApi.leaders({ stat: "pts", team_id: tid, limit: 8 }), [tid]);
  const goal = useLoad(() => legacyApi.leaders({ stat: "w", team_id: tid, position: "G", limit: 5 }), [tid]);
  const d = st.data;
  if (!d) return <Status state={st} />;
  const rows = d.seasons || [];
  const maxPts = Math.max(1, ...rows.map((r) => r.pts || 0));
  const champs = d.championships || [];
  const records = [
    d.best_season && { k: "Best season", v: `${d.best_season.pts} pts`, s: d.best_season.label },
    d.worst_season && rows.length > 1 && { k: "Worst season", v: `${d.worst_season.pts} pts`, s: d.worst_season.label },
    d.longest_playoff_streak != null && { k: "Longest playoff run", v: d.longest_playoff_streak, s: "straight seasons" },
    d.current_playoff_drought ? { k: "Current drought", v: d.current_playoff_drought, s: "seasons", tone: "coral" } : null,
    d.most_games && { k: "Most games", v: d.most_games.gp, s: d.most_games.name, pid: d.most_games.player_id },
    d.longest_captain && { k: "Longest captaincy", v: d.longest_captain.seasons, s: d.longest_captain.name, pid: d.longest_captain.player_id },
    d.top_homegrown && { k: "Best homegrown", v: `${d.top_homegrown.pts} pts`, s: d.top_homegrown.name, pid: d.top_homegrown.player_id },
    d.longest_serving_active && { k: "Longest-serving now", v: d.longest_serving_active.seasons, s: d.longest_serving_active.name, pid: d.longest_serving_active.player_id },
  ].filter(Boolean);
  const retired = Object.entries(d.retired_numbers || {}).sort((a, b) => Number(a[0]) - Number(b[0]));
  return (
    <div className="lgh-dash">
      <div className="lgh-tiles">
        <Tile label="Championships" value={champs.length} sub={champs.length ? champs.map(seasonLabel).join(" · ") : "none yet"} tone="gold" />
        <Tile label="Playoff seasons" value={d.playoff_appearances ?? 0} sub={`of ${rows.length} tracked`} tone="sky" />
        <Tile label="Best season" value={d.best_season ? d.best_season.pts : "–"} sub={d.best_season ? `pts · ${d.best_season.label}` : "no seasons yet"} tone="lime" />
        <Tile
          label={d.current_playoff_drought ? "Playoff drought" : "Playoff streak"}
          value={d.current_playoff_drought || d.longest_playoff_streak || 0}
          sub="seasons"
          tone={d.current_playoff_drought ? "coral" : "orange"}
        />
      </div>
      <Discovery tid={tid} onOpen={onOpen} />
      <Rafters champs={champs} finals={d.finals || []} retired={retired} openPlayer={openPlayer} />
      {rows.length === 0 ? <SeasonNow fs={fs} tid={tid} isUser={isUser} /> : null}
      <Panel title="Season by season" className="is-wide" right={<Legend />}>
        {rows.length === 0 ? (
          <Status state={{}} empty={d.coverage_note || "No finished seasons yet."} />
        ) : (
          <div className="lgh-bars" role="list" aria-label="Points by season">
            {rows.map((r) => (
              <div
                key={r.season}
                role="listitem"
                className={`lgh-bar ${r.champion ? "is-champ" : r.made_playoffs ? "is-po" : ""}`}
                title={`${r.label}: ${r.w}-${r.l}-${r.otl}, ${r.pts} pts${r.round ? `, playoff round ${r.round}` : ""}`}
              >
                <span className="lgh-bar__val">{r.pts}</span>
                <span className="lgh-bar__fill" style={{ height: `${Math.max(4, Math.round(((r.pts || 0) / maxPts) * 100))}%` }}>
                  {r.champion ? <i aria-label="champion">★</i> : null}
                </span>
                <span className="lgh-bar__lab">{r.label}</span>
              </div>
            ))}
          </div>
        )}
        {rows.length ? <Note>{d.coverage_note}</Note> : null}
      </Panel>
      <Panel title="Franchise scoring leaders" tone="gold">
        <Podium st={lead} stat="pts" openPlayer={openPlayer} />
      </Panel>
      <Panel title="Goaltending leaders" tone="sky">
        <Podium st={goal} stat="w" openPlayer={openPlayer} />
      </Panel>
      <Panel title="Records & streaks">
        {records.length === 0 ? (
          <Status state={{}} empty="Records appear after the first full season." />
        ) : (
          <div className="lgh-recs">
            {records.map((r) => (
              <div key={r.k} className={`lgh-rec ${r.tone ? `is-${r.tone}` : ""} ${r.pid ? "is-link" : ""}`} {...act(r.pid ? () => openPlayer(r.pid) : null)}>
                <small>{r.k}</small>
                <b>{r.v}</b>
                <span>{r.s}</span>
              </div>
            ))}
          </div>
        )}
      </Panel>
      <Panel title="Milestones" className="is-wide">
        <Timeline events={d.recent_events || []} openPlayer={openPlayer} empty="Nothing on record yet." />
      </Panel>
    </div>
  );
}

/** Banners hanging over the ice: one per title and final, one per retired number; empty hooks until earned. */
function Rafters({ champs, finals, retired, openPlayer }) {
  const runnerUp = finals.filter((f) => !champs.includes(f));
  const banners = [
    ...champs.map((s) => ({ key: `c${s}`, kind: "cup", top: "Champions", big: seasonLabel(s) })),
    ...runnerUp.map((s) => ({ key: `f${s}`, kind: "final", top: "Finalist", big: seasonLabel(s) })),
    ...retired.map(([n, r]) => ({ key: `n${n}`, kind: "num", top: r.name || "", big: `#${n}`, pid: r.player_id })),
  ];
  const empties = Math.max(0, 6 - banners.length);
  return (
    <Panel title="The rafters" className="is-wide lgh-rafters-panel" right={<span className="lgh-chips"><span>{champs.length} cups</span><span>{retired.length} numbers retired</span></span>}>
      <div className="lgh-rafters">
        <span className="lgh-rafters__beam" aria-hidden="true" />
        {banners.map((b) => (
          <div key={b.key} className={`lgh-banner is-${b.kind} ${b.pid ? "is-link" : ""}`} {...act(b.pid ? () => openPlayer(b.pid) : null)}>
            <small>{b.top}</small>
            <b>{b.big}</b>
            {b.kind === "cup" ? <i aria-hidden="true">🏆</i> : null}
          </div>
        ))}
        {Array.from({ length: empties }, (_, i) => (
          <div key={`e${i}`} className="lgh-banner is-empty" aria-hidden="true">
            <small>{i === 0 && !banners.length ? "First banner" : "Open hook"}</small>
            <b>?</b>
          </div>
        ))}
      </div>
      {!banners.length ? <Note>Nothing hangs here yet. Win a Cup, reach a final or retire a legend's number and his banner goes up.</Note> : null}
    </Panel>
  );
}

/** Before the first finished season the Hall has no history: show the season being written right now. */
function SeasonNow({ fs, tid }) {
  const rows = Array.isArray(fs?.standings) ? fs.standings : Array.isArray(fs?.standings?.rows) ? fs.standings.rows : [];
  const row = rows.find((t) => String(t.team_id ?? t.id ?? "") === String(tid));
  const rec = String(tid) === String(fs?.user_team_id ?? "") ? fs?.team?.record || {} : {};
  const w = Number(row?.w ?? rec.w ?? 0), l = Number(row?.l ?? rec.l ?? 0), otl = Number(row?.otl ?? rec.otl ?? 0);
  const gp = Number(row?.gp ?? rec.gp ?? w + l + otl);
  const pts = Number(row?.pts ?? rec.pts ?? 2 * w + otl);
  const sorted = [...rows].sort((a, b) => Number(b.pts ?? 0) - Number(a.pts ?? 0));
  const rank = row ? sorted.indexOf(row) + 1 : null;
  const pace = gp ? Math.round((pts / gp) * 82) : null;
  const steps = [
    { done: gp > 0, txt: "Play the regular season" },
    { done: false, txt: "Finish the season: standings, leaders and awards are archived" },
    { done: false, txt: "Playoffs: series results start the rivalry files" },
    { done: false, txt: "The draft class is filed under Drafts" },
  ];
  return (
    <>
      <Panel title={`The season being written · ${seasonLabel(fs?.season_year) || "now"}`} tone="sky" className="lgh-now">
        <div className="lgh-recs">
          <div className="lgh-rec">
            <small>Record</small>
            <b>{w}-{l}-{otl}</b>
            <span>{gp} games played</span>
          </div>
          <div className="lgh-rec is-gold">
            <small>Points</small>
            <b>{pts}</b>
            <span>{pace != null ? `${pace}-point pace` : "no games yet"}</span>
          </div>
          <div className="lgh-rec">
            <small>League rank</small>
            <b>{rank ? `#${rank}` : "–"}</b>
            <span>{rows.length ? `of ${rows.length}` : "standings not loaded"}</span>
          </div>
        </div>
        <div className="lgh-progress" aria-label={`${gp} of 82 games`}>
          <i style={{ width: `${Math.min(100, Math.round((gp / 82) * 100))}%` }} />
          <span>{gp} / 82</span>
        </div>
      </Panel>
      <Panel title="How the Hall fills up" className="lgh-how">
        <ol className="lgh-steps">
          {steps.map((s, i) => (
            <li key={i} className={s.done ? "is-done" : ""}>
              <span>{s.done ? "✓" : i + 1}</span>
              {s.txt}
            </li>
          ))}
        </ol>
      </Panel>
    </>
  );
}

const Legend = () => (
  <span className="lgh-legend">
    <i className="is-champ" /> Champion <i className="is-po" /> Playoffs <i /> Missed
  </span>
);

function Podium({ st, stat, openPlayer }) {
  const rows = st.data?.rows || [];
  if (!st.data) return <Status state={st} />;
  if (!rows.length) return <Status state={{}} empty={st.data.coverage_note || "No one on the board yet."} />;
  const top = rows.slice(0, 3);
  const rest = rows.slice(3);
  const statLabel = STATS.find((s) => s[0] === stat)?.[1] || stat;
  return (
    <div className="lgh-podium">
      <div className="lgh-podium__top">
        {top.map((r, i) => (
          <div key={`${r.player_id}-${i}`} className={`lgh-step is-${i + 1}`} {...act(() => openPlayer(r.player_id))}>
            <span className="lgh-step__rank">{i + 1}</span>
            <b className="lgh-step__name">{r.name}</b>
            <span className="lgh-step__val">
              {r.value} <small>{statLabel.toLowerCase()}</small>
            </span>
            <span className="lgh-step__sub">
              {r.gp != null ? `${r.gp} GP` : ""}{r.season ? ` · ${seasonLabel(r.season)}` : ""}
              {r.retired_season ? " · retired" : ""}
            </span>
          </div>
        ))}
      </div>
      {rest.length ? <Board rows={rest} start={4} stat={stat} openPlayer={openPlayer} /> : null}
    </div>
  );
}

function Board({ rows, start = 1, stat, openPlayer }) {
  const max = Math.max(1, ...rows.map((r) => Math.abs(Number(r.value) || 0)));
  const statLabel = STATS.find((s) => s[0] === stat)?.[1] || stat;
  return (
    <ol className="lgh-board" start={start}>
      {rows.map((r, i) => (
        <li key={`${r.player_id}-${r.season || ""}-${i}`} className="lgh-board__row" {...act(() => openPlayer(r.player_id))}>
          <span className="lgh-board__rank">{start + i}</span>
          <span className="lgh-board__name">
            {r.name}
            {r.season ? <small> {seasonLabel(r.season)}</small> : null}
            {r.retired_season ? <small> · ret.</small> : null}
          </span>
          <span className="lgh-board__gp">{r.gp != null ? `${r.gp} GP` : ""}</span>
          <span className="lgh-board__meter" aria-hidden="true">
            <i style={{ width: `${Math.round((Math.abs(Number(r.value) || 0) / max) * 100)}%` }} />
          </span>
          <b className="lgh-board__val" title={statLabel}>
            {r.value}
          </b>
        </li>
      ))}
    </ol>
  );
}

function Timeline({ events, openPlayer, empty }) {
  if (!events.length) return <Status state={{}} empty={empty} />;
  return (
    <ol className="lgh-timeline">
      {events.map((e) => {
        const pid = e.player_ids?.[0];
        return (
          <li key={e.id} className={`lgh-tl is-${e.type || "event"} ${pid ? "is-link" : ""}`} {...act(pid ? () => openPlayer(pid) : null)}>
            <span className="lgh-tl__season">{seasonLabel(e.season)}</span>
            <span className="lgh-tl__txt">
              {e.headline || pretty(e.type)}
              {e.type ? <small>{pretty(e.type)}</small> : null}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

/* ───────────────────────── Players */

function PlayerLegacy({ tid, player, setPlayer, teamName, isUser }) {
  const [stat, setStat] = useState("pts");
  const [scope, setScope] = useState("career");
  const [mine, setMine] = useState(false);
  const [page, setPage] = useState(0);
  const st = useLoad(
    () => legacyApi.leaders({ stat, scope, team_id: mine ? tid : undefined, position: stat === "w" || stat === "so" ? "G" : undefined, limit: 25, offset: page * 25 }),
    [stat, scope, mine, tid, page]
  );
  const rows = st.data?.rows || [];
  return (
    <div className="lgh-split">
      <Panel title="Leaderboard" tone="gold">
        <div className="lgh-filters">
          <Pills label="Stat" value={stat} options={STATS} onChange={(k) => { setStat(k); setPage(0); }} />
          <div className="lgh-filters__row">
            <Pills label="Scope" value={scope} options={[["career", "Career"], ["season", "Single season"]]} onChange={(k) => { setScope(k); setPage(0); }} />
            <Pills label="Franchise" value={mine ? "mine" : "all"} options={[["all", "Whole league"], ["mine", "This franchise"]]} onChange={(k) => { setMine(k === "mine"); setPage(0); }} />
          </div>
        </div>
        {!st.data ? <Status state={st} /> : rows.length ? <Board rows={rows} start={page * 25 + 1} stat={stat} openPlayer={setPlayer} /> : <Status state={{}} empty={st.data.coverage_note || "No one qualifies yet."} />}
        <Pager page={page} total={st.data?.total} per={25} onPage={setPage} />
      </Panel>
      {player ? (
        <Dossier pid={player} teamName={teamName} tid={tid} isUser={isUser} />
      ) : (
        <Panel title="Career dossier">
          <Status state={{}} empty="Pick a player on the board to open his career." />
        </Panel>
      )}
    </div>
  );
}

function Dossier({ pid, teamName, tid, isUser }) {
  const [n, setN] = useState(0);
  const st = useLoad(() => legacyApi.player(pid), [pid, n]);
  const [num, setNum] = useState("");
  const [msg, setMsg] = useState("");
  const d = st.data;
  if (!d) return <Panel title="Career dossier"><Status state={st} /></Panel>;
  if (!d.found) return <Panel title="Career dossier"><Status state={{}} empty="No history on record for this player." /></Panel>;
  const c = d.career_in_save || {};
  const goalie = isG(d.position);
  const playedHere = (d.teams || []).some((t) => String(t.team_id) === String(tid));
  const seasons = d.seasons || [];
  const peak = Math.max(1, ...seasons.map((s) => (goalie ? s.w : s.pts) || 0));
  const status = d.active ? "Active" : d.retired_season ? `Retired ${seasonLabel(d.retired_season)}` : "Not on a roster";
  return (
    <section className="lgh-panel lgh-dossier">
      <header className="lgh-dossier__banner">
        <span className="lgh-dossier__pos">{d.position || "–"}</span>
        <div>
          <h3>{d.name}</h3>
          <div className="lgh-badges">
            <span className={`lgh-badge ${d.active ? "is-lime" : ""}`}>{status}</span>
            {d.hall_of_fame ? <span className="lgh-badge is-gold">Hall of Fame</span> : null}
            {(d.championships || []).map((s) => (
              <span key={s} className="lgh-badge is-gold">
                ★ {seasonLabel(s)}
              </span>
            ))}
          </div>
        </div>
      </header>
      <div className="lgh-panel__body">
        <div className="lgh-statline">
          {(goalie ? [["gp", "GP"], ["w", "Wins"], ["so", "SO"], ["ga", "GA"]] : [["gp", "GP"], ["g", "Goals"], ["a", "Assists"], ["pts", "Points"], ["plus_minus", "+/-"]]).map(([k, l]) => (
            <Tile key={k} label={l} value={c[k] ?? 0} tone={k === "pts" || k === "w" ? "gold" : ""} />
          ))}
        </div>
        <div className="lgh-facts">
          <div>
            <small>Draft</small>
            <span>{d.draft ? `${seasonLabel(d.draft.season)} · round ${d.draft.round ?? "?"}${d.draft.overall ? `, #${d.draft.overall}` : ""} · ${teamName(d.draft.team_id)}` : "Not recorded in this save"}</span>
          </div>
          {d.awards?.length ? (
            <div>
              <small>Awards</small>
              <span>{d.awards.map((a) => `${a.award} ${seasonLabel(a.season)}`).join(" · ")}</span>
            </div>
          ) : null}
          {d.captaincy?.length ? (
            <div>
              <small>Letter</small>
              <span>{d.captaincy.map((t) => `${t.role === "C" ? "Captain" : "Alternate"}, ${teamName(t.team_id)} ${seasonLabel(t.start_season)}–${t.end_season ? seasonLabel(t.end_season) : "now"}`).join(" · ")}</span>
            </div>
          ) : null}
        </div>
        {seasons.length ? (
          <table className="lgh-table">
            <thead>
              <tr>
                <th>Season</th>
                <th>Team</th>
                <th>GP</th>
                {goalie ? (
                  <>
                    <th>W</th>
                    <th>SO</th>
                  </>
                ) : (
                  <>
                    <th>G</th>
                    <th>A</th>
                    <th>PTS</th>
                  </>
                )}
                <th className="is-meter" aria-label="Season output" />
              </tr>
            </thead>
            <tbody>
              {seasons.map((s) => {
                const v = (goalie ? s.w : s.pts) || 0;
                return (
                  <tr key={`${s.season}-${s.team_id}`} className={v === peak ? "is-peak" : ""}>
                    <td>{seasonLabel(s.season)}</td>
                    <td>{teamName(s.team_id)}</td>
                    <td>{s.gp}</td>
                    {goalie ? (
                      <>
                        <td>
                          <b>{s.w ?? 0}</b>
                        </td>
                        <td>{s.so ?? 0}</td>
                      </>
                    ) : (
                      <>
                        <td>{s.g}</td>
                        <td>{s.a}</td>
                        <td>
                          <b>{s.pts}</b>
                        </td>
                      </>
                    )}
                    <td className="is-meter">
                      <i style={{ width: `${Math.round((v / peak) * 100)}%` }} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : null}
        {(d.imported_seasons || []).length ? (
          <details className="lgh-details">
            <summary>Before this save · {d.imported_seasons.length} imported seasons</summary>
            <table className="lgh-table">
              <tbody>
                {d.imported_seasons.map((s, i) => (
                  <tr key={i}>
                    <td>{s.season}</td>
                    <td>{s.team}</td>
                    <td>{s.league}</td>
                    <td>{s.gp ?? "–"} GP</td>
                    <td>{s.pts ?? s.w ?? "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        ) : null}
        {(d.trades || []).length || (d.events || []).length ? (
          <ol className="lgh-timeline is-compact">
            {(d.trades || []).map((t) => (
              <li key={t.trade_id} className="lgh-tl is-trade">
                <span className="lgh-tl__season">{seasonLabel(t.season)}</span>
                <span className="lgh-tl__txt">
                  {t.headline || (t.teams || []).map(teamName).join(" ⇄ ")}
                  <small>trade</small>
                </span>
              </li>
            ))}
            {(d.events || []).map((e) => (
              <li key={e.id} className={`lgh-tl is-${e.type}`}>
                <span className="lgh-tl__season">{seasonLabel(e.season)}</span>
                <span className="lgh-tl__txt">
                  {e.payload?.threshold ? `${e.payload.threshold} ${pretty(e.payload.stat)}` : pretty(e.type)}
                  <small>{pretty(e.type)}</small>
                </span>
              </li>
            ))}
          </ol>
        ) : null}
        <Note>{d.coverage?.note}</Note>
        {isUser && playedHere && !d.active ? (
          <div className="lgh-retire">
            <span>Raise his number to the rafters</span>
            <input type="number" min={0} max={99} placeholder="#" value={num} onChange={(e) => setNum(e.target.value)} aria-label="Jersey number" />
            <button
              type="button"
              className="lgh-cta"
              disabled={num === ""}
              onClick={async () => {
                try {
                  await legacyApi.retireNumber(pid, Number(num));
                  setMsg(`#${num} retired.`);
                  setN(n + 1);
                } catch (e) {
                  setMsg(apiError(e));
                }
              }}
            >
              Retire number
            </button>
            {msg ? <span className="lgh-note">{msg}</span> : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}

/* ───────────────────────── League */

function LeagueHistory({ teamName, openPlayer }) {
  const st = useLoad(() => legacyApi.league(), []);
  const [season, setSeason] = useState(null);
  const sv = useLoad(() => (season ? legacyApi.season(season) : Promise.resolve(null)), [season]);
  const d = st.data;
  useEffect(() => {
    if (d?.seasons?.length && season == null) setSeason(d.seasons[d.seasons.length - 1].season);
  }, [d, season]);
  if (!d) return <Status state={st} />;
  if (!d.seasons.length) return <Status state={{}} empty={d.coverage_note} />;
  const s = sv.data;
  return (
    <div className="lgh-split is-narrow-left">
      <div className="lgh-stack">
        <Panel title="Champions" tone="gold">
          <ol className="lgh-champs">
            {[...d.seasons].reverse().map((x) => (
              <li key={x.season} className={`lgh-champ ${season === x.season ? "is-on" : ""}`} {...act(() => setSeason(x.season))}>
                <span className="lgh-champ__season">{x.label}</span>
                <span className="lgh-champ__team">{x.champion_name || "not recorded"}</span>
                {x.source === "backfill" ? <small>partial</small> : null}
              </li>
            ))}
          </ol>
        </Panel>
        {d.titles_by_team.length ? (
          <Panel title="Titles by franchise">
            <ol className="lgh-board is-plain">
              {d.titles_by_team.map((x, i) => (
                <li key={x.team_id} className="lgh-board__row">
                  <span className="lgh-board__rank">{i + 1}</span>
                  <span className="lgh-board__name">{x.name}</span>
                  <span className="lgh-cups" aria-label={`${x.titles.length} titles`}>
                    {"★".repeat(Math.min(10, x.titles.length))}
                  </span>
                </li>
              ))}
            </ol>
          </Panel>
        ) : null}
        {d.dynasties.length ? (
          <Panel title="Dynasties" tone="orange">
            {d.dynasties.map((x) => (
              <div key={x.team_id} className="lgh-rec is-gold">
                <small>3+ titles in 5 seasons</small>
                <b>{x.name}</b>
                <span>{x.titles.map(seasonLabel).join(" · ")}</span>
              </div>
            ))}
          </Panel>
        ) : null}
        {d.hall_of_fame.length ? (
          <Panel title="Hall of Fame" tone="sky">
            <ol className="lgh-timeline is-compact">
              {d.hall_of_fame.map((h) => (
                <li key={h.player_id} className="lgh-tl is-hall_of_fame is-link" {...act(() => openPlayer(h.player_id))}>
                  <span className="lgh-tl__season">{seasonLabel(h.inducted_season)}</span>
                  <span className="lgh-tl__txt">
                    {h.name}
                    <small>{(h.reasons || []).join(", ")}</small>
                  </span>
                </li>
              ))}
            </ol>
          </Panel>
        ) : null}
        <Note>{d.coverage_note}</Note>
      </div>
      <div className="lgh-stack">
        {!season ? null : !s ? (
          <Status state={sv} />
        ) : !s.found ? (
          <Status state={{}} empty="Season not on record." />
        ) : (
          <>
            <div className="lgh-tiles">
              <Tile label="Season" value={s.label} tone="sky" />
              <Tile label="Champion" value={s.champion_id ? s.team_names?.[s.champion_id] || teamName(s.champion_id) : "–"} tone="gold" />
              <Tile label="Finalist" value={s.finalist_id ? s.team_names?.[s.finalist_id] || teamName(s.finalist_id) : "–"} />
            </div>
            <div className="lgh-trio">
              {[["pts", "Points"], ["g", "Goals"], ["w", "Goalie wins"]].map(([k, l]) => (
                <Panel key={k} title={`${l} leaders`}>
                  {(s.leaders?.[k] || []).length ? <Board rows={s.leaders[k]} stat={k} openPlayer={openPlayer} /> : <Status state={{}} empty="Not recorded." />}
                </Panel>
              ))}
            </div>
            <Panel title="Final standings">
              <table className="lgh-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Team</th>
                    <th>W</th>
                    <th>L</th>
                    <th>OTL</th>
                    <th>PTS</th>
                    <th>Playoffs</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(s.standings || {})
                    .sort((a, b) => (a[1].league_rank || 99) - (b[1].league_rank || 99))
                    .map(([t, r]) => (
                      <tr key={t} className={t === String(s.champion_id) ? "is-champ" : r.made_playoffs ? "is-po" : ""}>
                        <td>{r.league_rank}</td>
                        <td>{s.team_names?.[t] || teamName(t)}</td>
                        <td>{r.w}</td>
                        <td>{r.l}</td>
                        <td>{r.otl}</td>
                        <td>
                          <b>{r.pts}</b>
                        </td>
                        <td>{r.made_playoffs == null ? "unknown" : r.made_playoffs ? (t === String(s.champion_id) ? "★ Champion" : `Round ${r.playoff_round_reached || 1}`) : "–"}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </Panel>
            {(s.awards || []).length ? (
              <Panel title="Awards" tone="orange">
                <div className="lgh-recs">
                  {s.awards.map((a) => (
                    <div key={a.award} className="lgh-rec">
                      <small>{a.award}</small>
                      <b>{a.player_name || "–"}</b>
                    </div>
                  ))}
                </div>
              </Panel>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}

/* ───────────────────────── Drafts */

function DraftHistory({ tid, teamName, openPlayer }) {
  const [mine, setMine] = useState(true);
  const st = useLoad(() => legacyApi.drafts(mine ? { team_id: tid } : {}), [tid, mine]);
  const d = st.data;
  return (
    <div className="lgh-stack">
      <div className="lgh-toolbar">
        <Pills label="Picks shown" value={mine ? "mine" : "all"} options={[["mine", "This franchise's picks"], ["all", "Every pick"]]} onChange={(k) => setMine(k === "mine")} />
      </div>
      {!d ? <Status state={st} /> : null}
      {d && d.classes.length === 0 ? <Status state={{}} empty="No draft classes on record yet. Drafts are archived when a season is finalized." /> : null}
      {d
        ? [...d.classes].reverse().map((c) => (
            <Panel
              key={c.season}
              title={`${c.label} draft`}
              right={
                <span className="lgh-chips">
                  <span>{c.size} picks</span>
                  <span>{c.class_gp} NHL GP</span>
                  <span>{c.class_pts} pts</span>
                  <span className="is-lime">{c.nhl_regulars} regulars</span>
                </span>
              }
            >
              <table className="lgh-table">
                <thead>
                  <tr>
                    <th>Pick</th>
                    <th>Rd</th>
                    <th>Player</th>
                    <th>By</th>
                    <th>NHL seasons</th>
                    <th>GP</th>
                    <th>PTS / W</th>
                    <th>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {c.picks.slice(0, mine ? 20 : 64).map((p, i) => {
                    const early = p.seasons_since_draft < 4;
                    const verdict = early ? "Too early" : p.nhl_gp >= 200 ? "Hit" : p.nhl_gp >= 40 ? "Played" : "Miss";
                    return (
                      <tr key={`${p.player_id || i}`} className={p.player_id ? "is-link" : ""} {...act(p.player_id ? () => openPlayer(p.player_id) : null)}>
                        <td>{p.overall ?? "–"}</td>
                        <td>{p.round ?? "–"}</td>
                        <td>
                          <b>{p.name}</b>
                        </td>
                        <td>{teamName(p.team_id)}</td>
                        <td>{p.nhl_seasons}</td>
                        <td>{p.nhl_gp}</td>
                        <td>{isG(p.position) ? p.nhl_w : p.nhl_pts}</td>
                        <td>
                          <span className={`lgh-verdict is-${verdict.replace(/\s/g, "").toLowerCase()}`}>{verdict}</span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </Panel>
          ))
        : null}
      {d ? <Note>{d.rule}</Note> : null}
    </div>
  );
}

/* ───────────────────────── Trades */

function TradeHistory({ tid, teamName, openPlayer }) {
  const [page, setPage] = useState(0);
  const st = useLoad(() => legacyApi.trades({ team_id: tid, limit: 15, offset: page * 15 }), [tid, page]);
  const d = st.data;
  if (!d) return <Status state={st} />;
  const nameThen = (t, x) => t.team_names_then?.[x] || teamName(x);
  const line = (m, k) => {
    const s = m[k] || {};
    return isG(m.position) ? `${s.gp ?? 0} GP · ${s.w ?? 0} W` : `${s.gp ?? 0} GP · ${s.pts ?? 0} pts`;
  };
  return (
    <div className="lgh-stack">
      {d.trades.length === 0 ? <Status state={{}} empty="No trades on record for this franchise." /> : null}
      {d.trades.map((t) => {
        const sides = (t.teams || []).map((x) => ({ id: x, players: t.players.filter((m) => String(m.to_team_id) === String(x)), picks: (t.picks || []).filter((p) => String(p.to_team_id) === String(x)) }));
        return (
          <Panel key={t.trade_id} title={t.headline || sides.map((s) => nameThen(t, s.id)).join(" ⇄ ")} right={<span className="lgh-chips"><span>{seasonLabel(t.season)}</span>{t.iso ? <span>{t.iso}</span> : null}</span>}>
            <div className={`lgh-deal ${sides.length > 2 ? "is-multi" : ""}`}>
              {sides.map((s, i) => (
                <React.Fragment key={s.id}>
                  {i > 0 ? <span className="lgh-deal__swap" aria-hidden="true">⇄</span> : null}
                  <div className={`lgh-deal__side ${String(s.id) === String(tid) ? "is-mine" : ""}`}>
                    <h4>{nameThen(t, s.id)} got</h4>
                    {s.players.map((m, j) => (
                      <div key={`${m.player_id}-${j}`} className="lgh-deal__player" {...act(() => openPlayer(m.player_id))}>
                        <b>{m.name || m.player_id}</b>
                        <span>
                          Before: {line(m, "before")}
                        </span>
                        <span className="is-after">
                          After: {line(m, "after_with_new_team")}
                          {m.seasons_after ? "" : " · no full season yet"}
                        </span>
                      </div>
                    ))}
                    {s.picks.map((p, j) => (
                      <div key={`pk${j}`} className="lgh-deal__pick">
                        {p.season || "?"} round {p.round || "?"} pick
                      </div>
                    ))}
                    {!s.players.length && !s.picks.length ? <span className="lgh-note">Nothing recorded</span> : null}
                    {typeof t.original_valuation?.[s.id] === "number" ? <span className="lgh-deal__val">Valued at the time: {t.original_valuation[s.id].toFixed(1)}</span> : null}
                  </div>
                </React.Fragment>
              ))}
            </div>
          </Panel>
        );
      })}
      <Pager page={page} total={d.total} per={15} onPage={setPage} newer="Newer" older="Older" />
      <Note>{d.note}</Note>
    </div>
  );
}

/* ───────────────────────── Eras */

function Eras({ tid }) {
  const st = useLoad(() => legacyApi.franchise(tid), [tid]);
  const eras = st.data?.eras?.eras || [];
  if (!st.data) return <Status state={st} />;
  if (!eras.length) return <Status state={{}} empty="Eras appear once seasons are on record." />;
  return (
    <div className="lgh-stack">
      <Panel title="The franchise story">
        <div className="lgh-eras">
          {eras.map((e, i) => (
            <div key={i} className={`lgh-era is-${e.tier}`} style={{ flexGrow: Math.max(1, e.seasons.length) }}>
              <b>{e.label}</b>
              <span>{e.seasons.length} {e.seasons.length === 1 ? "season" : "seasons"}</span>
            </div>
          ))}
        </div>
      </Panel>
      <div className="lgh-cards">
        {eras.map((e, i) => (
          <Panel key={i} title={e.label} tone={{ contender: "gold", playoff: "sky", bubble: "orange", rebuild: "coral" }[e.tier] || ""}>
            <div className="lgh-recs">
              <div className="lgh-rec">
                <small>Seasons</small>
                <b>{e.evidence.seasons}</b>
              </div>
              <div className="lgh-rec">
                <small>Points %</small>
                <b>{Math.round(e.evidence.avg_pts_pct * 1000) / 10}</b>
              </div>
              <div className="lgh-rec">
                <small>Playoffs</small>
                <b>{e.evidence.playoffs}</b>
              </div>
              <div className="lgh-rec is-gold">
                <small>Titles</small>
                <b>{e.evidence.championships.length}</b>
              </div>
            </div>
            {e.markers.length ? (
              <ol className="lgh-timeline is-compact">
                {e.markers.map((m) => (
                  <li key={m.id} className="lgh-tl">
                    <span className="lgh-tl__season">{seasonLabel(m.season)}</span>
                    <span className="lgh-tl__txt">{m.headline}</span>
                  </li>
                ))}
              </ol>
            ) : null}
          </Panel>
        ))}
      </div>
      <Note>{st.data.eras.rule}</Note>
    </div>
  );
}

/* ───────────────────────── Rivalries */

function Rivalries({ tid }) {
  const st = useLoad(() => legacyApi.rivalries(tid), [tid]);
  const d = st.data;
  if (!d) return <Status state={st} />;
  return (
    <div className="lgh-stack">
      {d.rivalries.length === 0 ? <Status state={{}} empty="No rivalry qualifies yet: it takes a playoff series or a dozen one-goal games." /> : null}
      <div className="lgh-cards">
        {d.rivalries.map((r) => (
          <section key={r.teams.join("|")} className={`lgh-panel lgh-rival ${r.score >= 70 ? "is-hot" : r.score >= 40 ? "is-warm" : ""}`}>
            <header className="lgh-rival__head">
              <b>{r.names[0]}</b>
              <span className="lgh-rival__vs">VS</span>
              <b>{r.names[1]}</b>
            </header>
            <div className="lgh-panel__body">
              <div className="lgh-heat" aria-label={`Rivalry heat ${r.score} of 100`}>
                <i style={{ width: `${Math.max(3, Math.min(100, r.score))}%` }} />
                <span>{r.score}</span>
              </div>
              <div className="lgh-recs">
                <div className="lgh-rec">
                  <small>Playoff series</small>
                  <b>{r.inputs.series}</b>
                </div>
                <div className="lgh-rec">
                  <small>Went seven</small>
                  <b>{r.inputs.seven_game}</b>
                </div>
                <div className="lgh-rec is-gold">
                  <small>Finals</small>
                  <b>{r.inputs.finals}</b>
                </div>
                <div className="lgh-rec">
                  <small>One-goal games</small>
                  <b>
                    {r.inputs.one_goal}
                    <em>/{r.inputs.gp}</em>
                  </b>
                </div>
              </div>
            </div>
          </section>
        ))}
      </div>
      <Note>How heat is scored: {d.formula}</Note>
    </div>
  );
}

/* ───────────────────────── Archive */

function Archive({ tid, openPlayer }) {
  const [type, setType] = useState("");
  const [page, setPage] = useState(0);
  const st = useLoad(() => legacyApi.timeline({ team_id: tid, type: type || undefined, limit: 30, offset: page * 30 }), [tid, type, page]);
  const opts = useMemo(() => EVENT_TYPES.map((t) => [t, t ? pretty(t) : "All events"]), []);
  const d = st.data;
  return (
    <div className="lgh-stack">
      <div className="lgh-toolbar">
        <Pills label="Event type" value={type} options={opts} onChange={(k) => { setType(k); setPage(0); }} />
      </div>
      <Panel title={d ? `${d.total ?? d.events.length} events on file` : "Events"}>
        {!d ? <Status state={st} /> : <Timeline events={d.events} openPlayer={openPlayer} empty="No events of this kind on record." />}
        <Pager page={page} total={d?.total} per={30} onPage={setPage} newer="Newer" older="Older" />
      </Panel>
    </div>
  );
}
