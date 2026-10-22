/**
 * Legacy Hall: the save's archive. Every section loads one slice from
 * /api/franchise/legacy/* when opened; nothing here is on the franchise state payload.
 * Empty sections say why (no seasons finished yet, data not tracked) instead of filling in.
 */
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { useGameUI } from "../game/GameUIContext";
import { SCREENS } from "../game/constants";
import { loadTeamIdentities } from "../components/teamIdentity/teamIdentityStore";
import { apiError, legacyApi, seasonLabel, takeLegacyFocus } from "../components/legacy/legacyApi";
import "../components/legacy/legacy.css";

const SECTIONS = [
  { id: "franchise", label: "Franchise Hall" },
  { id: "players", label: "Player Legacy" },
  { id: "league", label: "League History" },
  { id: "drafts", label: "Draft History" },
  { id: "trades", label: "Trades" },
  { id: "eras", label: "Eras" },
  { id: "rivalries", label: "Rivalries" },
  { id: "archive", label: "Archive" },
];
const STATS = [["pts", "Points"], ["g", "Goals"], ["a", "Assists"], ["gp", "Games"], ["plus_minus", "+/-"], ["w", "Goalie wins"], ["so", "Shutouts"]];

function useLoad(fn, deps) {
  const [state, setState] = useState({ data: null, err: "", loading: true });
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, err: "" }));
    fn()
      .then((data) => alive && setState({ data, err: "", loading: false }))
      .catch((e) => alive && setState({ data: null, err: apiError(e), loading: false }));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

function Status({ state, empty }) {
  if (state.loading) return <div className="lh-empty">Pulling the files…</div>;
  if (state.err) return <div className="lh-empty">{state.err}</div>;
  if (empty) return <div className="lh-empty">{empty}</div>;
  return null;
}

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
  const teamName = useCallback((id) => {
    const t = teams.find((x) => String(x.team_id) === String(id));
    return t ? t.team_name || t.abbreviation : id;
  }, [teams]);
  const openPlayer = (pid) => { setPlayer(pid); setSection("players"); };

  return (
    <div className="lh-screen lh-screen--hall">
      <header className="lh-rail">
        <button type="button" className="dd-btn dd-btn--secondary lh-back" onClick={() => setScreen(SCREENS.HUB)}>Back</button>
        <div className="lh-rail-title">
          <span className="lh-eyebrow">The permanent record of this save</span>
          <h1 className="dd-title lh-h1">Legacy Hall</h1>
        </div>
        <label className="lh-team-pick">
          <span>Franchise</span>
          <select value={tid} onChange={(e) => setTeamId(e.target.value)}>
            {teams.map((t) => <option key={t.team_id} value={String(t.team_id)}>{t.team_name}{String(t.team_id) === userTid ? " (you)" : ""}</option>)}
          </select>
        </label>
        <nav className="lh-tabs" aria-label="Legacy Hall sections">
          {SECTIONS.map((s) => (
            <button key={s.id} type="button" className={`lh-tab ${section === s.id ? "is-on" : ""}`} onClick={() => setSection(s.id)}>{s.label}</button>
          ))}
        </nav>
      </header>
      {tid ? <Discovery tid={tid} onOpen={(link) => {
        if (link?.type === "player") openPlayer(link.player_id);
        else if (link?.type === "trade") setSection("trades");
        else if (link?.type === "rivalry") setSection("rivalries");
        else if (link?.type === "season") setSection("league");
      }} /> : null}
      <main className="lh-hall-body">
        {section === "franchise" && tid && <FranchiseHall tid={tid} openPlayer={openPlayer} />}
        {section === "players" && <PlayerLegacy tid={tid} player={player} setPlayer={setPlayer} teamName={teamName} isUser={tid === userTid} />}
        {section === "league" && <LeagueHistory teamName={teamName} openPlayer={openPlayer} />}
        {section === "drafts" && <DraftHistory tid={tid} teamName={teamName} openPlayer={openPlayer} />}
        {section === "trades" && <TradeHistory tid={tid} teamName={teamName} openPlayer={openPlayer} />}
        {section === "eras" && tid && <Eras tid={tid} />}
        {section === "rivalries" && <Rivalries tid={tid} />}
        {section === "archive" && <Archive tid={tid} openPlayer={openPlayer} />}
      </main>
    </div>
  );
}

function Discovery({ tid, onOpen }) {
  const st = useLoad(() => legacyApi.trivia(tid), [tid]);
  const facts = st.data?.facts || [];
  if (st.loading || !facts.length) return null;
  return (
    <section className="lh-discovery" aria-label="From the archives">
      {facts.slice(0, 3).map((f) => (
        <button key={f.id} type="button" className="lh-trivia" onClick={() => onOpen(f.link)} disabled={!f.link}>
          <span className="lh-eyebrow">{f.kind}</span>
          <span className="lh-trivia-text">{f.text}</span>
          <span className="lh-muted">Source: {f.source}</span>
        </button>
      ))}
    </section>
  );
}

function FranchiseHall({ tid, openPlayer }) {
  const st = useLoad(() => legacyApi.franchise(tid), [tid]);
  const lead = useLoad(() => legacyApi.leaders({ stat: "pts", team_id: tid, limit: 8 }), [tid]);
  const goal = useLoad(() => legacyApi.leaders({ stat: "w", team_id: tid, position: "G", limit: 5 }), [tid]);
  const d = st.data;
  if (!d) return <Status state={st} />;
  const rows = d.seasons || [];
  const maxPts = Math.max(1, ...rows.map((r) => r.pts));
  return (
    <div className="lh-grid-hall">
      <div className="lh-sheet lh-sheet--wide">
        <div className="lh-sheet-head">
          <span className="lh-brass">{d.championships.length} championships{d.championships.length ? ` · ${d.championships.map(seasonLabel).join(", ")}` : ""}</span>
          <span>{d.playoff_appearances} playoff seasons · {rows.length} seasons tracked</span>
        </div>
        {rows.length === 0 ? <p className="lh-muted">{d.coverage_note}</p> : (
          <div className="lh-season-strip" role="list" aria-label="Season by season">
            {rows.map((r) => (
              <div key={r.season} role="listitem" className={`lh-season-bar ${r.champion ? "is-champ" : r.made_playoffs ? "is-po" : ""}`} title={`${r.label}: ${r.w}-${r.l}-${r.otl}, ${r.pts} pts${r.round ? `, round ${r.round}` : ""}`}>
                <span className="lh-season-fill" style={{ height: `${Math.round((r.pts / maxPts) * 100)}%` }} />
                <span className="lh-season-pts">{r.pts}</span>
                <span className="lh-season-label">{r.label}</span>
              </div>
            ))}
          </div>
        )}
        <p className="lh-muted">{d.coverage_note} Gold = champion, cyan = playoffs.</p>
      </div>
      <div className="lh-sheet">
        <span className="lh-fact-label">Franchise scoring leaders</span>
        <LeaderTable st={lead} stat="pts" openPlayer={openPlayer} />
      </div>
      <div className="lh-sheet">
        <span className="lh-fact-label">Goaltending leaders (wins)</span>
        <LeaderTable st={goal} stat="w" openPlayer={openPlayer} />
      </div>
      <div className="lh-sheet">
        <span className="lh-fact-label">Records & streaks</span>
        <table className="lh-table"><tbody>
          {d.best_season ? <tr><td>Best season</td><td>{d.best_season.label} · {d.best_season.pts} pts</td></tr> : null}
          {d.worst_season && rows.length > 1 ? <tr><td>Worst season</td><td>{d.worst_season.label} · {d.worst_season.pts} pts</td></tr> : null}
          {d.longest_playoff_streak != null ? <tr><td>Longest playoff streak</td><td>{d.longest_playoff_streak}</td></tr> : null}
          {d.current_playoff_drought ? <tr><td>Current drought</td><td>{d.current_playoff_drought} seasons</td></tr> : null}
          {d.most_games ? <tr className="is-link" onClick={() => openPlayer(d.most_games.player_id)}><td>Most games</td><td>{d.most_games.name} · {d.most_games.gp}</td></tr> : null}
          {d.longest_captain ? <tr className="is-link" onClick={() => openPlayer(d.longest_captain.player_id)}><td>Longest captaincy</td><td>{d.longest_captain.name} · {d.longest_captain.seasons} seasons</td></tr> : null}
        </tbody></table>
      </div>
      <div className="lh-sheet">
        <span className="lh-fact-label">Milestones</span>
        {(d.recent_events || []).length === 0 ? <p className="lh-muted">Nothing on record yet.</p> : null}
        {(d.recent_events || []).map((e) => (
          <div key={e.id} className={`lh-event ${e.player_ids?.[0] ? "is-link" : ""}`} onClick={() => e.player_ids?.[0] && openPlayer(e.player_ids[0])}>
            <b>{seasonLabel(e.season)}</b> {e.headline}
          </div>
        ))}
      </div>
    </div>
  );
}

function LeaderTable({ st, stat, openPlayer }) {
  const rows = st.data?.rows || [];
  if (!st.data) return <Status state={st} />;
  if (!rows.length) return <p className="lh-muted">{st.data.coverage_note}</p>;
  return (
    <table className="lh-table">
      <thead><tr><th>#</th><th>Player</th><th>GP</th><th>{STATS.find((s) => s[0] === stat)?.[1] || stat}</th></tr></thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={`${r.player_id}-${r.season || ""}`} className="is-link" onClick={() => openPlayer(r.player_id)}>
            <td>{i + 1}</td><td>{r.name}{r.season ? <span className="lh-muted"> {seasonLabel(r.season)}</span> : null}{r.retired_season ? <span className="lh-muted"> · ret.</span> : null}</td><td>{r.gp}</td><td><b>{r.value}</b></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PlayerLegacy({ tid, player, setPlayer, teamName, isUser }) {
  const [stat, setStat] = useState("pts");
  const [scope, setScope] = useState("career");
  const [mine, setMine] = useState(false);
  const [page, setPage] = useState(0);
  const st = useLoad(() => legacyApi.leaders({ stat, scope, team_id: mine ? tid : undefined, position: stat === "w" || stat === "so" ? "G" : undefined, limit: 25, offset: page * 25 }), [stat, scope, mine, tid, page]);
  return (
    <div className="lh-two lh-two--wide">
      <div className="lh-sheet">
        <div className="lh-filter-row">
          <select value={stat} onChange={(e) => { setStat(e.target.value); setPage(0); }}>{STATS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select>
          <select value={scope} onChange={(e) => { setScope(e.target.value); setPage(0); }}><option value="career">Career (in save)</option><option value="season">Single season</option></select>
          <label className="lh-field--check"><input type="checkbox" checked={mine} onChange={(e) => { setMine(e.target.checked); setPage(0); }} /> This franchise only</label>
        </div>
        <LeaderTable st={st} stat={stat} openPlayer={setPlayer} />
        {st.data && st.data.total > 25 ? (
          <div className="lh-row-actions">
            <button type="button" className="lh-chip" disabled={page === 0} onClick={() => setPage(page - 1)}>Prev</button>
            <span className="lh-muted">{page * 25 + 1}-{Math.min(st.data.total, (page + 1) * 25)} of {st.data.total}</span>
            <button type="button" className="lh-chip" disabled={(page + 1) * 25 >= st.data.total} onClick={() => setPage(page + 1)}>Next</button>
          </div>
        ) : null}
      </div>
      {player ? <Dossier pid={player} teamName={teamName} tid={tid} isUser={isUser} /> : <div className="lh-empty">Pick a player to open their career dossier.</div>}
    </div>
  );
}

function Dossier({ pid, teamName, tid, isUser }) {
  const [n, setN] = useState(0);
  const st = useLoad(() => legacyApi.player(pid), [pid, n]);
  const [num, setNum] = useState("");
  const [msg, setMsg] = useState("");
  const d = st.data;
  if (!d) return <Status state={st} />;
  if (!d.found) return <div className="lh-empty">No history on record for this player.</div>;
  const c = d.career_in_save || {};
  const goalie = String(d.position || "").startsWith("G");
  const playedHere = (d.teams || []).some((t) => String(t.team_id) === String(tid));
  return (
    <article className="lh-sheet lh-dossier">
      <div className="lh-sheet-head">
        <h2 className="lh-h2">{d.name}</h2>
        <span>{d.position} · {d.active ? "Active" : d.retired_season ? `Retired ${seasonLabel(d.retired_season)}` : "Not on a roster"}{d.hall_of_fame ? " · Hall of Fame" : ""}</span>
      </div>
      <div className="lh-dossier-facts">
        {d.draft ? <span>Drafted {seasonLabel(d.draft.season)}, round {d.draft.round ?? "?"}{d.draft.overall ? `, #${d.draft.overall}` : ""} by {teamName(d.draft.team_id)}</span> : <span className="lh-muted">Draft: not recorded in this save</span>}
        {d.championships?.length ? <span className="lh-brass">Champion {d.championships.map(seasonLabel).join(", ")}</span> : null}
        {d.awards?.length ? <span>{d.awards.map((a) => `${a.award} (${seasonLabel(a.season)})`).join(" · ")}</span> : null}
        {d.captaincy?.length ? <span>{d.captaincy.map((t) => `${t.role === "C" ? "Captain" : "Alternate"} of ${teamName(t.team_id)} ${seasonLabel(t.start_season)}–${t.end_season ? seasonLabel(t.end_season) : "now"}`).join(" · ")}</span> : null}
      </div>
      <div className="lh-statline">
        {(goalie ? [["gp", "GP"], ["w", "W"], ["so", "SO"], ["ga", "GA"]] : [["gp", "GP"], ["g", "G"], ["a", "A"], ["pts", "PTS"], ["plus_minus", "+/-"]]).map(([k, l]) => (
          <div key={k}><span className="lh-fact-label">{l}</span><b>{c[k] ?? 0}</b></div>
        ))}
      </div>
      <table className="lh-table">
        <thead><tr><th>Season</th><th>Team</th><th>GP</th>{goalie ? <><th>W</th><th>SO</th></> : <><th>G</th><th>A</th><th>PTS</th></>}</tr></thead>
        <tbody>
          {(d.seasons || []).map((s) => (
            <tr key={`${s.season}-${s.team_id}`}><td>{seasonLabel(s.season)}</td><td>{teamName(s.team_id)}</td><td>{s.gp}</td>
              {goalie ? <><td>{s.w ?? 0}</td><td>{s.so ?? 0}</td></> : <><td>{s.g}</td><td>{s.a}</td><td><b>{s.pts}</b></td></>}</tr>
          ))}
        </tbody>
      </table>
      {(d.imported_seasons || []).length ? (
        <details>
          <summary className="lh-muted">Before this save ({d.imported_seasons.length} imported seasons)</summary>
          <table className="lh-table"><tbody>
            {d.imported_seasons.map((s, i) => <tr key={i}><td>{s.season}</td><td>{s.team}</td><td>{s.league}</td><td>{s.gp ?? "–"} GP</td><td>{s.pts ?? s.w ?? "–"}</td></tr>)}
          </tbody></table>
        </details>
      ) : null}
      {(d.trades || []).length ? <div><span className="lh-fact-label">Traded</span>{d.trades.map((t) => <div key={t.trade_id} className="lh-event">{seasonLabel(t.season)} · {t.headline || t.teams.map(teamName).join(" ⇄ ")}</div>)}</div> : null}
      {(d.events || []).length ? <div><span className="lh-fact-label">Milestones</span>{d.events.map((e) => <div key={e.id} className="lh-event">{seasonLabel(e.season)} · {e.type.replace(/_/g, " ")} {e.payload?.threshold ? `${e.payload.threshold} ${e.payload.stat}` : ""}</div>)}</div> : null}
      <p className="lh-muted">{d.coverage?.note}</p>
      {isUser && playedHere && !d.active ? (
        <div className="lh-preset-row">
          <input type="number" min={0} max={99} placeholder="#" value={num} onChange={(e) => setNum(e.target.value)} />
          <button type="button" className="lh-chip" disabled={num === ""} onClick={async () => {
            try {
              await legacyApi.retireNumber(pid, Number(num));
              setMsg(`#${num} retired.`);
              setN(n + 1);
            } catch (e) { setMsg(apiError(e)); }
          }}>Retire number</button>
          {msg ? <span className="lh-muted">{msg}</span> : null}
        </div>
      ) : null}
    </article>
  );
}

function LeagueHistory({ teamName, openPlayer }) {
  const st = useLoad(() => legacyApi.league(), []);
  const [season, setSeason] = useState(null);
  const sv = useLoad(() => (season ? legacyApi.season(season) : Promise.resolve(null)), [season]);
  const d = st.data;
  if (!d) return <Status state={st} />;
  if (!d.seasons.length) return <div className="lh-empty">{d.coverage_note}</div>;
  const s = sv.data;
  return (
    <div className="lh-two lh-two--wide">
      <div className="lh-sheet">
        <span className="lh-fact-label">Champions</span>
        {[...d.seasons].reverse().map((x) => (
          <div key={x.season} className={`lh-event is-link ${season === x.season ? "is-sel" : ""}`} onClick={() => setSeason(x.season)}>
            <b>{x.label}</b> {x.champion_name || "champion not recorded"}{x.source === "backfill" ? <span className="lh-muted"> · partial record</span> : null}
          </div>
        ))}
        {d.dynasties.length ? <><span className="lh-fact-label">Dynasties (3+ titles in 5 seasons)</span>{d.dynasties.map((x) => <div key={x.team_id} className="lh-event lh-brass">{x.name}: {x.titles.map(seasonLabel).join(", ")}</div>)}</> : null}
        {d.titles_by_team.length ? <><span className="lh-fact-label">Titles by franchise</span>{d.titles_by_team.map((x) => <div key={x.team_id} className="lh-event">{x.name} · {x.titles.length}</div>)}</> : null}
        {d.hall_of_fame.length ? <><span className="lh-fact-label">Hall of Fame</span>{d.hall_of_fame.map((h) => <div key={h.player_id} className="lh-event is-link" onClick={() => openPlayer(h.player_id)}>{h.name} · {seasonLabel(h.inducted_season)} · {h.reasons.join(", ")}</div>)}</> : null}
        <p className="lh-muted">{d.coverage_note}</p>
      </div>
      <div className="lh-sheet">
        {!season ? <p className="lh-muted">Pick a season.</p> : !s ? <Status state={sv} /> : !s.found ? <p className="lh-muted">Season not on record.</p> : (
          <>
            <div className="lh-sheet-head"><h2 className="lh-h2">{s.label}</h2><span>{s.champion_id ? `Champion: ${s.team_names?.[s.champion_id]}` : ""}{s.finalist_id ? ` · Finalist: ${s.team_names?.[s.finalist_id]}` : ""}</span></div>
            <div className="lh-two">
              {[["pts", "Points"], ["g", "Goals"], ["w", "Goalie wins"]].map(([k, l]) => (
                <div key={k}><span className="lh-fact-label">{l}</span>{(s.leaders?.[k] || []).map((r) => <div key={r.player_id} className="lh-event is-link" onClick={() => openPlayer(r.player_id)}>{r.name} <b>{r.value}</b></div>)}</div>
              ))}
            </div>
            <table className="lh-table">
              <thead><tr><th>#</th><th>Team</th><th>W</th><th>L</th><th>OTL</th><th>PTS</th><th>Playoffs</th></tr></thead>
              <tbody>
                {Object.entries(s.standings || {}).sort((a, b) => (a[1].league_rank || 99) - (b[1].league_rank || 99)).map(([t, r]) => (
                  <tr key={t} className={t === s.champion_id ? "is-champ" : ""}><td>{r.league_rank}</td><td>{s.team_names?.[t] || teamName(t)}</td><td>{r.w}</td><td>{r.l}</td><td>{r.otl}</td><td><b>{r.pts}</b></td>
                    <td>{r.made_playoffs == null ? "unknown" : r.made_playoffs ? `round ${r.playoff_round_reached || 1}` : "–"}</td></tr>
                ))}
              </tbody>
            </table>
            {(s.awards || []).length ? <><span className="lh-fact-label">Awards</span>{s.awards.map((a) => <div key={a.award} className="lh-event">{a.award}: {a.player_name || "–"}</div>)}</> : null}
          </>
        )}
      </div>
    </div>
  );
}

function DraftHistory({ tid, teamName, openPlayer }) {
  const [mine, setMine] = useState(true);
  const st = useLoad(() => legacyApi.drafts(mine ? { team_id: tid } : {}), [tid, mine]);
  const d = st.data;
  if (!d) return <Status state={st} />;
  return (
    <div className="lh-sheet lh-sheet--wide">
      <div className="lh-filter-row"><label className="lh-field--check"><input type="checkbox" checked={mine} onChange={(e) => setMine(e.target.checked)} /> This franchise's picks only</label></div>
      {d.classes.length === 0 ? <p className="lh-muted">No draft classes on record yet. Drafts are archived when a season is finalized.</p> : null}
      {[...d.classes].reverse().map((c) => (
        <div key={c.season} className="lh-draft-class">
          <div className="lh-sheet-head"><b>{c.label} draft</b><span>{c.size} picks · {c.class_gp} NHL games · {c.class_pts} pts · {c.nhl_regulars} regulars</span></div>
          <table className="lh-table">
            <thead><tr><th>Pick</th><th>Rd</th><th>Player</th><th>By</th><th>NHL seasons</th><th>GP</th><th>PTS / W</th><th></th></tr></thead>
            <tbody>
              {c.picks.slice(0, mine ? 20 : 64).map((p, i) => (
                <tr key={`${p.player_id || i}`} className={p.player_id ? "is-link" : ""} onClick={() => p.player_id && openPlayer(p.player_id)}>
                  <td>{p.overall ?? "–"}</td><td>{p.round ?? "–"}</td><td>{p.name}</td><td>{teamName(p.team_id)}</td><td>{p.nhl_seasons}</td><td>{p.nhl_gp}</td><td>{String(p.position || "").startsWith("G") ? p.nhl_w : p.nhl_pts}</td>
                  <td className="lh-muted">{p.seasons_since_draft < 4 ? "too early" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
      <p className="lh-muted">{d.rule}</p>
    </div>
  );
}

function TradeHistory({ tid, teamName, openPlayer }) {
  const [page, setPage] = useState(0);
  const st = useLoad(() => legacyApi.trades({ team_id: tid, limit: 15, offset: page * 15 }), [tid, page]);
  const d = st.data;
  if (!d) return <Status state={st} />;
  return (
    <div className="lh-sheet lh-sheet--wide">
      {d.trades.length === 0 ? <p className="lh-muted">No trades on record for this franchise.</p> : null}
      {d.trades.map((t) => (
        <div key={t.trade_id} className="lh-trade">
          <div className="lh-sheet-head"><b>{seasonLabel(t.season)} · {t.headline || t.teams.map((x) => t.team_names_then?.[x] || teamName(x)).join(" ⇄ ")}</b><span>{t.iso || ""}</span></div>
          <table className="lh-table">
            <thead><tr><th>Player</th><th>To</th><th>Before (GP / PTS or W)</th><th>With new club</th></tr></thead>
            <tbody>
              {t.players.map((m, i) => (
                <tr key={`${m.player_id}-${i}`} className="is-link" onClick={() => openPlayer(m.player_id)}>
                  <td>{m.name || m.player_id}</td><td>{t.team_names_then?.[m.to_team_id] || teamName(m.to_team_id)}</td>
                  <td>{m.before?.gp ?? 0} / {String(m.position || "").startsWith("G") ? `${m.before?.w ?? 0} W` : m.before?.pts ?? 0}</td><td>{m.after_with_new_team?.gp ?? 0} / {String(m.position || "").startsWith("G") ? `${m.after_with_new_team?.w ?? 0} W` : m.after_with_new_team?.pts ?? 0}{m.seasons_after ? "" : <span className="lh-muted"> (no full season yet)</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {t.picks?.length ? <div className="lh-muted">Picks: {t.picks.map((p) => `${p.season || "?"} R${p.round || "?"} → ${teamName(p.to_team_id)}`).join(", ")}</div> : null}
          {Object.keys(t.original_valuation || {}).length ? <div className="lh-muted">Valuation at the time: {Object.entries(t.original_valuation).map(([k, v]) => `${teamName(k)} ${typeof v === "number" ? v.toFixed(1) : JSON.stringify(v)}`).join(" · ")}</div> : null}
        </div>
      ))}
      <div className="lh-row-actions">
        <button type="button" className="lh-chip" disabled={page === 0} onClick={() => setPage(page - 1)}>Newer</button>
        <span className="lh-muted">{d.total} trades</span>
        <button type="button" className="lh-chip" disabled={(page + 1) * 15 >= d.total} onClick={() => setPage(page + 1)}>Older</button>
      </div>
      <p className="lh-muted">{d.note}</p>
    </div>
  );
}

function Eras({ tid }) {
  const st = useLoad(() => legacyApi.franchise(tid), [tid]);
  const eras = st.data?.eras?.eras || [];
  if (!st.data) return <Status state={st} />;
  if (!eras.length) return <div className="lh-empty">Eras appear once seasons are on record.</div>;
  return (
    <div className="lh-sheet lh-sheet--wide">
      <div className="lh-era-track">
        {eras.map((e, i) => (
          <div key={i} className={`lh-era lh-era--${e.tier}`} style={{ flexGrow: e.seasons.length }}>
            <b>{e.label}</b>
            <span>{e.evidence.seasons} seasons · avg pts% {Math.round(e.evidence.avg_pts_pct * 1000) / 10} · {e.evidence.playoffs} playoffs{e.evidence.championships.length ? ` · ${e.evidence.championships.length} titles` : ""}</span>
            {e.markers.map((m) => <span key={m.id} className="lh-muted">{seasonLabel(m.season)} {m.headline}</span>)}
          </div>
        ))}
      </div>
      <p className="lh-muted">{st.data.eras.rule}</p>
    </div>
  );
}

function Rivalries({ tid }) {
  const st = useLoad(() => legacyApi.rivalries(tid), [tid]);
  const d = st.data;
  if (!d) return <Status state={st} />;
  return (
    <div className="lh-sheet lh-sheet--wide">
      {d.rivalries.length === 0 ? <p className="lh-muted">No rivalry qualifies yet: it takes a playoff series or a dozen one-goal games.</p> : null}
      {d.rivalries.map((r) => (
        <div key={r.teams.join("|")} className="lh-rivalry">
          <div className="lh-rivalry-meter"><span style={{ width: `${r.score}%` }} /></div>
          <b>{r.names.join(" vs ")}</b>
          <span className="lh-muted">{r.score} · {r.inputs.series} series ({r.inputs.seven_game} went seven, {r.inputs.finals} finals) · {r.inputs.one_goal} one-goal games of {r.inputs.gp}</span>
        </div>
      ))}
      <p className="lh-muted">Formula: {d.formula}</p>
    </div>
  );
}

function Archive({ tid, openPlayer }) {
  const [type, setType] = useState("");
  const [page, setPage] = useState(0);
  const st = useLoad(() => legacyApi.timeline({ team_id: tid, type: type || undefined, limit: 30, offset: page * 30 }), [tid, type, page]);
  const types = useMemo(() => ["", "championship", "career_milestone", "franchise_record", "captaincy_start", "captaincy_end", "rebrand", "retirement", "hall_of_fame", "number_retired", "first_playoffs_in_save"], []);
  const d = st.data;
  return (
    <div className="lh-sheet lh-sheet--wide">
      <div className="lh-filter-row">
        <select value={type} onChange={(e) => { setType(e.target.value); setPage(0); }}>{types.map((t) => <option key={t} value={t}>{t ? t.replace(/_/g, " ") : "All events"}</option>)}</select>
      </div>
      {!d ? <Status state={st} /> : d.events.length === 0 ? <p className="lh-muted">No events of this kind on record.</p> : d.events.map((e) => (
        <div key={e.id} className={`lh-event ${e.player_ids?.[0] ? "is-link" : ""}`} onClick={() => e.player_ids?.[0] && openPlayer(e.player_ids[0])}>
          <b>{seasonLabel(e.season)}</b> {e.headline || e.type.replace(/_/g, " ")}{e.iso ? <span className="lh-muted"> · {e.iso}</span> : null}
        </div>
      ))}
      {d && d.total > 30 ? (
        <div className="lh-row-actions">
          <button type="button" className="lh-chip" disabled={page === 0} onClick={() => setPage(page - 1)}>Newer</button>
          <button type="button" className="lh-chip" disabled={(page + 1) * 30 >= d.total} onClick={() => setPage(page + 1)}>Older</button>
        </div>
      ) : null}
    </div>
  );
}
