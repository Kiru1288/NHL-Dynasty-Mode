import React from "react";
import { createPortal } from "react-dom";
import { getInternationalHub, getInternationalTournament } from "../../services/franchiseService";
import { OAR_FLAG_URL, flagApiUrl } from "../../utils/countryFlags";

/**
 * International hub: 4 Nations Face-Off, Winter Olympics, IIHF World Championship.
 * Backend: services/international_tournaments.py (GET /api/franchise/international[/id]).
 */

const KIND_ICON = { four_nations: "4N", olympics: "OLY", worlds: "IIHF" };
const STAGE_ORDER = ["qualification", "quarterfinal", "semifinal", "bronze", "final"];

function Flag({ nation, size = 28 }) {
  const iso = String(nation?.flag || "").toUpperCase();
  const src = iso === "OAR" ? OAR_FLAG_URL : iso ? flagApiUrl(iso, size >= 40 ? 64 : 48) : null;
  return src ? (
    <img className="intl-flag" src={src} alt={nation?.name || ""} width={size} height={Math.round(size * 0.72)} loading="lazy" />
  ) : (
    <span className="intl-flag is-blank" style={{ width: size, height: Math.round(size * 0.72) }}>
      {nation?.code || "?"}
    </span>
  );
}

function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(`${iso}T12:00:00`);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function statusChip(status) {
  if (status === "live") return <span className="intl-chip is-live">Live</span>;
  if (status === "complete") return <span className="intl-chip is-done">Final</span>;
  if (status === "ready") return <span className="intl-chip">Rosters set</span>;
  return <span className="intl-chip is-soon">Upcoming</span>;
}

export default function InternationalTournamentMenu({ onClose, initialId = "" }) {
  const [hub, setHub] = React.useState(null);
  const [tid, setTid] = React.useState(initialId);
  const [data, setData] = React.useState(null);
  const [tab, setTab] = React.useState("overview");
  const [nation, setNation] = React.useState("");
  const [openGame, setOpenGame] = React.useState("");
  const [error, setError] = React.useState("");

  React.useEffect(() => {
    let off = false;
    getInternationalHub()
      .then((h) => {
        if (off) return;
        setHub(h);
        if (!initialId) {
          const rows = h?.tournaments || [];
          const pick = rows.find((t) => t.status === "live") || rows.find((t) => t.status !== "complete") || rows[0];
          if (pick) setTid(pick.id);
        }
      })
      .catch((e) => setError(String(e?.message || "Couldn't load the international hub.")));
    return () => {
      off = true;
    };
  }, [initialId]);

  React.useEffect(() => {
    if (!tid) return undefined;
    let off = false;
    setData(null);
    getInternationalTournament(tid)
      .then((d) => {
        if (off) return;
        setData(d);
        const user = (d?.user_players || [])[0]?.nation;
        setNation((prev) => prev || user || (d?.teams || [])[0]?.code || "");
      })
      .catch((e) => setError(String(e?.message || "Couldn't load the tournament.")));
    return () => {
      off = true;
    };
  }, [tid]);

  React.useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose?.();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const teams = data?.teams || [];
  const byCode = Object.fromEntries(teams.map((t) => [t.code, t]));
  const nat = (code) => byCode[code] || { code, name: code, flag: "" };
  const games = data?.games || [];
  const tabs = [
    ["overview", "Overview"],
    ["schedule", data?.projected ? "Field" : "Games"],
    ["standings", "Standings"],
    ["bracket", "Playoffs"],
    ["rosters", "Rosters"],
    ["leaders", "Leaders"],
  ].filter(([k]) => !(data?.projected && ["bracket", "leaders"].includes(k)));

  const knockout = STAGE_ORDER.map((st) => ({ st, games: games.filter((g) => g.stage === st) })).filter((x) => x.games.length);
  const team = byCode[nation];
  const roster = team ? [...team.roster].sort((a, b) => ({ G: 2, D: 1, F: 0 }[a.pos] - { G: 2, D: 1, F: 0 }[b.pos]) || b.ovr - a.ovr) : [];
  const byDay = games.reduce((acc, g) => {
    (acc[g.date] = acc[g.date] || []).push(g);
    return acc;
  }, {});

  const body = (
    <div className="intl" role="dialog" aria-modal="true" aria-label="International hockey">
      <header className="intl-bar">
        <div className="intl-bar__title">
          <span>International hockey</span>
          <strong>{data?.name || "Best on best"}</strong>
        </div>
        <nav className="intl-pick" aria-label="Tournaments this season">
          {(hub?.tournaments || []).map((t) => (
            <button key={t.id} type="button" className={t.id === tid ? "is-on" : ""} onClick={() => { setTid(t.id); setTab("overview"); setNation(""); }}>
              <b>{KIND_ICON[t.kind]}</b>
              <span>
                {t.label}
                <small>
                  {fmtDate(t.start)}–{fmtDate(t.end)}
                </small>
              </span>
              {statusChip(t.status)}
            </button>
          ))}
        </nav>
        <button type="button" className="intl-close" onClick={onClose}>
          Back <b aria-hidden="true">✕</b>
        </button>
      </header>

      {error ? <p className="intl-error">{error}</p> : null}
      {!data ? (
        <p className="intl-loading">Loading…</p>
      ) : (
        <div className="intl-body">
          <section className="intl-hero">
            <Flag nation={data.host_nation} size={56} />
            <div>
              <h2>{data.label}</h2>
              <p>
                {data.host_label ? `${data.host_label} · ` : ""}
                {fmtDate(data.start)} – {fmtDate(data.end)}, {String(data.end || "").slice(0, 4)}
                {" · "}
                {teams.length} nations
              </p>
            </div>
            <div className="intl-hero__status">
              {statusChip(data.status)}
              {data.projected ? <small>Projected rosters until the tournament opens</small> : null}
              {data.status === "live" && data.upcoming?.length ? <small>{data.upcoming.length} games still to play</small> : null}
            </div>
          </section>

          <div className="intl-tabs" role="tablist">
            {tabs.map(([k, label]) => (
              <button key={k} type="button" role="tab" aria-selected={tab === k} className={tab === k ? "is-on" : ""} onClick={() => setTab(k)}>
                {label}
              </button>
            ))}
          </div>

          <div className="intl-panel">
            {tab === "overview" ? (
              <div className="intl-overview">
                {data.placements?.length ? (
                  <div className="intl-podium">
                    {[1, 0, 2].map((i) =>
                      data.placements[i] ? (
                        <div key={i} className={`intl-step is-${["gold", "silver", "bronze"][i]}`}>
                          <Flag nation={data.placements[i]} size={64} />
                          <b>{data.placements[i].name}</b>
                          <span>{data.kind === "four_nations" ? ["Champion", "Runner-up", "Third"][i] : ["Gold", "Silver", "Bronze"][i]}</span>
                        </div>
                      ) : null
                    )}
                  </div>
                ) : (
                  <div className="intl-field">
                    {Object.entries(data.groups || {}).map(([g, codes]) => (
                      <div key={g} className="intl-card">
                        <span className="intl-lab">{g === "RR" ? "Round robin" : `Group ${g}`}</span>
                        {codes.map((c) => (
                          <button key={c} type="button" className="intl-nation" onClick={() => { setNation(c); setTab("rosters"); }}>
                            <Flag nation={nat(c)} />
                            <b>{nat(c).name}</b>
                            <em>{nat(c).strength != null ? Math.round(nat(c).strength) : ""}</em>
                          </button>
                        ))}
                      </div>
                    ))}
                  </div>
                )}
                <div className="intl-side">
                  {data.mvp ? (
                    <div className="intl-card is-mvp">
                      <span className="intl-lab">Tournament MVP</span>
                      <b>{data.mvp.name}</b>
                      <small>
                        {nat(data.mvp.code).name}
                        {data.mvp.stats ? ` · ${data.mvp.stats.g}G ${data.mvp.stats.a}A` : ""}
                      </small>
                    </div>
                  ) : null}
                  {data.tension?.heated ? (
                    <div className="intl-card is-tension">
                      <span className="intl-lab">Political tension</span>
                      <b>
                        {data.tension.pair.map((c) => nat(c).name).join(" vs ")}
                      </b>
                      {data.tension.demands?.length ? (
                        <ul>
                          {data.tension.demands.map((d) => (
                            <li key={d.id} className={d.is_user ? "is-user" : ""}>
                              {d.name} ({d.team}) wants a trade to {d.wants}
                            </li>
                          ))}
                        </ul>
                      ) : data.tension.frost?.length ? (
                        <p>Frosty dressing rooms: {data.tension.frost.join(", ")}</p>
                      ) : (
                        <p>Heated, but no one has asked out yet.</p>
                      )}
                    </div>
                  ) : data.status !== "complete" ? (
                    <div className="intl-card">
                      <span className="intl-lab">Stakes</span>
                      <p>Medals lift morale and go on a player's record. There's a {hub?.tension_chance_pct ?? 20}% chance things get political enough to spill into the NHL.</p>
                    </div>
                  ) : null}
                  {data.user_players?.length ? (
                    <div className="intl-card">
                      <span className="intl-lab">Your players</span>
                      {data.user_players.map((p) => (
                        <div key={p.id} className="intl-mine">
                          <Flag nation={nat(p.nation)} size={22} />
                          <b>{p.name}</b>
                          <em>{p.pos}</em>
                          <small>{p.gp != null ? `${p.g || 0}G ${p.a || 0}A` : `${Math.round(p.ovr)} OVR`}</small>
                        </div>
                      ))}
                    </div>
                  ) : null}
                  {hub?.next_best_on_best?.length ? (
                    <div className="intl-card">
                      <span className="intl-lab">Next best-on-best</span>
                      {hub.next_best_on_best.map((n) => (
                        <p key={n.id}>
                          {n.name} · {fmtDate(n.start)}
                        </p>
                      ))}
                    </div>
                  ) : null}
                </div>
              </div>
            ) : null}

            {tab === "schedule" ? (
              data.projected ? (
                <p className="intl-empty">The schedule is set when the tournament opens. Check the Rosters tab for each nation's projected team.</p>
              ) : (
                <div className="intl-days">
                  {Object.entries(byDay).sort(([a], [b]) => (a < b ? -1 : 1)).map(([day, list]) => (
                    <div key={day} className="intl-day">
                      <span className="intl-lab">{fmtDate(day)}</span>
                      {list.map((g) => (
                        <GameRow key={g.id} g={g} nat={nat} open={openGame === g.id} onToggle={() => setOpenGame(openGame === g.id ? "" : g.id)} />
                      ))}
                    </div>
                  ))}
                  {data.upcoming?.length ? (
                    <div className="intl-day">
                      <span className="intl-lab">Still to come</span>
                      {data.upcoming.map((g) => (
                        <div key={g.id} className="intl-game is-soon">
                          <em>{fmtDate(g.date)} · {g.label}</em>
                          <span className="intl-gt">
                            <Flag nation={nat(g.home)} size={22} /> {nat(g.home).name}
                          </span>
                          <b>vs</b>
                          <span className="intl-gt">
                            {nat(g.away).name} <Flag nation={nat(g.away)} size={22} />
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              )
            ) : null}

            {tab === "standings" ? (
              <div className="intl-tables">
                {Object.entries(data.standings && Object.keys(data.standings).length ? data.standings : {}).map(([g, rows]) => (
                  <table key={g} className="intl-table">
                    <caption>{g === "RR" ? "Round robin" : `Group ${g}`}</caption>
                    <thead>
                      <tr>
                        <th>#</th><th>Nation</th><th>GP</th><th>W</th><th>OTW</th><th>OTL</th><th>L</th><th>GF</th><th>GA</th><th>PTS</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => (
                        <tr key={r.code}>
                          <td>{r.rank}</td>
                          <td className="intl-tn"><Flag nation={nat(r.code)} size={22} /> {nat(r.code).name}</td>
                          <td>{r.gp}</td><td>{r.w}</td><td>{r.otw}</td><td>{r.otl}</td><td>{r.l}</td><td>{r.gf}</td><td>{r.ga}</td>
                          <td><b>{r.pts}</b></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ))}
                {!Object.keys(data.standings || {}).length ? <p className="intl-empty">Standings appear once games are played.</p> : null}
              </div>
            ) : null}

            {tab === "bracket" ? (
              <div className="intl-bracket">
                {knockout.length ? (
                  knockout.map(({ st, games: gs }) => (
                    <div key={st} className="intl-round">
                      <span className="intl-lab">{gs[0].label}</span>
                      {gs.map((g) => (
                        <GameRow key={g.id} g={g} nat={nat} compact open={openGame === g.id} onToggle={() => setOpenGame(openGame === g.id ? "" : g.id)} />
                      ))}
                    </div>
                  ))
                ) : (
                  <p className="intl-empty">The playoff round hasn't started yet.</p>
                )}
              </div>
            ) : null}

            {tab === "rosters" ? (
              <div className="intl-rosters">
                <div className="intl-nations">
                  {teams.map((t) => (
                    <button key={t.code} type="button" className={t.code === nation ? "is-on" : ""} onClick={() => setNation(t.code)}>
                      <Flag nation={t} size={26} />
                      <span>{t.name}</span>
                      <em>{t.strength != null ? Math.round(t.strength) : ""}</em>
                    </button>
                  ))}
                </div>
                {team ? (
                  <table className="intl-table is-roster">
                    <caption>
                      <Flag nation={team} size={30} /> {team.name}
                      {data.projected ? " · projected" : ""}
                    </caption>
                    <thead>
                      <tr>
                        <th>Pos</th><th>Player</th><th>Club</th><th>Age</th><th>OVR</th>
                        {!data.projected ? (<><th>GP</th><th>G</th><th>A</th><th>PTS</th><th>SV%</th></>) : null}
                      </tr>
                    </thead>
                    <tbody>
                      {roster.map((r) => (
                        <tr key={r.id} className={`${r.is_user ? "is-user" : ""}${r.npc ? " is-npc" : ""}`}>
                          <td>{r.position || r.pos}</td>
                          <td><b>{r.name}</b></td>
                          <td>{r.club}</td>
                          <td>{r.age || ""}</td>
                          <td>{Math.round(r.ovr)}</td>
                          {!data.projected ? (
                            <>
                              <td>{r.gp ?? 0}</td>
                              <td>{r.pos === "G" ? "—" : r.g ?? 0}</td>
                              <td>{r.pos === "G" ? "—" : r.a ?? 0}</td>
                              <td>{r.pos === "G" ? `${r.w ?? 0}W` : r.pts ?? 0}</td>
                              <td>{r.pos === "G" && r.sv_pct ? r.sv_pct.toFixed(3).replace(/^0/, "") : "—"}</td>
                            </>
                          ) : null}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : null}
              </div>
            ) : null}

            {tab === "leaders" ? (
              <table className="intl-table">
                <caption>Scoring leaders</caption>
                <thead>
                  <tr><th>#</th><th>Player</th><th>Nation</th><th>Club</th><th>GP</th><th>G</th><th>A</th><th>PTS</th></tr>
                </thead>
                <tbody>
                  {(data.leaders || []).map((s, i) => (
                    <tr key={s.id} className={s.is_user ? "is-user" : ""}>
                      <td>{i + 1}</td>
                      <td><b>{s.name}</b></td>
                      <td className="intl-tn"><Flag nation={nat(s.code)} size={20} /> {s.code}</td>
                      <td>{s.club}</td>
                      <td>{s.gp}</td><td>{s.g}</td><td>{s.a}</td><td><b>{s.pts}</b></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
  return typeof document !== "undefined" ? createPortal(body, document.body) : body;
}

function GameRow({ g, nat, open, onToggle, compact = false }) {
  const homeWon = g.home_score > g.away_score;
  return (
    <div className={`intl-game${open ? " is-open" : ""}${compact ? " is-compact" : ""}`}>
      <button type="button" className="intl-game__row" onClick={onToggle}>
        <em>{g.label}{g.decided !== "REG" ? ` · ${g.decided}` : ""}</em>
        <span className={`intl-gt${homeWon ? " is-win" : ""}`}>
          <Flag nation={nat(g.home)} size={24} /> {nat(g.home).name}
        </span>
        <b>
          {g.home_score}–{g.away_score}
        </b>
        <span className={`intl-gt is-away${!homeWon ? " is-win" : ""}`}>
          {nat(g.away).name} <Flag nation={nat(g.away)} size={24} />
        </span>
      </button>
      {open ? (
        <div className="intl-game__box">
          {g.goals.map((x, i) => (
            <p key={i}>
              <Flag nation={nat(x.team)} size={16} /> <b>{x.scorer}</b>
              {x.assists.length ? <span> ({x.assists.join(", ")})</span> : <span> (unassisted)</span>}
            </p>
          ))}
          <small>
            Shots {g.shots?.[g.home] ?? "—"}–{g.shots?.[g.away] ?? "—"} · Goalies: {g.goalies?.[g.home]} / {g.goalies?.[g.away]}
          </small>
        </div>
      ) : null}
    </div>
  );
}
