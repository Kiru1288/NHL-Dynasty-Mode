import React, { useCallback, useMemo, useState } from "react";
import { buildWjcDraftStockRows, buildWjcStatLeaders, isWjcNpc } from "./wjcBroadcastBuilder";
import { GameResultModal, ProspectDetailModal } from "./WorldJuniorsBroadcastPanels";
import { wjcFlagUrl } from "../../utils/countryFlags";
import { asArray, resolveWorldJuniorsPayload } from "./wjcPayload";

/**
 * World Juniors · holiday rink village.
 * Advent-calendar days (Dec 26 – Jan 5), face-off cards for the day's games, a banner wall of
 * standings that flips to an ice bracket, your prospects as stockings on the mantel, and the
 * scoring race as sleds on a snowy track. Same props as the old WorldJuniorsMenu.
 */

const DAY_DATES = ["Dec 26", "Dec 27", "Dec 28", "Dec 29", "Dec 30", "Dec 31", "Jan 1", "Jan 2", "Jan 3", "Jan 4", "Jan 5"];
const DAY_TAG = ["Opening day", "", "", "", "New Year's Eve", "", "New Year's Day", "Quarterfinals", "", "Semifinals", "Medal day"];

function num(v, d = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

function Flag({ code, size = 40 }) {
  const src = code ? wjcFlagUrl(code, size >= 48 ? 64 : 48) : null;
  return src ? (
    <img className="wjv-flag" src={src} alt={code} width={size} height={Math.round(size * 0.72)} loading="lazy" />
  ) : (
    <span className="wjv-flag is-blank" style={{ width: size, height: Math.round(size * 0.72) }}>{code || "?"}</span>
  );
}

function collectGames(payload) {
  if (asArray(payload?.all_games).length) return asArray(payload.all_games);
  const p = payload?.playoffs || {};
  return [
    ...asArray(payload?.round_robin_games),
    ...asArray(p.quarterfinals),
    ...asArray(p.semifinals),
    ...(p.bronze ? [p.bronze] : []),
    ...(p.gold ? [p.gold] : []),
  ];
}

const played = (g) => g?.home_goals != null && g?.away_goals != null;
const code3 = (g, side) => String(g?.[side] || g?.[`${side}_label`] || "?").slice(0, 3).toUpperCase();

export default function WorldJuniorsRevamp({ franchiseState, eventData, onClose, onBackToHub, onSimNextTournamentDay, onOpenDraftBoard }) {
  const payload = useMemo(() => resolveWorldJuniorsPayload(franchiseState, eventData), [franchiseState, eventData]);
  const games = useMemo(() => collectGames(payload), [payload]);
  const stock = useMemo(() => buildWjcDraftStockRows(payload, franchiseState), [payload, franchiseState]);
  const leaders = useMemo(() => buildWjcStatLeaders(payload), [payload]);
  const labelOf = useCallback(
    (code) => asArray(payload.countries).find((c) => String(c?.code) === String(code))?.label || code,
    [payload.countries]
  );

  // game_day may be 0- or 1-based depending on the bundle; normalise to door index 0..10.
  const dayBase = useMemo(() => {
    const ds = games.map((g) => num(g.game_day, -1)).filter((d) => d >= 0);
    return ds.length && !ds.includes(0) ? 1 : 0;
  }, [games]);
  const doorOf = useCallback((g) => Math.max(0, Math.min(10, num(g.game_day, 0) - dayBase)), [dayBase]);
  const today = Math.max(0, Math.min(10, num(payload.wjc_day, 1) - 1));
  const [door, setDoor] = useState(null);
  const activeDoor = door == null ? today : door;
  const dayGames = games.filter((g) => doorOf(g) === activeDoor);

  const [wall, setWall] = useState("banners");
  const [mantel, setMantel] = useState("mine");
  const [game, setGame] = useState(null);
  const [prospect, setProspect] = useState(null);
  const [busy, setBusy] = useState(false);

  const standings = [...asArray(payload.standings)].sort((a, b) => num(b.pts) - num(a.pts) || num(b.gf) - num(b.ga) - (num(a.gf) - num(a.ga)));
  const mine = stock.filter((r) => r.is_user_prospect && !isWjcNpc(r));
  const risers = [...stock].filter((r) => !isWjcNpc(r) && r.stock_delta != null).sort((a, b) => num(b.stock_delta) - num(a.stock_delta)).slice(0, 8);
  const mantelRows = mantel === "mine" ? mine : risers;
  const scorers = asArray(leaders?.byPts).slice(0, 6);
  const topPts = Math.max(1, ...scorers.map((s) => num(s.pts)));
  const playoffs = payload.playoffs || {};
  const medals = payload.medals_final ? payload.medal_labels || {} : null;
  const complete = Boolean(payload.medals_final);

  const leave = () => (onClose ? onClose() : onBackToHub?.());
  const sim = async () => {
    if (busy || typeof onSimNextTournamentDay !== "function") return;
    setBusy(true);
    try {
      await onSimNextTournamentDay();
      setDoor(null);
    } finally {
      setBusy(false);
    }
  };
  const openProspect = (row) => {
    if (!row || isWjcNpc(row)) return;
    setProspect(row);
  };
  const statsFor = (row) => asArray(payload.player_stats).find((p) => String(p?.player_id) === String(row?.player_id));

  return (
    <section className="wjv" aria-label="World Juniors">
      <div className="wjv-snow" aria-hidden="true" />
      <div className="wjv-lights" aria-hidden="true">
        {Array.from({ length: 28 }, (_, i) => <i key={i} />)}
      </div>

      <header className="wjv-top">
        <div className="wjv-marquee">
          <span className="wjv-kicker">IIHF U20 · Boxing Day to January 5</span>
          <h1>World Juniors</h1>
          <span className="wjv-sub">
            {payload.isPreTournament ? "The puck drops on Boxing Day" : complete ? "Tournament complete" : `Day ${activeDoor + 1} of 11 · ${DAY_DATES[today]}`}
          </span>
        </div>
        <div className="wjv-actions">
          {typeof onOpenDraftBoard === "function" ? (
            <button type="button" className="wjv-btn" onClick={() => onOpenDraftBoard()}>Draft board</button>
          ) : null}
          {typeof onSimNextTournamentDay === "function" && !complete ? (
            <button type="button" className="wjv-btn is-go" disabled={busy} onClick={sim}>{busy ? "Skating…" : "Sim next day"}</button>
          ) : null}
          <button type="button" className="wjv-btn is-back" onClick={leave}>Back</button>
        </div>
      </header>

      <nav className="wjv-advent" aria-label="Tournament days">
        {DAY_DATES.map((d, i) => {
          const gs = games.filter((g) => doorOf(g) === i);
          const done = gs.length > 0 && gs.every(played);
          const state = i === activeDoor ? "is-on" : done || i < today ? "is-open" : i === today ? "is-today" : "is-locked";
          return (
            <button key={d} type="button" className={`wjv-door ${state}${i === today ? " is-now" : ""}`} onClick={() => setDoor(i)} disabled={state === "is-locked" && !gs.length}>
              <b>{i + 1}</b>
              <span>{d}</span>
              <em>{DAY_TAG[i] || (gs.length ? `${gs.length} games` : "")}</em>
            </button>
          );
        })}
      </nav>

      <div className="wjv-village">
        <aside className="wjv-wall">
          <div className="wjv-switch">
            <button type="button" className={wall === "banners" ? "is-on" : ""} onClick={() => setWall("banners")}>Banner wall</button>
            <button type="button" className={wall === "bracket" ? "is-on" : ""} onClick={() => setWall("bracket")}>Ice bracket</button>
          </div>
          {wall === "banners" ? (
            <div className="wjv-banners">
              {standings.length ? (
                standings.map((r, i) => (
                  <div key={r.code} className={`wjv-pennant${i < 8 ? " is-in" : ""}`} style={{ "--i": i }}>
                    <Flag code={r.code} size={34} />
                    <b>{r.label || labelOf(r.code)}</b>
                    <span>{num(r.w)}-{num(r.l)}</span>
                    <strong>{num(r.pts)}</strong>
                  </div>
                ))
              ) : (
                <p className="wjv-empty">Banners go up when the first games are played.</p>
              )}
            </div>
          ) : (
            <div className="wjv-bracket">
              {[["Quarterfinals", asArray(playoffs.quarterfinals)], ["Semifinals", asArray(playoffs.semifinals)], ["Bronze", playoffs.bronze ? [playoffs.bronze] : []], ["Gold", playoffs.gold ? [playoffs.gold] : []]].map(([title, list]) => (
                <div key={title} className={`wjv-round is-${title.toLowerCase()}`}>
                  <span className="wjv-lab">{title}</span>
                  {list.length ? (
                    list.map((g, k) => (
                      <button key={k} type="button" className="wjv-slot" onClick={() => setGame(g)}>
                        <span className={played(g) && num(g.home_goals) > num(g.away_goals) ? "is-win" : ""}><Flag code={g.home} size={22} /> {code3(g, "home")} <b>{g.home_goals ?? ""}</b></span>
                        <span className={played(g) && num(g.away_goals) > num(g.home_goals) ? "is-win" : ""}><Flag code={g.away} size={22} /> {code3(g, "away")} <b>{g.away_goals ?? ""}</b></span>
                      </button>
                    ))
                  ) : (
                    <p className="wjv-empty">To be decided</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </aside>

        <main className="wjv-rink">
          <div className="wjv-rink-head">
            <span className="wjv-lab">{activeDoor === today && !complete ? "Tonight at the rink" : DAY_DATES[activeDoor]}</span>
            <h2>{DAY_TAG[activeDoor] || "Preliminary round"}</h2>
          </div>
          {medals ? (
            <div className="wjv-podium">
              {[["silver", medals.silver], ["gold", medals.gold], ["bronze", medals.bronze]].map(([tier, label]) => (
                <div key={tier} className={`wjv-step is-${tier}`}>
                  <b>{label || "—"}</b>
                  <span>{tier}</span>
                </div>
              ))}
            </div>
          ) : null}
          <div className="wjv-faceoffs">
            {dayGames.length ? (
              dayGames.map((g, k) => {
                const done = played(g);
                const hw = done && num(g.home_goals) > num(g.away_goals);
                return (
                  <button key={k} type="button" className={`wjv-faceoff${done ? " is-final" : ""}`} onClick={() => done && setGame(g)}>
                    <span className="wjv-round-tag">{g.round || "Preliminary round"}</span>
                    <div className={`wjv-side${hw ? " is-win" : ""}`}>
                      <Flag code={g.home} size={64} />
                      <b>{g.home_label || labelOf(g.home)}</b>
                    </div>
                    <div className="wjv-jumbo">
                      {done ? (
                        <>
                          <strong>{g.home_goals}</strong>
                          <i>:</i>
                          <strong>{g.away_goals}</strong>
                        </>
                      ) : (
                        <em>VS</em>
                      )}
                      <small>{done ? (g.overtime || g.ot ? "Final · OT" : "Final") : "Puck drop tonight"}</small>
                    </div>
                    <div className={`wjv-side is-away${done && !hw ? " is-win" : ""}`}>
                      <Flag code={g.away} size={64} />
                      <b>{g.away_label || labelOf(g.away)}</b>
                    </div>
                  </button>
                );
              })
            ) : (
              <p className="wjv-empty is-big">{payload.isPreTournament ? "Doors open on Boxing Day. Sim to the opening night." : "No games behind this door yet."}</p>
            )}
          </div>
        </main>

        <aside className="wjv-mantel">
          <div className="wjv-switch">
            <button type="button" className={mantel === "mine" ? "is-on" : ""} onClick={() => setMantel("mine")}>Your prospects</button>
            <button type="button" className={mantel === "risers" ? "is-on" : ""} onClick={() => setMantel("risers")}>Draft risers</button>
          </div>
          <div className="wjv-stockings">
            {mantelRows.length ? (
              mantelRows.map((r) => {
                const d = num(r.stock_delta, 0);
                return (
                  <button key={r.player_id} type="button" className="wjv-stocking" onClick={() => openProspect(r)}>
                    <Flag code={r.wjc_country} size={26} />
                    <b>{r.name}</b>
                    <span>{r.position} · {r.wjc_country_label || labelOf(r.wjc_country)}</span>
                    <em>{num(r.tournament_gp)} GP · {num(r.tournament_g)}G {Math.max(0, num(r.tournament_pts) - num(r.tournament_g))}A</em>
                    {r.stock_delta != null ? (
                      <strong className={d > 0 ? "is-up" : d < 0 ? "is-down" : ""}>{d > 0 ? `▲ ${d}` : d < 0 ? `▼ ${Math.abs(d)}` : "＝"} on the board</strong>
                    ) : r.skill_bump ? (
                      <strong className={num(r.skill_bump) >= 0 ? "is-up" : "is-down"}>{num(r.skill_bump) >= 0 ? "+" : ""}{num(r.skill_bump).toFixed(1)} skills</strong>
                    ) : null}
                  </button>
                );
              })
            ) : (
              <p className="wjv-empty">{mantel === "mine" ? "None of your prospects made a national team." : "Draft stock moves once games are played."}</p>
            )}
          </div>
        </aside>
      </div>

      <footer className="wjv-sled">
        <span className="wjv-lab">Scoring race</span>
        <div className="wjv-track">
          {scorers.length ? (
            scorers.map((s, i) => (
              <div key={s.player_id || i} className="wjv-lane">
                <span className="wjv-lane-name"><Flag code={s.country || s.wjc_country} size={18} /> {s.name}</span>
                <i className="wjv-lane-track">
                  <b className="wjv-sledge" style={{ left: `${Math.max(2, (num(s.pts) / topPts) * 92)}%` }}>🛷</b>
                </i>
                <strong>{num(s.pts)} PTS</strong>
              </div>
            ))
          ) : (
            <p className="wjv-empty">The race starts with the first goal.</p>
          )}
        </div>
      </footer>

      <GameResultModal game={game} onClose={() => setGame(null)} formatScoreLine={(g) => `${code3(g, "home")} ${g?.home_goals ?? ""} — ${code3(g, "away")} ${g?.away_goals ?? ""}`} gameCode={code3} />
      <ProspectDetailModal
        prospect={prospect}
        tournamentStats={prospect ? statsFor(prospect) : null}
        franchiseState={franchiseState}
        onClose={() => setProspect(null)}
        onOpenDraftBoard={onOpenDraftBoard}
      />
    </section>
  );
}
