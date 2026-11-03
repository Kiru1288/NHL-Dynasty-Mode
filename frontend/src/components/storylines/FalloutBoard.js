import React, { useMemo, useState } from "react";

/* Fallout: league discipline and off-ice consequences, split into
   "your club" and "around the league", with names instead of raw ids.
   Every field is optional; older saves without enrichment still render. */

const arr = (v) => (Array.isArray(v) ? v : []);
const obj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : {});
const s = (v) => (v == null ? "" : String(v));
const money = (v) => {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? `$${n.toFixed(2)}M` : "";
};
const label = (key) => s(key).replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());

const STATUS_COPY = {
  suspended: "Suspended",
  banned: "Banned",
  investigative_leave: "Investigative leave",
  personal_leave: "Personal leave",
  deceased: "Deceased",
  unavailable: "Unavailable",
};

function teamName(row, teamsById) {
  if (row?.team_name) return s(row.team_name);
  const t = teamsById[s(row?.team_id)];
  return t ? s(t.name || t.abbr || row.team_id) : s(row?.team_id || "Unknown club");
}

function FalloutCard({ tone = "", kicker, title, children, stamp }) {
  return (
    <article className={`ccx-panel fo-card ${tone ? `is-${tone}` : ""}`}>
      <header className="fo-card__head">
        <span className="ccx-kicker">{kicker}</span>
        {stamp ? <span className={`ccx-stamp ${tone ? `is-${tone}` : ""}`}>{stamp}</span> : null}
      </header>
      <h4 className="fo-card__title">{title}</h4>
      <div className="fo-card__body">{children}</div>
    </article>
  );
}

export default function FalloutBoard({ narrativeUniverse, stories = [], onOpenStory, franchiseState }) {
  const [scope, setScope] = useState("club");
  const userTid = s(franchiseState?.user_team_id ?? franchiseState?.team?.id ?? "");
  const teamsById = useMemo(() => {
    const out = {};
    for (const o of arr(franchiseState?.roster_browser?.organizations)) {
      out[s(o?.team_id)] = { name: o?.name, abbr: o?.abbr };
    }
    return out;
  }, [franchiseState?.roster_browser?.organizations]);

  const nu = obj(narrativeUniverse);
  const isMine = (row) => (row?.is_user != null ? Boolean(row.is_user) : !!userTid && s(row?.team_id) === userTid);

  const sanctions = arr(nu.team_sanctions).filter((x) => x && x.active !== false);
  const picks = arr(nu.forfeited_picks).filter(Boolean);
  const capRows = arr(nu.cap_penalty_rows).length
    ? arr(nu.cap_penalty_rows)
    : Object.entries(obj(nu.cap_penalties))
        .map(([team_id, amt]) => ({ team_id, amount_m: typeof amt === "object" && amt ? Number(amt.cap_penalty_m || amt.amount || 0) : Number(amt) }))
        .filter((r) => Number(r.amount_m) > 0);
  const availability = Object.entries(obj(nu.player_availability))
    .map(([pid, row]) => ({ pid, ...obj(row) }))
    .filter((r) => r.active !== false && s(r.status) !== "available");
  const promises = obj(obj(nu.player_meetings).promises);
  const broken = arr(promises.broken).slice(-6).reverse();
  const budget = obj(nu.major_event_budget);

  const pick = (rows) => rows.filter((r) => (scope === "club" ? isMine(r) : !isMine(r)));
  const mySanctions = pick(sanctions);
  const myPicks = pick(picks);
  const myCaps = pick(capRows);
  const myOut = pick(availability.filter((r) => r.team_id || r.team_name)).concat(
    scope === "club" ? availability.filter((r) => !r.team_id && !r.team_name) : []
  );
  const myBroken = scope === "club" ? broken : [];

  const clubCount = sanctions.filter(isMine).length + capRows.filter(isMine).length + availability.filter(isMine).length + broken.length;
  const leagueCount = sanctions.filter((r) => !isMine(r)).length + availability.filter((r) => (r.team_id || r.team_name) && !isMine(r)).length;
  const nothing = !mySanctions.length && !myPicks.length && !myCaps.length && !myOut.length && !myBroken.length;

  const linkedStory = (eventId) =>
    eventId ? stories.find((st) => s(st?.causeEventId) === s(eventId) || s(st?.raw?.cause_event_id) === s(eventId)) : null;

  return (
    <div className="fo-board">
      <header className="ccx-head fo-board__head">
        <div>
          <span className="ccx-kicker">League office</span>
          <h3 className="ccx-title">Fallout</h3>
          <p className="fo-board__lede">
            Fines, cap penalties, forfeited picks, suspensions and broken promises: the bill for what happened off the ice.
          </p>
        </div>
        <div className="ccx-tabs" role="tablist" aria-label="Fallout scope">
          {[["club", "Your club", clubCount], ["league", "Around the league", leagueCount]].map(([id, text, n]) => (
            <button key={id} type="button" role="tab" aria-selected={scope === id} className={`ccx-tab ${scope === id ? "is-active" : ""}`} onClick={() => setScope(id)}>
              {text}
              {n ? <span className="ccx-tab__count">{n}</span> : null}
            </button>
          ))}
        </div>
      </header>

      {budget.target ? (
        <p className="fo-board__meter">
          Major league incidents this season <b>{Number(budget.generated) || 0}</b> / {Number(budget.target)}
        </p>
      ) : null}

      {nothing ? (
        <div className="ccx-panel fo-empty">
          <span className="ccx-stamp is-good">Clean sheet</span>
          <h4>{scope === "club" ? "Nothing hanging over your club" : "The league office is quiet"}</h4>
          <p>
            {scope === "club"
              ? "No fines, cap penalties, forfeited picks, suspensions or broken promises right now. Keep it that way."
              : "No other club is serving a sanction at the moment."}
          </p>
        </div>
      ) : (
        <div className="fo-grid">
          {mySanctions.map((x) => {
            const story = linkedStory(x.source_event_id);
            const xp = arr(x.forfeited_picks);
            return (
              <FalloutCard key={s(x.id)} tone="bad" kicker={scope === "club" ? "Sanction" : teamName(x, teamsById)} title={label(x.event_type || "League sanction")} stamp="Active">
                <ul className="fo-facts">
                  {x.season ? <li><span>Season</span><b>{s(x.season)}</b></li> : null}
                  {money(x.fine_m) ? <li><span>Fine</span><b>{money(x.fine_m)}</b></li> : null}
                  {money(x.cap_penalty_m) ? <li><span>Cap penalty</span><b>{money(x.cap_penalty_m)}</b></li> : null}
                  {xp.length ? <li><span>Picks lost</span><b>{xp.map((p) => `${s(p.draft_year)} R${s(p.round)}`).join(", ")}</b></li> : null}
                </ul>
                {story && onOpenStory ? (
                  <button type="button" className="ccx-btn fo-link" onClick={() => onOpenStory(story.id)}>Open the story</button>
                ) : null}
              </FalloutCard>
            );
          })}

          {myCaps.length ? (
            <FalloutCard tone="warn" kicker="Cap ledger" title="Cap penalties" stamp={`${myCaps.length}`}>
              <ul className="fo-facts">
                {myCaps.map((r) => (
                  <li key={s(r.team_id)}><span>{teamName(r, teamsById)}</span><b>{money(r.amount_m) || "—"}</b></li>
                ))}
              </ul>
            </FalloutCard>
          ) : null}

          {myPicks.length ? (
            <FalloutCard tone="warn" kicker="Draft table" title="Forfeited picks" stamp={`${myPicks.length}`}>
              <ul className="fo-facts">
                {myPicks.map((p, i) => (
                  <li key={s(p.pick_id || `${p.team_id}-${p.draft_year}-${p.round}-${i}`)}>
                    <span>{scope === "club" ? "Pick" : teamName(p, teamsById)}</span>
                    <b>{s(p.draft_year)} round {s(p.round)}</b>
                  </li>
                ))}
              </ul>
            </FalloutCard>
          ) : null}

          {myOut.length ? (
            <FalloutCard tone="bad" kicker="Off-ice" title="Unavailable players" stamp={`${myOut.length}`}>
              <ul className="fo-facts">
                {myOut.map((r) => {
                  const left = [Number(r.games_remaining) > 0 ? `${Number(r.games_remaining)} gm` : "", Number(r.days_remaining) > 0 ? `${Number(r.days_remaining)} d` : ""].filter(Boolean).join(" · ");
                  return (
                    <li key={r.pid}>
                      <span>
                        <b className="fo-name">{s(r.player_name) || "Unnamed player"}</b>
                        <small>{STATUS_COPY[s(r.status)] || label(r.status || "unavailable")}{r.reason_public || r.reason ? ` · ${s(r.reason_public || r.reason).replace(/_/g, " ")}` : ""}{scope === "league" ? ` · ${teamName(r, teamsById)}` : ""}</small>
                      </span>
                      <b>{left || "—"}</b>
                    </li>
                  );
                })}
              </ul>
            </FalloutCard>
          ) : null}

          {myBroken.length ? (
            <FalloutCard tone="warn" kicker="Locker room" title="Broken promises" stamp={`${myBroken.length}`}>
              <ul className="fo-facts">
                {myBroken.map((p, i) => (
                  <li key={s(p.id || i)}>
                    <span>
                      <b className="fo-name">{s(p.player_name || "Player")}</b>
                      <small>{s(p.description || label(p.type || "promise"))}</small>
                    </span>
                  </li>
                ))}
              </ul>
              <p className="fo-hint">Trust and morale took the hit. A "repair relationship" meeting can win some of it back.</p>
            </FalloutCard>
          ) : null}
        </div>
      )}
    </div>
  );
}
